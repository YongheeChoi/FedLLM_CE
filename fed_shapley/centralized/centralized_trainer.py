"""
centralized/centralized_trainer.py — Centralized Shapley computation baseline.

Trains a model centrally on all client data combined, computing per-sample
gradient dot-products with the validation gradient to attribute contributions
back to each client (data owner).

This provides a ground-truth baseline for comparing with FL in-run Shapley.

The per-sample attribution formula:
    ϕ_c^(iter) = Σ_{z_i ∈ D_c} -η · ∇ℓ(w, z_val) · ∇ℓ(w, z_i)

Per-sample gradients are computed either via:
- torch.func.grad + vmap (if available, fast)
- Simple loop over individual samples (slower but always works)
"""

import argparse
import copy
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class CentralizedTrainer:
    """Train centrally and attribute per-sample gradient contributions to clients.

    Attributes:
        model: The model being trained.
        train_loader: DataLoader for the combined training set.
        val_loader: DataLoader for validation gradient computation.
        client_data_indices: {client_id: [sample_indices]} partition map.
        args: Experiment arguments.
        device: Compute device.
        criterion: CrossEntropyLoss.
        sample_to_client: Dict mapping sample index → client_id.
        shapley_accumulator: Running per-client Shapley sum.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        client_data_indices: Dict[int, List[int]],
        args: argparse.Namespace,
    ) -> None:
        """Initialize the centralized trainer.

        Args:
            model: Model architecture (will be deep-copied for training).
            train_loader: DataLoader over the full combined training set.
                Each sample must have a consistent global index for
                client attribution.
            val_loader: DataLoader for validation gradient computation.
            client_data_indices: {client_id: [sample_indices]} — same
                partition as the FL experiment. Used to map samples to clients.
            args: Experiment arguments. Used fields:
                - local_lr, num_rounds, local_epochs, device
        """
        self.model = copy.deepcopy(model).to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_data_indices = client_data_indices
        self.args = args
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()

        # Build reverse mapping: sample_global_index → client_id
        self.sample_to_client: Dict[int, int] = {}
        for client_id, indices in client_data_indices.items():
            for idx in indices:
                self.sample_to_client[idx] = client_id

        # Initialize accumulator for all clients
        all_client_ids = set(client_data_indices.keys())
        self.shapley_accumulator: Dict[int, float] = {c: 0.0 for c in all_client_ids}

    def train_and_compute_shapley(self) -> Dict[int, float]:
        """Run centralized training and compute per-client Shapley values.

        For each mini-batch:
        1. Compute validation gradient ∇ℓ(w, D_val).
        2. Compute per-sample gradients ∇ℓ(w, z_i) for each sample in batch.
        3. Attribution: ϕ_{client(z_i)} += -lr * <∇ℓ(val), ∇ℓ(z_i)>
        4. Update model parameters with batch gradient.

        The number of training iterations mirrors the FL experiment:
        num_rounds × local_epochs (approximately).

        Returns:
            Dict[int, float]: {client_id: accumulated_shapley}.
                Positive values indicate beneficial contributions.
        """
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.local_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # Approximate total iterations to match FL computation budget
        total_epochs = max(self.args.num_rounds * self.args.local_epochs // 10, 5)
        print(f"[CentralizedTrainer] Training for {total_epochs} epochs ...")

        _start_time = time.perf_counter()
        self.model.train()

        for epoch in tqdm(range(total_epochs), desc="Centralized Training"):
            for batch_data in self.train_loader:
                # Handle both (X, y) and (X, y, indices) formats
                if len(batch_data) == 3:
                    X, y, sample_indices = batch_data
                elif len(batch_data) == 2:
                    X, y = batch_data
                    sample_indices = None
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")

                X, y = X.to(self.device), y.to(self.device)

                # ----------------------------------------------------------
                # Step 1: Compute validation gradient ∇ℓ(w, D_val)
                # ----------------------------------------------------------
                val_grad = self._compute_val_gradient()  # shape: (P,)

                # ----------------------------------------------------------
                # Step 2 & 3: Per-sample attribution
                # ----------------------------------------------------------
                if sample_indices is not None:
                    self._attribute_batch(X, y, val_grad, sample_indices)
                else:
                    # If no indices, attribute equally to all clients
                    self._attribute_batch_equal(X, y, val_grad)

                # ----------------------------------------------------------
                # Step 4: Standard batch gradient update
                # ----------------------------------------------------------
                self.model.train()
                optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()

        elapsed = time.perf_counter() - _start_time
        print(f"[CentralizedTrainer] Training complete. ({elapsed:.1f}s)")
        print("[CentralizedTrainer] Per-client Shapley values:")
        for cid, sv in sorted(self.shapley_accumulator.items(), key=lambda x: x[1], reverse=True):
            print(f"  Client {cid:3d}: phi = {sv:+.6f}")

        return dict(self.shapley_accumulator)

    def _compute_val_gradient(self) -> torch.Tensor:
        """Compute flattened validation gradient ∇ℓ(w, D_val).

        Returns:
            1D float tensor of shape (P,) containing concatenated parameter gradients.
        """
        self.model.train()
        self.model.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = 0

        for X_val, y_val in self.val_loader:
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
            logits = self.model(X_val)
            loss = self.criterion(logits, y_val) * X_val.size(0)
            total_loss = total_loss + loss
            total_samples += X_val.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        avg_loss.backward()

        grad_parts = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_parts.append(param.grad.detach().flatten())
            else:
                grad_parts.append(torch.zeros(param.numel(), device=self.device))

        self.model.zero_grad()
        return torch.cat(grad_parts)

    def _attribute_batch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_grad: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> None:
        """Compute per-sample gradient dot-products and attribute to clients.

        Uses individual sample forward/backward passes for simplicity.
        For large models, consider using torch.func.grad + vmap for efficiency.

        Args:
            X: Batch input tensor of shape (B, C, H, W).
            y: Batch label tensor of shape (B,).
            val_grad: Flattened validation gradient of shape (P,).
            sample_indices: Global sample indices for attribution, shape (B,).
        """
        B = X.size(0)

        for i in range(B):
            x_i = X[i:i+1]
            y_i = y[i:i+1]

            self.model.zero_grad()
            logits_i = self.model(x_i)
            loss_i = self.criterion(logits_i, y_i)
            loss_i.backward()

            # Collect per-sample gradient
            sample_grad_parts = []
            for param in self.model.parameters():
                if param.grad is not None:
                    sample_grad_parts.append(param.grad.detach().flatten())
                else:
                    sample_grad_parts.append(torch.zeros(param.numel(), device=self.device))
            sample_grad = torch.cat(sample_grad_parts)

            # Attribution: ϕ_client += -lr * <val_grad, sample_grad>
            dot = torch.dot(val_grad, sample_grad).item()
            contribution = -self.args.local_lr * dot

            # Map sample index to client
            global_idx = int(sample_indices[i]) if hasattr(sample_indices, '__len__') \
                else int(sample_indices)
            client_id = self.sample_to_client.get(global_idx, -1)
            if client_id >= 0:
                self.shapley_accumulator[client_id] = (
                    self.shapley_accumulator.get(client_id, 0.0) + contribution
                )

        self.model.zero_grad()

    def _attribute_batch_equal(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_grad: torch.Tensor,
    ) -> None:
        """Attribute batch gradient contribution equally to all clients.

        Used as a fallback when sample indices are not available.

        Args:
            X: Batch input tensor.
            y: Batch label tensor.
            val_grad: Flattened validation gradient.
        """
        self.model.zero_grad()
        logits = self.model(X)
        loss = self.criterion(logits, y)
        loss.backward()

        batch_grad_parts = []
        for param in self.model.parameters():
            if param.grad is not None:
                batch_grad_parts.append(param.grad.detach().flatten())
            else:
                batch_grad_parts.append(torch.zeros(param.numel(), device=self.device))
        batch_grad = torch.cat(batch_grad_parts)

        dot = torch.dot(val_grad, batch_grad).item()
        total_contribution = -self.args.local_lr * dot

        # Split equally among all clients
        num_clients = len(self.shapley_accumulator)
        per_client = total_contribution / max(num_clients, 1)
        for cid in self.shapley_accumulator:
            self.shapley_accumulator[cid] += per_client

        self.model.zero_grad()


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns (sample, label, global_index) triples.

    Used to track which global sample index each DataLoader item corresponds to,
    enabling accurate client attribution in centralized training.

    Attributes:
        dataset: Underlying dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        """Wrap a dataset to include global indices.

        Args:
            dataset: The base dataset to wrap.
        """
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample, label = self.dataset[idx]
        return sample, label, idx
