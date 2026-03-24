"""
fl/server.py — FL Server: aggregation, validation, and gradient computation.

The server holds the global model and performs:
1. FedAvg aggregation of client weight updates (Δw_c)
2. Validation/test evaluation
3. Gradient computation ∇ℓ(w_t, D_val) for Shapley value calculation
4. Hessian-vector products H(D_val) @ v for second-order Shapley terms

Key formula:
    FedAvg: w_{t+1} = w_t - Σ_c (weight_c * Δw_c)
    where Δw_c = w_local_c - w_global (negative of the usual update convention)
"""

import copy
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader


class Server:
    """Federated learning server managing the global model and aggregation.

    Attributes:
        global_model: The shared global model.
        val_loader: Validation DataLoader for gradient and HVP computation.
        device: Compute device string ('cuda' or 'cpu').
        args: Parsed experiment arguments.
        _criterion: CrossEntropyLoss for computing validation gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
        args: argparse.Namespace,
    ) -> None:
        """Initialize server with a model and validation data.

        Args:
            model: Initial global model (will be moved to device).
            val_loader: DataLoader for the validation set.
            device: Target compute device ('cuda' or 'cpu').
            args: Experiment arguments (used for configuration flags).
        """
        self.global_model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.args = args
        self._criterion = nn.CrossEntropyLoss()

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
    ) -> None:
        """Apply FedAvg aggregation: w_{t+1} = w_t - Σ_c weight_c * Δw_c.

        Args:
            client_updates: List of state_dict diffs, each Δw_c = w_local - w_global.
                Keys are parameter names; values are CPU tensors.
            client_weights: Per-client aggregation weights (sum to 1).
                Equal weights = 1/|C_t|; data-proportional weights = n_c/Σn.

        Notes:
            Updates are applied in-place to global_model parameters.
            The aggregation uses the convention that positive Δw_c increases
            the parameter value, so we subtract the weighted sum.
        """
        assert len(client_updates) == len(client_weights), \
            "Number of updates must match number of weights."
        assert abs(sum(client_weights) - 1.0) < 1e-5, \
            f"Weights must sum to 1, got {sum(client_weights):.6f}"

        global_state = self.global_model.state_dict()

        with torch.no_grad():
            for name, param in global_state.items():
                # Accumulate weighted Δw sum
                weighted_delta = torch.zeros_like(param, dtype=torch.float32)
                for update, w in zip(client_updates, client_weights):
                    if name in update:
                        # Move to same device as param
                        delta = update[name].to(param.device).float()
                        weighted_delta += w * delta

                # w_{t+1} = w_t + Σ weight_c * Δw_c
                # (Δw_c = w_local - w_global, so adding moves toward clients)
                global_state[name] = param.float() + weighted_delta

        self.global_model.load_state_dict(global_state)

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of the current global model state_dict.

        Returns:
            Deep-copied state_dict with all tensors on CPU.
        """
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def get_trainable_param_names(self) -> List[str]:
        """Return names of all trainable (requires_grad) parameters.

        Returns:
            Sorted list of parameter name strings.

        Notes:
            This defines the canonical parameter subset used for gradient
            dot-products and Shapley computation.  BatchNorm running stats
            (running_mean, running_var, num_batches_tracked) are excluded.
        """
        return [n for n, p in self.global_model.named_parameters() if p.requires_grad]

    def compute_validation_gradient(self) -> torch.Tensor:
        """Compute the flattened validation gradient ∇ℓ(w_t, D_val).

        Averages the cross-entropy loss over all validation batches,
        then computes gradients w.r.t. all **trainable** model parameters.

        Returns:
            1D tensor: concatenation of trainable parameter gradients, shape (P,)
            where P = total number of trainable parameters.

        Notes:
            - Only parameters with requires_grad=True are included.
            - BatchNorm running stats are excluded (consistent with
              flatten_update_for_shapley).
        """
        self.global_model.train()
        self.global_model.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = 0

        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.global_model(X)
            loss = self._criterion(logits, y) * X.size(0)
            total_loss = total_loss + loss
            total_samples += X.size(0)

        avg_loss = total_loss / total_samples
        avg_loss.backward()

        # Concatenate only trainable parameter gradients
        grad_parts = []
        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_parts.append(param.grad.detach().flatten())
                else:
                    grad_parts.append(torch.zeros(param.numel(), device=self.device))

        self.global_model.zero_grad()
        return torch.cat(grad_parts)

    def compute_validation_hessian_vector_product(
        self, vector: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Hessian-vector product H^(D_val) @ vector.

        Uses double backpropagation: computes Hv without materializing H.

        Formula:
            H @ v = ∂/∂w [∇ℓ(w, D_val)ᵀ @ v]

        Args:
            vector: Flat parameter vector v of shape (P,), same device as model.

        Returns:
            Flat tensor Hv of shape (P,), detached from computation graph.

        Notes:
            - Requires create_graph=True on the first backward pass.
            - This operation is O(2 × backward pass) and memory-intensive.
              Only use when use_second_order=True.
            - Vector is normalized to the model's device automatically.
        """
        vector = vector.to(self.device)
        params = [p for p in self.global_model.parameters() if p.requires_grad]

        self.global_model.train()
        self.global_model.zero_grad()

        # --- Step 1: Forward pass with graph retained ---
        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = 0

        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.global_model(X)
            loss = self._criterion(logits, y) * X.size(0)
            total_loss = total_loss + loss
            total_samples += X.size(0)

        avg_loss = total_loss / total_samples

        # --- Step 2: First backward with create_graph=True ---
        # ∇ℓ w.r.t. each parameter, retaining graph for second backward
        grads = autograd.grad(avg_loss, params, create_graph=True)

        # --- Step 3: Flatten gradient and compute dot product with v ---
        # grad_flat · v = Σ_i grad_i · v_i
        grad_flat = torch.cat([g.reshape(-1) for g in grads])
        dot_product = (grad_flat * vector).sum()

        # --- Step 4: Second backward to get Hv = ∂(grad·v)/∂w ---
        hvp_grads = autograd.grad(dot_product, params, retain_graph=False)

        hvp_flat = torch.cat([h.reshape(-1).detach() for h in hvp_grads])

        self.global_model.zero_grad()
        return hvp_flat

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model on validation and test sets.

        Args:
            test_loader: DataLoader for the held-out test set.

        Returns:
            Dict with keys: 'val_loss', 'val_acc', 'test_loss', 'test_acc'.
            All values are Python floats.
        """
        self.global_model.eval()

        def _eval_loader(loader: DataLoader) -> Tuple[float, float]:
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.global_model(X)
                    loss = self._criterion(logits, y)
                    total_loss += loss.item() * X.size(0)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += X.size(0)
            return total_loss / max(total, 1), correct / max(total, 1)

        val_loss, val_acc = _eval_loader(self.val_loader)
        test_loss, test_acc = _eval_loader(test_loader)

        return {
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
