"""
shapley/exact_shapley.py — Exact Shapley value computation via subset enumeration.

Computes the ground-truth Shapley value by evaluating utility for all 2^n subsets
of clients. Exponential complexity; only feasible for n ≤ 10.

The utility function is:
    U(S) = val_loss(w_t - η Σ_{c∈S} Δw_c) - val_loss(w_t)

Exact Shapley formula:
    phi_c = (1/n) * sum over S subset of [n] not containing c of [U(S+c) - U(S)] / C(n-1, |S|)

where C(n-1, k) is the binomial coefficient "n-1 choose k".
"""

import argparse
import copy
import itertools
import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ExactShapley:
    """Exact Shapley value computation via 2^n subset enumeration.

    Only use for small numbers of clients (≤ 10) due to exponential cost.

    Attributes:
        server: FL Server for model state access and validation loss computation.
        cumulative: Running sum of exact Shapley values per client.
        round_history: Per-round Shapley value snapshots.
    """

    def __init__(self, server: "Server", args: argparse.Namespace) -> None:
        """Initialize exact Shapley calculator.

        Args:
            server: FL Server instance. Must expose global_model, val_loader,
                device, and _criterion attributes.
            args: Experiment arguments (currently unused, kept for API consistency).
        """
        self.server = server
        self.args = args
        self.cumulative: Dict[int, float] = {}
        self.round_history: List[Dict] = []

    def compute_round_exact_shapley(
        self,
        client_updates: List[torch.Tensor],
        client_ids: List[int],
        eta: float,
        round_idx: int,
    ) -> Dict[int, float]:
        """Compute exact Shapley values by enumerating all 2^n subsets.

        Args:
            client_updates: List of flattened Δw_c tensors (CPU or GPU).
            client_ids: Client integer IDs corresponding to each update.
            eta: Effective learning rate η.
            round_idx: Round index for bookkeeping.

        Returns:
            Dict[int, float]: {client_id: exact_shapley_value} for this round.

        Raises:
            AssertionError: If len(client_ids) > 10 (exponential cost guard).
        """
        n = len(client_ids)
        assert n <= 10, (
            f"Exact Shapley requires n ≤ 10 clients, got {n}. "
            f"Use MonteCarloShapley for larger sets."
        )

        device = self.server.device
        updates_device = [u.to(device) for u in client_updates]

        # ------------------------------------------------------------------
        # Pre-compute utility for all 2^n subsets
        # Represent subsets as bitmasks for efficiency
        # ------------------------------------------------------------------
        # Map client_ids to local indices 0..n-1
        id_to_local = {cid: i for i, cid in enumerate(client_ids)}

        # Cache utility values: bitmask (int) → U(S)
        utility_cache: Dict[int, float] = {}

        # Baseline: U(∅) = 0 by definition
        utility_cache[0] = 0.0

        # Get current global model state
        global_state = self.server.get_model_state()

        # Enumerate all non-empty subsets
        for mask in range(1, 1 << n):
            subset_local_indices = [i for i in range(n) if (mask >> i) & 1]
            subset_updates = [updates_device[i] for i in subset_local_indices]

            utility_cache[mask] = self._compute_utility(
                global_state, subset_updates, eta
            )

        # ------------------------------------------------------------------
        # Exact Shapley formula:
        # ϕ_c = Σ_{S ⊆ [n]\{c}} [U(S∪{c}) - U(S)] * |S|!(n-1-|S|)! / n!
        # ------------------------------------------------------------------
        shapley: Dict[int, float] = {}

        for local_i, cid in enumerate(client_ids):
            phi = 0.0
            c_bit = 1 << local_i

            # Enumerate all subsets S that do NOT contain c
            for mask in range(1 << n):
                if (mask >> local_i) & 1:
                    continue  # c is in this subset; skip

                s_size = bin(mask).count("1")
                mask_with_c = mask | c_bit

                marginal = utility_cache[mask_with_c] - utility_cache[mask]

                # Weight = |S|! * (n-1-|S|)! / n!
                weight = (
                    math.factorial(s_size)
                    * math.factorial(n - 1 - s_size)
                    / math.factorial(n)
                )
                phi += weight * marginal

            shapley[cid] = phi

        self.round_history.append({"round": round_idx, **shapley})
        return shapley

    def _compute_utility(
        self,
        global_state: Dict[str, torch.Tensor],
        client_updates_subset: List[torch.Tensor],
        eta: float,
    ) -> float:
        """Apply subset updates and return validation loss delta U(S).

        U(S) = val_loss(w_t - η Σ_{c∈S} Δw_c) - val_loss(w_t)

        Since val_loss(w_t) is constant, we return:
        U(S) = val_loss(w_new(S)) - val_loss(w_t)

        Args:
            global_state: Current global model weights (CPU state_dict).
            client_updates_subset: List of flattened Δw_c tensors for subset S.
            eta: Learning rate for applying updates.

        Returns:
            Scalar utility value U(S).
        """
        device = self.server.device
        criterion = self.server._criterion

        # ------------------------------------------------------------------
        # Compute baseline val_loss(w_t) using current global model
        # (No need to load state since global_model is already at w_t)
        # ------------------------------------------------------------------
        baseline_loss = _compute_val_loss(
            self.server.global_model, self.server.val_loader, criterion, device
        )

        # ------------------------------------------------------------------
        # Build new model state: w_new = w_t - η Σ_{c∈S} Δw_c
        # ------------------------------------------------------------------
        # We need a temporary model for evaluation
        temp_model = copy.deepcopy(self.server.global_model)

        # Reconstruct state dict with flat updates
        # Sum the flat updates and apply to each parameter
        if client_updates_subset:
            # Sum all updates in subset
            total_flat_delta = torch.zeros_like(client_updates_subset[0])
            for delta_flat in client_updates_subset:
                total_flat_delta += delta_flat

            # Apply to trainable model parameters only
            with torch.no_grad():
                offset = 0
                for param in temp_model.parameters():
                    if param.requires_grad:
                        numel = param.numel()
                        delta_slice = total_flat_delta[offset: offset + numel]
                        delta_reshaped = delta_slice.reshape(param.shape).to(device)
                        # w_new = w_t + eta * Δw (Δw = w_local - w_global)
                        param.data += eta * delta_reshaped
                        offset += numel

        # ------------------------------------------------------------------
        # Compute val_loss(w_new(S))
        # ------------------------------------------------------------------
        new_loss = _compute_val_loss(temp_model, self.server.val_loader, criterion, device)

        del temp_model
        # U(S) = loss_before - loss_after (positive = helpful subset)
        return float(baseline_loss - new_loss)

    def accumulate(self, round_shapley: Dict[int, float]) -> None:
        """Add per-round exact Shapley values to cumulative sum.

        Args:
            round_shapley: {client_id: ϕ_c^(t)} from compute_round_exact_shapley.
        """
        for cid, sv in round_shapley.items():
            self.cumulative[cid] = self.cumulative.get(cid, 0.0) + sv

    def get_cumulative(self) -> Dict[int, float]:
        """Return cumulative exact Shapley values.

        Returns:
            Dict[int, float]: {client_id: Σ_t ϕ_c^(t)}.
        """
        return dict(self.cumulative)


def _compute_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Compute average validation loss without gradients.

    Args:
        model: Model to evaluate.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Average loss as a Python float.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

    return total_loss / max(total_samples, 1)
