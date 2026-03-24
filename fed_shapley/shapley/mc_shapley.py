"""
shapley/mc_shapley.py — Monte Carlo approximation of Shapley values.

Approximates Shapley values by averaging marginal contributions over
random permutations of clients, rather than enumerating all 2^n subsets.

The MC estimator:
    ϕ_c ≈ (1/T) Σ_{t=1}^{T} [U(S_before ∪ {c}) - U(S_before)]
where S_before = clients preceding c in permutation π_t.

Expected value converges to exact Shapley for T → ∞.
Complexity: O(T * n * val_loss) vs O(2^n * val_loss) for exact.

Reference: Castro et al. (2009) "Polynomial calculation of the Shapley value
based on sampling" — Computers & Operations Research.
"""

import argparse
import copy
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MonteCarloShapley:
    """Monte Carlo approximation of Shapley values for FL clients.

    Suitable when num_clients > 10 and exact enumeration is infeasible.

    Attributes:
        server: FL Server instance.
        num_permutations: Number of random permutations to sample.
        cumulative: Running sum of MC Shapley values per client.
        round_history: Per-round Shapley value snapshots.
    """

    def __init__(self, server: "Server", args: argparse.Namespace) -> None:
        """Initialize Monte Carlo Shapley calculator.

        Args:
            server: FL Server instance with global_model, val_loader, device.
            args: Experiment arguments. Used fields:
                - mc_permutations: int number of permutations
        """
        self.server = server
        self.args = args
        self.num_permutations = getattr(args, "mc_permutations", 1000)
        self.cumulative: Dict[int, float] = {}
        self.round_history: List[Dict] = []

    def compute_round_mc_shapley(
        self,
        client_updates: List[torch.Tensor],
        client_ids: List[int],
        eta: float,
        round_idx: int,
    ) -> Dict[int, float]:
        """Approximate Shapley values via random permutation sampling.

        For each permutation π:
            For each position i in π:
                S_before = {π[0], ..., π[i-1]}
                marginal_c += U(S_before ∪ {c}) - U(S_before)
        ϕ_c = marginal_c / num_permutations

        Args:
            client_updates: List of flattened Δw_c tensors.
            client_ids: Client integer IDs.
            eta: Effective learning rate.
            round_idx: Round index for bookkeeping.

        Returns:
            Dict[int, float]: {client_id: mc_shapley_value} for this round.
        """
        n = len(client_ids)
        device = self.server.device
        updates_device = [u.to(device) for u in client_updates]

        criterion = self.server._criterion

        # ------------------------------------------------------------------
        # Cache baseline validation loss (empty set utility = 0 by convention)
        # ------------------------------------------------------------------
        baseline_loss = _compute_val_loss_mc(
            self.server.global_model, self.server.val_loader, criterion, device
        )

        # ------------------------------------------------------------------
        # Accumulate marginal contributions over permutations
        # ------------------------------------------------------------------
        marginal_sums: Dict[int, float] = {cid: 0.0 for cid in client_ids}

        rng = random.Random(round_idx * 31337)  # Deterministic per-round seed

        for perm_idx in range(self.num_permutations):
            # Sample a random permutation
            perm = list(range(n))
            rng.shuffle(perm)  # in-place shuffle of local indices

            # Track cumulative update for this permutation
            current_flat = torch.zeros_like(updates_device[0])

            # val_loss with current cumulative updates (starts at baseline)
            prev_loss = baseline_loss

            for pos, local_i in enumerate(perm):
                cid = client_ids[local_i]

                # Add client local_i to the current set
                current_flat = current_flat + updates_device[local_i]

                # Build temp model with w_new = w_t - η * current_flat
                new_loss = _compute_val_loss_with_flat_delta(
                    self.server.global_model,
                    current_flat,
                    eta,
                    self.server.val_loader,
                    criterion,
                    device,
                )

                # Marginal contribution: U(S∪{c}) - U(S)
                # Note: utility is defined as loss DECREASE (lower = better)
                marginal = prev_loss - new_loss  # positive if this client helps

                marginal_sums[cid] += marginal
                prev_loss = new_loss

        # Normalize by number of permutations
        shapley: Dict[int, float] = {
            cid: marginal_sums[cid] / self.num_permutations
            for cid in client_ids
        }

        self.round_history.append({"round": round_idx, **shapley})
        return shapley

    # Keep API consistent with InRunDataShapley and ExactShapley
    def compute_round_shapley(
        self,
        client_updates: List[torch.Tensor],
        client_ids: List[int],
        eta: float,
        round_idx: int,
    ) -> Dict[int, float]:
        """Alias for compute_round_mc_shapley (unified API).

        Args:
            client_updates: Flattened Δw_c tensors.
            client_ids: Client IDs.
            eta: Learning rate.
            round_idx: Round index.

        Returns:
            Dict[int, float]: MC Shapley values for this round.
        """
        return self.compute_round_mc_shapley(
            client_updates, client_ids, eta, round_idx
        )

    def accumulate(self, round_shapley: Dict[int, float]) -> None:
        """Accumulate per-round MC Shapley into cumulative sum.

        Args:
            round_shapley: {client_id: ϕ_c^(t)} from compute_round_mc_shapley.
        """
        for cid, sv in round_shapley.items():
            self.cumulative[cid] = self.cumulative.get(cid, 0.0) + sv

    def get_cumulative(self) -> Dict[int, float]:
        """Return cumulative MC Shapley values.

        Returns:
            Dict[int, float]: {client_id: Σ_t ϕ_c^(t)}.
        """
        return dict(self.cumulative)


# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def _compute_val_loss_mc(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Compute average validation loss for the current model state.

    Args:
        model: Model to evaluate (not modified).
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Average loss as Python float.
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


def _compute_val_loss_with_flat_delta(
    base_model: nn.Module,
    flat_delta: torch.Tensor,
    eta: float,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Compute val loss after applying a flat parameter delta.

    Applies w_new = w_base - η * flat_delta to a temporary copy of the model,
    evaluates, then discards the temporary model.

    Args:
        base_model: Starting model (not modified).
        flat_delta: 1D tensor of aggregated parameter deltas.
        eta: Learning rate scaling factor.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Average validation loss as Python float.
    """
    temp_model = copy.deepcopy(base_model)
    temp_model.eval()

    with torch.no_grad():
        offset = 0
        for param in temp_model.parameters():
            if param.requires_grad:
                numel = param.numel()
                delta_slice = flat_delta[offset: offset + numel].reshape(param.shape)
                # w_new = w_t + eta * Δw (Δw = w_local - w_global)
                param.data += eta * delta_slice.to(device)
                offset += numel

    loss = _compute_val_loss_mc(temp_model, val_loader, criterion, device)
    del temp_model
    return loss
