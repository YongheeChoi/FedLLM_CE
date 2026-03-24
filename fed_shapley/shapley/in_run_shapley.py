"""
shapley/in_run_shapley.py — In-Run Data Shapley for Federated Learning.

Computes per-client Shapley values from gradient dot-products in a single
training run, without enumerating exponentially many subsets.

Key formulas:
    1st order:  ϕ_c^(t) = -η · ∇ℓ(w_t, z_val) · Δw_c^(t)
    2nd order:  ϕ_c^(t) += (η²/2) · Δw_c^(t)ᵀ H^(z_val) Σ_{c'} Δw_{c'}^(t)
    Cumulative: ϕ_c = Σ_{t: c ∈ C_t} ϕ_c^(t)

Reference: "Data Shapley in One Training Run" (ICLR 2025 Outstanding Paper Runner-up)
"""

import argparse
from typing import Dict, List, Optional

import torch


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all floating-point parameters in a state_dict to a 1D vector.

    Args:
        state_dict: Dict mapping parameter names to tensors.

    Returns:
        1D float tensor of shape (P,) where P is the total number of
        floating-point parameters.

    Notes:
        Non-floating-point tensors (e.g., BatchNorm's num_batches_tracked)
        are excluded to avoid type errors in gradient computations.
    """
    parts = []
    for v in state_dict.values():
        if v.dtype.is_floating_point:
            parts.append(v.detach().reshape(-1).float())
    return torch.cat(parts) if parts else torch.tensor([])


def unflatten_to_state_dict(
    vector: torch.Tensor,
    reference: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reconstruct a state_dict from a flat vector using a reference structure.

    Args:
        vector: 1D tensor of length P (total float parameters).
        reference: Reference state_dict whose shapes define the reconstruction.

    Returns:
        Reconstructed state_dict with the same structure as reference.
        Non-floating-point keys are copied from reference unchanged.

    Raises:
        RuntimeError: If vector length doesn't match total float parameter count.
    """
    result: Dict[str, torch.Tensor] = {}
    offset = 0

    for name, ref_tensor in reference.items():
        if ref_tensor.dtype.is_floating_point:
            numel = ref_tensor.numel()
            result[name] = vector[offset: offset + numel].reshape(ref_tensor.shape)
            offset += numel
        else:
            result[name] = ref_tensor.clone()

    if offset != len(vector):
        raise RuntimeError(
            f"Vector length {len(vector)} != total float params {offset}"
        )

    return result


class InRunDataShapley:
    """Compute per-client Shapley values via gradient dot-products.

    This avoids the exponential-cost exact Shapley computation by exploiting
    the linearity of the Shapley value under first-order FL model updates.

    The key insight is that:
        ϕ_c^(t) ≈ -η · <∇ℓ(w_t, D_val), Δw_c^(t)>
    which measures how much client c's update aligns with the direction that
    reduces validation loss.

    Attributes:
        server: Reference to the FL server (for gradient/HVP computation).
        use_second_order: Whether to include the Hessian correction term.
        cumulative: Running sum of Shapley values per client.
        round_history: Per-round Shapley value snapshots.
    """

    def __init__(self, server: "Server", args: argparse.Namespace) -> None:
        """Initialize the In-Run Shapley calculator.

        Args:
            server: FL Server instance (must have compute_validation_gradient
                and optionally compute_validation_hessian_vector_product).
            args: Experiment arguments. Used fields:
                - use_second_order: bool
        """
        self.server = server
        self.use_second_order = getattr(args, "use_second_order", False)
        self.cumulative: Dict[int, float] = {}
        self.round_history: List[Dict] = []

    def compute_round_shapley(
        self,
        client_updates: List[torch.Tensor],
        client_ids: List[int],
        eta: float,
        round_idx: int,
    ) -> Dict[int, float]:
        """Compute Shapley values for all clients in a given round.

        Steps:
        1. Compute validation gradient ∇ℓ(w_t, D_val).
        2. First-order term: ϕ_c^(1) = -η · <∇ℓ, Δw_c>
        3. (Optional) Second-order Hessian correction.
        4. Store and return round Shapley dict.

        Args:
            client_updates: List of flattened Δw_c tensors (CPU or GPU).
                Each has shape (P,) where P = total float parameters.
            client_ids: List of client integer IDs (same length as client_updates).
            eta: Effective learning rate (η) for the Shapley formula.
                For equal-weight FedAvg: η = local_lr.
            round_idx: 1-based communication round index for bookkeeping.

        Returns:
            Dict[int, float]: {client_id: ϕ_c^(t)} for this round only.
        """
        device = self.server.device

        # Move all updates to computation device
        updates_device = [u.to(device) for u in client_updates]

        # ------------------------------------------------------------------
        # Step 1: ∇ℓ(w_t, D_val)
        # ------------------------------------------------------------------
        val_grad = self.server.compute_validation_gradient()  # shape: (P,)

        # ------------------------------------------------------------------
        # Step 2: First-order Shapley terms
        # ϕ_c^(1) = -η * <∇ℓ, Δw_c>
        # ------------------------------------------------------------------
        first_order: Dict[int, float] = {}
        for cid, delta in zip(client_ids, updates_device):
            # Dot product: gradient alignment with client's update direction
            dot = torch.dot(val_grad.float(), delta.float()).item()
            # Negative sign: positive alignment → validation loss decreases
            first_order[cid] = -eta * dot  # ϕ_c^(1)

        # ------------------------------------------------------------------
        # Step 3: Second-order Hessian correction (optional)
        # ϕ_c^(2) = (η²/2) * Δw_c^T · H · Σ_{c'} Δw_{c'}
        # ------------------------------------------------------------------
        second_order: Dict[int, float] = {}
        if self.use_second_order:
            # Aggregate all client updates: Σ_c Δw_c
            aggregated = torch.zeros_like(updates_device[0])
            for delta in updates_device:
                aggregated = aggregated + delta

            # H · (Σ Δw_c)
            hvp = self.server.compute_validation_hessian_vector_product(aggregated)

            for cid, delta in zip(client_ids, updates_device):
                # Δw_c^T · (H · Σ Δw_{c'})
                hess_term = torch.dot(delta.float(), hvp.float()).item()
                second_order[cid] = (eta ** 2 / 2) * hess_term  # ϕ_c^(2)

        # ------------------------------------------------------------------
        # Step 4: Combine and store
        # ϕ_c^(t) = ϕ_c^(1) + ϕ_c^(2)
        # ------------------------------------------------------------------
        round_shapley: Dict[int, float] = {}
        for cid in client_ids:
            round_shapley[cid] = first_order[cid] + second_order.get(cid, 0.0)

        self.round_history.append({"round": round_idx, **round_shapley})
        return round_shapley

    def accumulate(self, round_shapley: Dict[int, float]) -> None:
        """Add per-round Shapley values to the running cumulative sum.

        Args:
            round_shapley: {client_id: ϕ_c^(t)} from compute_round_shapley.
        """
        for cid, sv in round_shapley.items():
            self.cumulative[cid] = self.cumulative.get(cid, 0.0) + sv

    def get_cumulative(self) -> Dict[int, float]:
        """Return the current cumulative Shapley value dict.

        Returns:
            Dict[int, float]: {client_id: Σ_t ϕ_c^(t)}.
            Note: Clients never selected will not appear in this dict.
        """
        return dict(self.cumulative)
