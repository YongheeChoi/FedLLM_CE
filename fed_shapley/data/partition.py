"""
data/partition.py — Federated data partitioning strategies.

Supports:
- IID: uniform random split across clients
- Dirichlet non-IID: label distribution heterogeneity via Dir(alpha)
- Quantity skew: heterogeneous dataset sizes via Dir(beta), composable with
  either IID or Dirichlet label distributions

Reference: "Data Shapley in One Training Run" (ICLR 2025)
"""

import numpy as np
from typing import Dict, List, Optional

from torch.utils.data import Dataset


def partition_data(
    dataset: Dataset,
    num_clients: int,
    partition_type: str,
    num_classes: int,
    dirichlet_alpha: float = 0.5,
    quantity_skew: bool = False,
    quantity_beta: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Partition dataset indices across clients.

    Args:
        dataset: PyTorch dataset with accessible targets/labels.
        num_clients: Number of FL clients.
        partition_type: 'iid' or 'dirichlet'.
        num_classes: Total number of classes.
        dirichlet_alpha: Concentration parameter for Dirichlet label distribution.
            Smaller alpha → more non-IID. Used only when partition_type='dirichlet'.
        quantity_skew: If True, apply Dirichlet-based quantity skew (heterogeneous
            dataset sizes across clients).
        quantity_beta: Dirichlet concentration for quantity skew.
            Smaller beta → more skewed sizes.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping client_id (int) -> list of sample indices (List[int]).

    Notes:
        - Minimum 10 samples per client enforced when quantity_skew=True.
        - For Dirichlet partitioning, some clients may get 0 samples for rare
          classes; these are filled from the IID remainder to guarantee minimum.
    """
    rng = np.random.default_rng(seed)

    # Extract all labels
    if hasattr(dataset, "targets"):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        all_labels = np.array(dataset.labels)
    else:
        # Fallback: iterate (slow)
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

    n_samples = len(all_labels)
    all_indices = np.arange(n_samples)

    # -----------------------------------------------------------------
    # Step 1: Determine per-client data quotas (quantity skew)
    # -----------------------------------------------------------------
    if quantity_skew:
        proportions = rng.dirichlet([quantity_beta] * num_clients)
        quotas = (proportions * n_samples).astype(int)
        # Ensure minimum 10 samples per client
        quotas = np.maximum(quotas, 10)
        # Adjust total to match n_samples exactly
        diff = n_samples - quotas.sum()
        # Distribute remainder to the client with most samples
        if diff > 0:
            quotas[np.argmax(quotas)] += diff
        elif diff < 0:
            # Remove excess from largest clients
            for _ in range(-diff):
                idx = np.argmax(quotas - 10)
                if quotas[idx] > 10:
                    quotas[idx] -= 1
    else:
        # Equal quotas
        base = n_samples // num_clients
        quotas = np.full(num_clients, base, dtype=int)
        quotas[:n_samples % num_clients] += 1  # distribute remainder

    # -----------------------------------------------------------------
    # Step 2: Distribute samples according to label strategy
    # -----------------------------------------------------------------
    if partition_type == "iid":
        client_indices = _iid_partition(all_indices, quotas, rng)

    elif partition_type == "dirichlet":
        client_indices = _dirichlet_partition(
            all_indices, all_labels, num_clients, num_classes,
            dirichlet_alpha, quotas, quantity_skew, rng,
        )
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}. "
                         f"Choose 'iid' or 'dirichlet'.")

    return client_indices


def _iid_partition(
    indices: np.ndarray,
    quotas: np.ndarray,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    """Shuffle indices and split by quota.

    Args:
        indices: Array of all sample indices.
        quotas: Per-client sample count array of shape (num_clients,).
        rng: NumPy random generator.

    Returns:
        Dict[int, List[int]]: Client ID -> sample indices.
    """
    shuffled = rng.permutation(indices)
    client_indices: Dict[int, List[int]] = {}
    start = 0
    for client_id, quota in enumerate(quotas):
        client_indices[client_id] = shuffled[start: start + quota].tolist()
        start += quota
    return client_indices


def _dirichlet_partition(
    indices: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    quotas: np.ndarray,
    quantity_skew: bool,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    """Partition using Dirichlet label distribution.

    For each class c, sample proportions from Dir(alpha) across clients,
    then allocate indices accordingly.

    Args:
        indices: All sample indices.
        labels: Label for each sample (same length as indices).
        num_clients: Total FL clients.
        num_classes: Number of label classes.
        alpha: Dirichlet concentration.
        quotas: Per-client total sample quotas (used when quantity_skew=True).
        quantity_skew: Whether quantity quotas should be respected.
        rng: NumPy random generator.

    Returns:
        Dict[int, List[int]]: Client ID -> sample indices.
    """
    # Group indices by class
    class_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, lbl in zip(indices, labels[indices]):
        class_indices[int(lbl)].append(int(idx))

    # Shuffle within each class
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # Sample Dirichlet proportions: shape (num_classes, num_clients)
    proportions = rng.dirichlet([alpha] * num_clients, size=num_classes)

    client_buckets: Dict[int, List[int]] = {c: [] for c in range(num_clients)}

    for c in range(num_classes):
        cls_idxs = class_indices[c]
        n_cls = len(cls_idxs)
        if n_cls == 0:
            continue

        # Convert proportions to counts
        counts = (proportions[c] * n_cls).astype(int)
        counts[-1] = n_cls - counts[:-1].sum()  # fix rounding
        counts = np.maximum(counts, 0)

        ptr = 0
        for client_id in range(num_clients):
            cnt = counts[client_id]
            client_buckets[client_id].extend(cls_idxs[ptr: ptr + cnt])
            ptr += cnt

    # If quantity_skew, trim/pad each client bucket to match its quota
    if quantity_skew:
        all_remaining = []
        for client_id in range(num_clients):
            bucket = client_buckets[client_id]
            quota = quotas[client_id]
            if len(bucket) > quota:
                # Return excess to pool
                rng.shuffle(bucket)
                all_remaining.extend(bucket[quota:])
                client_buckets[client_id] = bucket[:quota]

        # Distribute remaining indices to under-quota clients
        rng.shuffle(all_remaining)
        ptr = 0
        for client_id in range(num_clients):
            deficit = quotas[client_id] - len(client_buckets[client_id])
            if deficit > 0 and ptr < len(all_remaining):
                fill = all_remaining[ptr: ptr + deficit]
                client_buckets[client_id].extend(fill)
                ptr += len(fill)

    return client_buckets


def get_client_stats(
    client_indices: Dict[int, List[int]],
    dataset: Dataset,
    num_classes: int,
) -> Dict[int, Dict[str, object]]:
    """Compute per-client data statistics including class distribution.

    Args:
        client_indices: {client_id: [sample_indices]} from partition_data.
        dataset: The original dataset (used to read labels).
        num_classes: Total number of classes.

    Returns:
        Dict mapping client_id to a stats dict with keys:
        - 'num_samples': int total samples
        - 'class_counts': List[int] count per class
        - 'class_fractions': List[float] fraction per class
        - 'dominant_class': int most frequent class
        - 'num_represented_classes': int number of classes with >0 samples
    """
    if hasattr(dataset, "targets"):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        all_labels = np.array(dataset.labels)
    else:
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

    stats: Dict[int, Dict] = {}
    for client_id, idxs in client_indices.items():
        if len(idxs) == 0:
            stats[client_id] = {
                "num_samples": 0,
                "class_counts": [0] * num_classes,
                "class_fractions": [0.0] * num_classes,
                "dominant_class": -1,
                "num_represented_classes": 0,
            }
            continue

        client_labels = all_labels[idxs]
        counts = [int(np.sum(client_labels == c)) for c in range(num_classes)]
        total = len(idxs)
        fractions = [c / total for c in counts]

        stats[client_id] = {
            "num_samples": total,
            "class_counts": counts,
            "class_fractions": fractions,
            "dominant_class": int(np.argmax(counts)),
            "num_represented_classes": int(np.sum(np.array(counts) > 0)),
        }

    return stats
