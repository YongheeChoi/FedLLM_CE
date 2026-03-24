"""
eval/noisy_client.py — Noisy client detection via Shapley value ranking.

Tests whether Shapley values can identify adversarial/noisy clients without
any explicit knowledge of which clients are noisy.

Hypothesis: Noisy clients (label flippers, random updaters) should receive
lower Shapley values because their updates harm validation performance.

Evaluation: AUROC of using (-shapley_value) as a score for classifying
clients as noisy vs. clean.

AUROC interpretation:
- 1.0: Perfect detection (all noisy clients rank below all clean clients)
- 0.5: Random baseline (Shapley values contain no signal about noisiness)
- 0.0: Inverse detection (would be perfect if we flip the score)
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def noisy_client_detection(
    shapley_values: Dict[int, float],
    noisy_client_ids: List[int],
    all_client_ids: List[int],
) -> float:
    """Compute AUROC for noisy client detection using Shapley value ranking.

    Uses -shapley_value as the anomaly score: clients with lower (more negative)
    Shapley values are considered more likely to be noisy.

    Args:
        shapley_values: {client_id: shapley_value} from the main experiment.
        noisy_client_ids: Ground-truth list of noisy client IDs.
        all_client_ids: Complete list of all client IDs.

    Returns:
        AUROC score (float in [0, 1]).
        Returns 0.5 if there are no positive or no negative examples.

    Notes:
        - Clients not in shapley_values get shapley=0 (neutral attribution).
        - Uses sklearn.metrics.roc_auc_score for the AUROC computation.
    """
    if not noisy_client_ids:
        print("[NoisyDetection] No noisy clients specified. AUROC undefined (returning 0.5).")
        return 0.5

    noisy_set = set(noisy_client_ids)
    client_ids = all_client_ids

    # Build binary labels and scores
    y_true = []   # 1 = noisy, 0 = clean
    y_score = []  # higher score → more likely noisy

    for cid in client_ids:
        label = 1 if cid in noisy_set else 0
        # Use negative shapley as "noisiness score"
        # (noisier clients should have lower shapley → higher -shapley)
        score = -shapley_values.get(cid, 0.0)
        y_true.append(label)
        y_score.append(score)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Check that both classes are present
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("[NoisyDetection] Warning: Only one class present. AUROC undefined (returning 0.5).")
        return 0.5

    auroc = float(roc_auc_score(y_true, y_score))

    # Print detailed detection stats
    noisy_shapley = [shapley_values.get(cid, 0.0) for cid in noisy_client_ids]
    clean_ids = [cid for cid in client_ids if cid not in noisy_set]
    clean_shapley = [shapley_values.get(cid, 0.0) for cid in clean_ids]

    print(f"\n[NoisyDetection] Noisy client detection results:")
    print(f"  Noisy clients ({len(noisy_client_ids)}): "
          f"mean phi = {np.mean(noisy_shapley):.6f} +/- {np.std(noisy_shapley):.6f}")
    print(f"  Clean clients ({len(clean_ids)}): "
          f"mean phi = {np.mean(clean_shapley):.6f} +/- {np.std(clean_shapley):.6f}")
    print(f"  AUROC = {auroc:.4f}")

    # Check ranking: are all noisy clients in the bottom k?
    sorted_by_shapley = sorted(shapley_values.items(), key=lambda x: x[1])
    bottom_k = [cid for cid, _ in sorted_by_shapley[:len(noisy_client_ids)]]
    detected = sum(1 for cid in noisy_client_ids if cid in bottom_k)
    print(f"  Detection@{len(noisy_client_ids)}: "
          f"{detected}/{len(noisy_client_ids)} noisy clients in bottom-k by Shapley")

    return auroc


def compute_detection_metrics(
    shapley_values: Dict[int, float],
    noisy_client_ids: List[int],
    all_client_ids: List[int],
) -> Dict[str, float]:
    """Extended noisy client detection metrics beyond AUROC.

    Args:
        shapley_values: {client_id: shapley_value}.
        noisy_client_ids: Ground-truth noisy client IDs.
        all_client_ids: All client IDs.

    Returns:
        Dict with metrics:
        - 'auroc': AUROC score
        - 'precision_at_k': Precision@k (k = number of noisy clients)
        - 'recall_at_k': Recall@k
        - 'mean_noisy_shapley': Mean Shapley of noisy clients
        - 'mean_clean_shapley': Mean Shapley of clean clients
        - 'separation_ratio': (mean_clean - mean_noisy) / std_all
    """
    auroc = noisy_client_detection(shapley_values, noisy_client_ids, all_client_ids)

    noisy_set = set(noisy_client_ids)
    k = len(noisy_client_ids)

    # Sort by ascending Shapley (lowest = most suspicious)
    sorted_clients = sorted(shapley_values.items(), key=lambda x: x[1])
    bottom_k = [cid for cid, _ in sorted_clients[:k]]

    true_positives = sum(1 for cid in bottom_k if cid in noisy_set)
    precision_at_k = true_positives / max(k, 1)
    recall_at_k = true_positives / max(len(noisy_client_ids), 1)

    all_values = list(shapley_values.values())
    noisy_values = [shapley_values.get(cid, 0.0) for cid in noisy_client_ids]
    clean_values = [shapley_values.get(cid, 0.0) for cid in all_client_ids
                    if cid not in noisy_set]

    mean_noisy = float(np.mean(noisy_values)) if noisy_values else float("nan")
    mean_clean = float(np.mean(clean_values)) if clean_values else float("nan")
    std_all = float(np.std(all_values)) if all_values else 1.0
    sep_ratio = (mean_clean - mean_noisy) / max(std_all, 1e-12)

    return {
        "auroc": auroc,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mean_noisy_shapley": mean_noisy,
        "mean_clean_shapley": mean_clean,
        "separation_ratio": sep_ratio,
    }
