"""
eval/fidelity.py — Fidelity metrics comparing in-run Shapley to ground truth.

Measures how well in-run Shapley approximates the ground-truth Shapley values
(computed by exact enumeration or MC) using:
- RMSE: absolute estimation error
- Spearman ρ: rank-order correlation (most relevant for data selection)
- Pearson r: linear correlation

A high Spearman correlation is the primary criterion of interest because
Shapley values are typically used for ranking clients, not exact values.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_fidelity(
    in_run_values: Dict[int, float],
    ground_truth_values: Dict[int, float],
) -> Dict[str, float]:
    """Compute fidelity metrics between in-run and ground-truth Shapley values.

    Aligns both dicts by client_id (clients present in both dicts only),
    then computes correlation and error metrics.

    Args:
        in_run_values: {client_id: in_run_shapley} estimates.
        ground_truth_values: {client_id: ground_truth_shapley} reference values.

    Returns:
        Dict with keys:
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'pearson_r': Pearson correlation coefficient
        - 'pearson_p': Pearson p-value
        - 'spearman_r': Spearman rank correlation
        - 'spearman_p': Spearman p-value
        - 'n_clients': number of clients aligned

    Notes:
        Returns NaN for all metrics if fewer than 3 aligned clients exist.
    """
    # Find common client IDs
    common_ids = sorted(
        set(in_run_values.keys()) & set(ground_truth_values.keys())
    )

    if len(common_ids) < 3:
        print(f"[Fidelity] Warning: Only {len(common_ids)} common clients. "
              f"Cannot compute correlations reliably.")
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "n_clients": len(common_ids),
        }

    in_run_arr = np.array([in_run_values[c] for c in common_ids])
    gt_arr = np.array([ground_truth_values[c] for c in common_ids])

    # RMSE and MAE
    diff = in_run_arr - gt_arr
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # Pearson correlation
    if np.std(in_run_arr) < 1e-12 or np.std(gt_arr) < 1e-12:
        pearson_r, pearson_p = float("nan"), float("nan")
    else:
        pearson_r, pearson_p = stats.pearsonr(gt_arr, in_run_arr)
        pearson_r, pearson_p = float(pearson_r), float(pearson_p)

    # Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(gt_arr, in_run_arr)
    spearman_r, spearman_p = float(spearman_r), float(spearman_p)

    result = {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n_clients": len(common_ids),
    }

    print(f"[Fidelity] n={len(common_ids)} clients | "
          f"RMSE={rmse:.6f} | MAE={mae:.6f} | "
          f"Pearson r={pearson_r:.4f} (p={pearson_p:.4f}) | "
          f"Spearman rho={spearman_r:.4f} (p={spearman_p:.4f})")

    return result


def plot_fidelity_scatter(
    in_run: List[float],
    ground_truth: List[float],
    title: str = "In-Run vs Ground Truth Shapley",
    save_path: str = "fidelity_scatter.png",
) -> None:
    """Scatter plot of in-run vs. ground-truth Shapley values.

    Draws an identity (y=x) reference line and annotates with correlation stats.

    Args:
        in_run: List of in-run Shapley estimates (aligned with ground_truth).
        ground_truth: List of ground-truth Shapley values.
        title: Plot title.
        save_path: Output file path for the PNG.
    """
    in_run_arr = np.array(in_run)
    gt_arr = np.array(ground_truth)

    if len(in_run_arr) >= 3:
        pearson_r, _ = stats.pearsonr(gt_arr, in_run_arr)
        spearman_r, _ = stats.spearmanr(gt_arr, in_run_arr)
        corr_str = f"Pearson r={pearson_r:.3f}\nSpearman ρ={spearman_r:.3f}"
    else:
        corr_str = "n < 3"

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_arr, in_run_arr, color="steelblue", alpha=0.8,
               edgecolors="black", s=70, zorder=3)

    # Identity reference line
    lim_min = min(gt_arr.min(), in_run_arr.min())
    lim_max = max(gt_arr.max(), in_run_arr.max())
    margin = (lim_max - lim_min) * 0.05 if lim_max != lim_min else 0.1
    ax.plot(
        [lim_min - margin, lim_max + margin],
        [lim_min - margin, lim_max + margin],
        "r--", linewidth=1.5, label="y = x (perfect)", zorder=2,
    )

    ax.set_xlabel("Ground Truth Shapley ϕ_c", fontsize=12)
    ax.set_ylabel("In-Run Shapley ϕ̂_c", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title=corr_str, fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fidelity] Saved scatter plot: {save_path}")
