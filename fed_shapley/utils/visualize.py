"""
utils/visualize.py — Visualization utilities for Shapley values and FL diagnostics.

Provides bar charts for client contributions, scatter plots for fidelity assessment,
line plots for client removal experiments, and heatmaps for data partition analysis.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_shapley_bar(
    shapley_values: Dict[int, float],
    title: str = "Client Shapley Values",
    save_path: str = "shapley_bar.png",
) -> None:
    """Create a bar chart of per-client Shapley contributions.

    Args:
        shapley_values: {client_id: shapley_value} mapping.
        title: Plot title string.
        save_path: Full path where the PNG will be saved.

    Notes:
        Bars are colored green for positive contributions and red for negative.
        A dashed horizontal line at y=0 marks the contribution threshold.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    client_ids = sorted(shapley_values.keys())
    values = [shapley_values[c] for c in client_ids]

    colors = ["steelblue" if v >= 0 else "tomato" for v in values]

    fig, ax = plt.subplots(figsize=(max(8, len(client_ids) * 0.8), 5))
    bars = ax.bar([str(c) for c in client_ids], values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    # Annotate bar tops
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + abs(max(values) - min(values)) * 0.02 if val >= 0 \
            else bar.get_height() - abs(max(values) - min(values)) * 0.04
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8, rotation=45,
        )

    ax.set_xlabel("Client ID", fontsize=12)
    ax.set_ylabel("Shapley Value ϕ_c", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved shapley bar chart: {save_path}")


def plot_fidelity_scatter(
    in_run: List[float],
    ground_truth: List[float],
    title: str = "In-Run vs Ground Truth Shapley",
    save_path: str = "fidelity_scatter.png",
) -> None:
    """Scatter plot comparing in-run Shapley estimates to ground truth.

    Args:
        in_run: Per-client in-run Shapley estimates (same order as ground_truth).
        ground_truth: Per-client ground truth Shapley values.
        title: Plot title string.
        save_path: Full path where the PNG will be saved.

    Notes:
        An identity line (y=x) is drawn for reference.
        Pearson and Spearman correlations are shown in the legend.
    """
    from scipy import stats

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    in_run_arr = np.array(in_run)
    gt_arr = np.array(ground_truth)

    pearson_r, pearson_p = stats.pearsonr(gt_arr, in_run_arr)
    spearman_r, spearman_p = stats.spearmanr(gt_arr, in_run_arr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_arr, in_run_arr, color="steelblue", alpha=0.7, edgecolors="black", s=60, zorder=3)

    # Identity line
    lim_min = min(gt_arr.min(), in_run_arr.min())
    lim_max = max(gt_arr.max(), in_run_arr.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot(
        [lim_min - margin, lim_max + margin],
        [lim_min - margin, lim_max + margin],
        "r--", linewidth=1.5, label="y = x", zorder=2,
    )

    ax.set_xlabel("Ground Truth Shapley", fontsize=12)
    ax.set_ylabel("In-Run Shapley", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(
        title=f"Pearson r={pearson_r:.3f}\nSpearman ρ={spearman_r:.3f}",
        fontsize=9, title_fontsize=9,
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved fidelity scatter: {save_path}")


def plot_client_removal(
    removal_rates: List[float],
    accs_high: List[float],
    accs_low: List[float],
    accs_random: List[float],
    save_path: str = "client_removal.png",
) -> None:
    """Line plot showing accuracy vs. fraction of clients removed.

    Three removal strategies are compared:
    - High-to-low (remove most valuable clients first)
    - Low-to-high (remove least valuable clients first)
    - Random order (baseline)

    Args:
        removal_rates: Fractions of clients removed, e.g. [0.1, 0.2, ..., 1.0].
        accs_high: Test accuracy after removing top-Shapley clients.
        accs_low: Test accuracy after removing bottom-Shapley clients.
        accs_random: Test accuracy after removing random clients.
        save_path: Full path where the PNG will be saved.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(removal_rates, accs_high, "o-", color="tomato", label="Remove high-ϕ first", linewidth=2)
    ax.plot(removal_rates, accs_low, "s-", color="steelblue", label="Remove low-ϕ first", linewidth=2)
    ax.plot(removal_rates, accs_random, "^--", color="gray", label="Random removal", linewidth=1.5)

    ax.set_xlabel("Fraction of Clients Removed", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Client Removal Experiment", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, max(removal_rates) + 0.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved client removal plot: {save_path}")


def plot_partition_heatmap(
    client_data: Dict[int, List[int]],
    num_classes: int,
    save_path: str = "partition_heatmap.png",
) -> None:
    """Heatmap showing the class distribution across clients.

    Args:
        client_data: {client_id: [class_labels]} mapping.
            The values should be a list of integer class labels
            (or counts per class in a pre-aggregated form).
        num_classes: Total number of classes in the dataset.
        save_path: Full path where the PNG will be saved.

    Notes:
        Each row = one client; each column = one class.
        Cell values are normalized within each client (fraction of that client's
        data belonging to each class), so rows sum to 1.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    num_clients = len(client_data)
    matrix = np.zeros((num_clients, num_classes))

    for i, (client_id, labels) in enumerate(sorted(client_data.items())):
        labels_arr = np.array(labels)
        for c in range(num_classes):
            matrix[i, c] = np.sum(labels_arr == c)
        row_sum = matrix[i].sum()
        if row_sum > 0:
            matrix[i] /= row_sum  # normalize to fractions

    fig_width = max(10, num_classes * 0.4)
    fig_height = max(4, num_clients * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        xticklabels=[str(c) for c in range(num_classes)],
        yticklabels=[f"Client {c}" for c in sorted(client_data.keys())],
        linewidths=0.3 if num_classes <= 20 else 0,
        cbar_kws={"label": "Fraction of Client Data"},
    )
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Client", fontsize=12)
    ax.set_title("Data Partition: Class Distribution per Client", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved partition heatmap: {save_path}")
