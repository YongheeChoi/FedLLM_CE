"""
config.py — Argument parsing for FedShapley experiments.

All hyperparameters for FL training, data partitioning, Shapley computation,
evaluation, and logging are defined here with sensible defaults.
"""

import argparse
from typing import Optional


def get_args() -> argparse.Namespace:
    """Parse and return all experiment arguments.

    Returns:
        argparse.Namespace: Parsed arguments with all defaults set.
    """
    parser = argparse.ArgumentParser(
        description="Federated Learning with In-Run Data Shapley Values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--exp_name", type=str, default="exp",
        help="Experiment name used for output directory naming."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device: 'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Root directory for experiment outputs."
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--dataset", type=str,
        choices=["cifar10", "cifar100", "tinyimagenet"],
        default="cifar10",
        help="Dataset to use."
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data_cache",
        help="Directory to cache downloaded datasets."
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model", type=str, choices=["resnet18"], default="resnet18",
        help="Model architecture."
    )

    # -------------------------------------------------------------------------
    # Federated Learning
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--num_clients", type=int, default=10,
        help="Total number of FL clients."
    )
    parser.add_argument(
        "--clients_per_round", type=int, default=None,
        help="Number of clients sampled per round. None means all clients."
    )
    parser.add_argument(
        "--num_rounds", type=int, default=100,
        help="Total number of FL communication rounds."
    )
    parser.add_argument(
        "--local_epochs", type=int, default=5,
        help="Number of local SGD epochs per client per round."
    )
    parser.add_argument(
        "--local_lr", type=float, default=0.01,
        help="Learning rate for client-side SGD."
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=64,
        help="Mini-batch size for client local training."
    )
    parser.add_argument(
        "--aggregation", type=str, choices=["fedavg"], default="fedavg",
        help="Server aggregation rule."
    )

    # -------------------------------------------------------------------------
    # Data Partition
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--partition", type=str,
        choices=["iid", "dirichlet"],
        default="iid",
        help="Data partition strategy across clients."
    )
    parser.add_argument(
        "--dirichlet_alpha", type=float, default=0.5,
        help="Concentration parameter for Dirichlet label distribution (lower = more non-IID)."
    )
    parser.add_argument(
        "--quantity_skew", action="store_true", default=False,
        help="Apply quantity skew (heterogeneous dataset sizes) across clients."
    )
    parser.add_argument(
        "--quantity_beta", type=float, default=0.5,
        help="Dirichlet concentration for quantity skew."
    )

    # -------------------------------------------------------------------------
    # Shapley
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--use_second_order", action="store_true", default=False,
        help="Include second-order (Hessian) term in Shapley computation."
    )
    parser.add_argument(
        "--num_val_samples", type=int, default=500,
        help="Number of validation samples extracted from training data."
    )
    parser.add_argument(
        "--mc_permutations", type=int, default=1000,
        help="Number of random permutations for Monte Carlo Shapley."
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--eval_every", type=int, default=1,
        help="Evaluate on val/test every this many rounds."
    )
    parser.add_argument(
        "--noisy_clients", type=int, nargs="+", default=[],
        help="List of client IDs that are injected with noise."
    )
    parser.add_argument(
        "--noise_type", type=str,
        choices=["label_flip", "random_update"],
        default="label_flip",
        help="Type of noise injected into noisy clients."
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.5,
        help="Fraction of data affected by noise (currently informational)."
    )
    parser.add_argument(
        "--run_centralized", action="store_true", default=False,
        help="Run centralized training for ground-truth Shapley computation."
    )
    parser.add_argument(
        "--run_exact_shapley", action="store_true", default=False,
        help="Run exact (exponential) or MC Shapley for comparison."
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--use_wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="fed_shapley",
        help="W&B project name."
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name. Defaults to exp_name."
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="W&B entity (team or username)."
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true", default=False,
        help="Enable TensorBoard logging."
    )
    parser.add_argument(
        "--log_every", type=int, default=1,
        help="Log metrics every this many rounds."
    )

    args = parser.parse_args()

    # Post-processing: if clients_per_round is None, set to num_clients
    if args.clients_per_round is None:
        args.clients_per_round = args.num_clients

    return args
