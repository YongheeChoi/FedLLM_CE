"""
eval/client_removal.py — Client removal experiment for validating Shapley values.

Tests whether Shapley values correctly identify the most (and least) valuable
clients by:
1. Removing clients with highest Shapley → accuracy should drop sharply.
2. Removing clients with lowest Shapley → accuracy should be stable.
3. Removing clients randomly → intermediate accuracy decrease (baseline).

A good Shapley measure should show:
    acc(remove_high) << acc(remove_random) < acc(remove_low)
"""

import argparse
import copy
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def client_removal_experiment(
    server: "Server",
    clients: List["Client"],
    shapley_values: Dict[int, float],
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> Dict[str, List[float]]:
    """Run client removal experiments for all three removal strategies.

    For each removal fraction in [0.1, 0.2, ..., 0.9]:
    - Remove top-Shapley clients (high-to-low removal)
    - Remove bottom-Shapley clients (low-to-high removal)
    - Remove random clients (averaged over 3 trials)
    For each case, retrain FL from scratch with the remaining clients and
    record final test accuracy.

    Args:
        server: FL Server (provides model architecture and evaluate()).
        clients: List of all FL Client instances.
        shapley_values: {client_id: shapley_value} from main experiment.
        test_loader: DataLoader for final test evaluation.
        args: Experiment arguments. Used fields:
            - num_rounds (20% of original used for efficiency)
            - local_epochs, local_lr, clients_per_round, device, seed,
              aggregation, eval_every

    Returns:
        Dict with keys:
        - 'removal_rates': list of removal fractions
        - 'accs_high': test accuracy after removing highest-Shapley clients
        - 'accs_low': test accuracy after removing lowest-Shapley clients
        - 'accs_random': test accuracy after removing random clients
        - 'n_clients_remaining': list of remaining client counts
    """
    from fl.server import Server as ServerClass
    from fl.trainer import FLTrainer
    from shapley.in_run_shapley import InRunDataShapley

    n = len(clients)
    # Sort clients by Shapley value
    sorted_by_shapley = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ids_high_first = [cid for cid, _ in sorted_by_shapley]  # best → worst
    sorted_ids_low_first = sorted_ids_high_first[::-1]              # worst → best

    removal_rates = [round(r * 0.1, 1) for r in range(1, 10)]  # 0.1 to 0.9

    accs_high: List[float] = []
    accs_low: List[float] = []
    accs_random: List[float] = []
    n_remaining: List[int] = []

    # Use fewer rounds for efficiency (20% of original)
    fast_rounds = max(int(args.num_rounds * 0.2), 5)
    print(f"\n[ClientRemoval] Running removal experiments with {fast_rounds} rounds each ...")

    for removal_rate in removal_rates:
        n_remove = max(1, int(n * removal_rate))
        n_keep = n - n_remove
        n_remaining.append(n_keep)

        if n_keep < 1:
            accs_high.append(float("nan"))
            accs_low.append(float("nan"))
            accs_random.append(float("nan"))
            continue

        print(f"\n  Removal rate {removal_rate:.1f}: remove {n_remove}, keep {n_keep}")

        # --- High removal: remove top shapley clients ---
        keep_ids_high = set(sorted_ids_high_first[n_remove:])
        acc_high = _retrain_and_eval(
            server, clients, list(keep_ids_high), test_loader, args, fast_rounds
        )
        accs_high.append(acc_high)

        # --- Low removal: remove bottom shapley clients ---
        keep_ids_low = set(sorted_ids_low_first[n_remove:])
        acc_low = _retrain_and_eval(
            server, clients, list(keep_ids_low), test_loader, args, fast_rounds
        )
        accs_low.append(acc_low)

        # --- Random removal: average over 3 trials ---
        all_ids = list(range(n))
        rng = random.Random(args.seed + 999)
        random_accs = []
        for trial in range(3):
            rng.shuffle(all_ids)
            keep_ids_random = set(all_ids[n_remove:])
            acc = _retrain_and_eval(
                server, clients, list(keep_ids_random), test_loader, args, fast_rounds
            )
            random_accs.append(acc)
        accs_random.append(sum(random_accs) / len(random_accs))

        print(f"    High-removal acc: {acc_high:.4f} | "
              f"Low-removal acc: {acc_low:.4f} | "
              f"Random acc: {accs_random[-1]:.4f}")

    return {
        "removal_rates": removal_rates,
        "accs_high": accs_high,
        "accs_low": accs_low,
        "accs_random": accs_random,
        "n_clients_remaining": n_remaining,
    }


def _retrain_and_eval(
    server: "Server",
    all_clients: List["Client"],
    keep_client_ids: List[int],
    test_loader: DataLoader,
    args: argparse.Namespace,
    num_rounds: int,
) -> float:
    """Retrain FL from scratch with a subset of clients and return test accuracy.

    Args:
        server: Reference server (for architecture and val_loader).
        all_clients: Complete list of all client instances.
        keep_client_ids: IDs of clients to include in retraining.
        test_loader: Test DataLoader.
        args: Experiment arguments.
        num_rounds: Number of rounds for this retraining run.

    Returns:
        Final test accuracy as float in [0, 1].
    """
    from models.resnet import get_model
    from data.datasets import get_num_classes
    from fl.server import Server

    # Create a fresh model
    num_classes = get_num_classes(args.dataset)
    fresh_model = get_model(args.model, num_classes, args.dataset)

    # Create a temporary server with the same val_loader
    temp_server = Server(
        model=fresh_model,
        val_loader=server.val_loader,
        device=args.device,
        args=args,
    )

    # Select clients to use
    active_clients = [all_clients[cid] for cid in keep_client_ids if cid < len(all_clients)]

    if not active_clients:
        return 0.0

    clients_per_round = min(args.clients_per_round, len(active_clients))
    rng = random.Random(args.seed + 12345)

    for round_idx in range(1, num_rounds + 1):
        # Sample clients
        selected = rng.sample(active_clients, min(clients_per_round, len(active_clients)))
        global_state = temp_server.get_model_state()

        updates = []
        weights = []
        for client in selected:
            delta = client.local_train(global_state)
            updates.append(delta)
            weights.append(1.0 / len(selected))

        temp_server.aggregate(updates, weights)

    # Final evaluation
    metrics = temp_server.evaluate(test_loader)
    return metrics["test_acc"]
