"""
main.py — Entry point for Federated Learning In-Run Data Shapley experiments.

Orchestrates:
1. Argument parsing and seed setting
2. Dataset loading and federated data partitioning
3. Server, client, and Shapley calculator initialization
4. FL training with in-run Shapley computation
5. Optional: exact/MC Shapley, centralized baseline, fidelity evaluation
6. Noisy client detection evaluation
7. Visualization and result persistence

Key formula:
    ϕ_c = Σ_t -η · <∇ℓ(w_t, D_val), Δw_c^(t)>
    (+ optional second-order Hessian correction)
"""

import os
import sys
import copy
import json
import random
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

# ---- Project imports --------------------------------------------------------
from config import get_args
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from utils.timer import CostTracker
from utils.visualize import (
    plot_shapley_bar,
    plot_fidelity_scatter as vis_fidelity_scatter,
    plot_partition_heatmap,
)

from data.datasets import load_dataset, get_num_classes
from data.partition import partition_data, get_client_stats

from models.resnet import get_model

from fl.server import Server
from fl.client import Client
from fl.trainer import FLTrainer

from shapley.in_run_shapley import InRunDataShapley
from shapley.exact_shapley import ExactShapley
from shapley.mc_shapley import MonteCarloShapley

from centralized.centralized_trainer import CentralizedTrainer, IndexedDataset

from eval.fidelity import compute_fidelity, plot_fidelity_scatter
from eval.noisy_client import noisy_client_detection, compute_detection_metrics


def main():
    """Main experiment runner."""

    # =========================================================================
    # 1. Parse arguments and set seed
    # =========================================================================
    args = get_args()
    set_seed(args.seed)

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Main] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # Create output directory (single shared folder for all experiments)
    exp_output_dir = args.output_dir
    os.makedirs(exp_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FedShapley Experiment: {args.exp_name}")
    print(f"  Device: {args.device}")
    print(f"  Dataset: {args.dataset} | Partition: {args.partition}")
    print(f"  Clients: {args.num_clients} | Rounds: {args.num_rounds}")
    print(f"  Second-order: {args.use_second_order}")
    print(f"{'='*60}\n")

    # =========================================================================
    # 2. Load dataset
    # =========================================================================
    print("[Main] Loading dataset ...")
    train_dataset, test_dataset = load_dataset(args.dataset, args.data_dir)
    num_classes = get_num_classes(args.dataset)
    print(f"[Main] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, "
          f"Classes: {num_classes}")

    # =========================================================================
    # 3. Create validation set (extracted from training set)
    # =========================================================================
    # Use last num_val_samples from training set as validation
    # These are excluded from client data
    n_train = len(train_dataset)
    val_indices = list(range(n_train - args.num_val_samples, n_train))
    train_indices_pool = list(range(n_train - args.num_val_samples))

    val_subset = Subset(train_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.local_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.local_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )
    print(f"[Main] Val size: {len(val_subset)}, test size: {len(test_dataset)}")

    # =========================================================================
    # 4. Partition training data across clients
    # =========================================================================
    print(f"[Main] Partitioning data ({args.partition}) ...")

    # Create a sub-dataset containing only train_indices_pool
    pool_dataset = Subset(train_dataset, train_indices_pool)

    # Partition the pool indices (0..len(pool)-1)
    client_local_indices = partition_data(
        dataset=pool_dataset,
        num_clients=args.num_clients,
        partition_type=args.partition,
        num_classes=num_classes,
        dirichlet_alpha=args.dirichlet_alpha,
        quantity_skew=args.quantity_skew,
        quantity_beta=args.quantity_beta,
        seed=args.seed,
    )

    # Map local pool indices back to global dataset indices
    client_global_indices: Dict[int, List[int]] = {}
    for cid, local_idxs in client_local_indices.items():
        client_global_indices[cid] = [train_indices_pool[i] for i in local_idxs]

    # Print partition stats
    client_stats = get_client_stats(client_local_indices, pool_dataset, num_classes)
    for cid, stat in client_stats.items():
        marker = " [NOISY]" if cid in (args.noisy_clients or []) else ""
        print(f"  Client {cid:3d}{marker}: {stat['num_samples']} samples, "
              f"{stat['num_represented_classes']} classes, "
              f"dominant={stat['dominant_class']}")

    # =========================================================================
    # 5. Create model, server, and clients
    # =========================================================================
    print("[Main] Creating model, server, and clients ...")
    global_model = get_model(args.model, num_classes, args.dataset)
    global_model = global_model.to(args.device)

    server = Server(
        model=global_model,
        val_loader=val_loader,
        device=args.device,
        args=args,
    )

    # Create per-client DataLoaders
    clients: List[Client] = []
    for cid in range(args.num_clients):
        client_subset = Subset(train_dataset, client_global_indices[cid])
        client_loader = DataLoader(
            client_subset,
            batch_size=args.local_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(args.device == "cuda"),
            drop_last=False,
        )
        client = Client(
            client_id=cid,
            train_loader=client_loader,
            device=args.device,
            args=args,
            model_template=global_model,
        )
        clients.append(client)

    print(f"[Main] Created {len(clients)} clients.")

    # =========================================================================
    # 5b. Initialize cost tracker and estimate model FLOPs
    # =========================================================================
    cost_tracker = CostTracker()

    # Determine input shape for FLOPs estimation
    if args.dataset in ("cifar10", "cifar100"):
        input_shape = (1, 3, 32, 32)
    elif args.dataset == "tinyimagenet":
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)

    flops = cost_tracker.estimate_model_flops(global_model, input_shape, args.device)
    if flops:
        from utils.timer import _format_flops
        print(f"[Main] Estimated FLOPs per forward pass: {_format_flops(flops)}")

    # =========================================================================
    # 6. Create Shapley calculators
    # =========================================================================
    in_run_shapley = InRunDataShapley(server=server, args=args)

    ground_truth_shapley: Optional[object] = None
    if args.run_exact_shapley:
        if args.num_clients <= 10:
            ground_truth_shapley = ExactShapley(server=server, args=args)
            print(f"[Main] Using ExactShapley (n={args.num_clients} <= 10).")
        else:
            ground_truth_shapley = MonteCarloShapley(server=server, args=args)
            print(f"[Main] Using MonteCarloShapley (n={args.num_clients} > 10, "
                  f"T={args.mc_permutations} permutations).")

    # =========================================================================
    # 7. Create logger
    # =========================================================================
    logger = ExperimentLogger(args)

    # =========================================================================
    # 8. Run FL training
    # =========================================================================
    print(f"\n[Main] Starting FL training ...")
    cost_tracker.start("fl_training_total")
    trainer = FLTrainer(
        server=server,
        clients=clients,
        shapley_calculator=in_run_shapley,
        test_loader=test_loader,
        args=args,
        logger=logger,
        cost_tracker=cost_tracker,
    )
    fl_results = trainer.train()
    cost_tracker.stop("fl_training_total")

    cumulative_shapley = fl_results["cumulative_shapley"]
    final_metrics = fl_results.get("final_metrics", {})

    # Pass per-round timing data to logger
    if "round_timing_history" in fl_results:
        logger.log_round_timing(fl_results["round_timing_history"])

    print(f"\n[Main] FL Training complete.")
    print(f"  Final val_acc:  {final_metrics.get('val_acc', 'N/A'):.4f}")
    print(f"  Final test_acc: {final_metrics.get('test_acc', 'N/A'):.4f}")

    # =========================================================================
    # 9. Optional: Ground-truth Shapley (exact or MC)
    # =========================================================================
    gt_shapley_values: Optional[Dict[int, float]] = None

    if ground_truth_shapley is not None:
        print(f"\n[Main] Computing ground-truth Shapley values ...")
        cost_tracker.start("gt_shapley")
        # Run one pass of ground-truth Shapley using all client updates
        # from the final round (approximation: use current model as w_T)
        global_state = server.get_model_state()

        # Collect updates from all clients for a final ground-truth pass
        all_updates = []
        all_ids = list(range(args.num_clients))
        trainable_names = server.get_trainable_param_names()
        from fl.trainer import _flatten_update_trainable
        for cid in all_ids:
            delta = clients[cid].local_train(global_state)
            flat_delta = _flatten_update_trainable(delta, trainable_names)
            all_updates.append(flat_delta)

        if isinstance(ground_truth_shapley, ExactShapley):
            gt_round = ground_truth_shapley.compute_round_exact_shapley(
                client_updates=all_updates,
                client_ids=all_ids,
                eta=args.local_lr,
                round_idx=0,
            )
            # ExactShapley: 2^n subsets, each needs val forward pass
            n = args.num_clients
            val_size = len(server.val_loader.dataset)
            bs = args.local_batch_size
            val_batches = (val_size + bs - 1) // bs
            cost_tracker.add_forward_passes("gt_shapley", (2**n) * val_batches)
        else:
            gt_round = ground_truth_shapley.compute_round_mc_shapley(
                client_updates=all_updates,
                client_ids=all_ids,
                eta=args.local_lr,
                round_idx=0,
            )
            # MC Shapley: T permutations × n positions, each needs val forward pass
            val_size = len(server.val_loader.dataset)
            bs = args.local_batch_size
            val_batches = (val_size + bs - 1) // bs
            cost_tracker.add_forward_passes(
                "gt_shapley", args.mc_permutations * args.num_clients * val_batches
            )
        ground_truth_shapley.accumulate(gt_round)
        gt_shapley_values = ground_truth_shapley.get_cumulative()
        cost_tracker.stop("gt_shapley")

        print("[Main] Ground-truth Shapley values:")
        for cid, sv in sorted(gt_shapley_values.items(), key=lambda x: x[1], reverse=True):
            print(f"  Client {cid}: phi_GT = {sv:+.6f}")

    # =========================================================================
    # 10. Optional: Centralized training baseline
    # =========================================================================
    centralized_shapley: Optional[Dict[int, float]] = None

    if args.run_centralized:
        print("\n[Main] Running centralized training for baseline comparison ...")
        cost_tracker.start("centralized_training")
        # Create indexed dataset for sample → client attribution
        indexed_train = IndexedDataset(Subset(train_dataset, train_indices_pool))
        central_loader = DataLoader(
            indexed_train,
            batch_size=args.local_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(args.device == "cuda"),
        )
        fresh_model = get_model(args.model, num_classes, args.dataset)
        central_trainer = CentralizedTrainer(
            model=fresh_model,
            train_loader=central_loader,
            val_loader=val_loader,
            client_data_indices=client_global_indices,
            args=args,
        )
        centralized_shapley = central_trainer.train_and_compute_shapley()
        cost_tracker.stop("centralized_training")

        # Estimate centralized FLOPs: per batch = val_grad + per-sample attribution + batch update
        total_epochs = max(args.num_rounds * args.local_epochs // 10, 5)
        pool_size = len(train_indices_pool)
        bs = args.local_batch_size
        batches_per_epoch = (pool_size + bs - 1) // bs
        val_batches = (args.num_val_samples + bs - 1) // bs
        # Each batch: val_grad(val_batches fwd) + B per-sample fwd + 1 batch fwd
        total_central_fwd = total_epochs * batches_per_epoch * (val_batches + bs + 1)
        cost_tracker.add_forward_passes("centralized_training", total_central_fwd)

    # =========================================================================
    # 11. Fidelity evaluation
    # =========================================================================
    fidelity_metrics: Optional[Dict] = None

    if gt_shapley_values is not None:
        print("\n[Main] Computing fidelity metrics (in-run vs ground-truth) ...")
        fidelity_metrics = compute_fidelity(cumulative_shapley, gt_shapley_values)
        logger.log_final_summary({"fidelity/" + k: v for k, v in fidelity_metrics.items()})

    if centralized_shapley is not None:
        print("\n[Main] Computing fidelity metrics (in-run vs centralized) ...")
        fidelity_central = compute_fidelity(cumulative_shapley, centralized_shapley)
        logger.log_final_summary({"fidelity_central/" + k: v for k, v in fidelity_central.items()})

    if centralized_shapley is not None and gt_shapley_values is not None:
        print("\n[Main] Computing fidelity metrics (centralized vs ground-truth) ...")
        fidelity_cent_gt = compute_fidelity(centralized_shapley, gt_shapley_values)
        logger.log_final_summary({"fidelity_cent_gt/" + k: v for k, v in fidelity_cent_gt.items()})

    # =========================================================================
    # 12. Noisy client detection
    # =========================================================================
    if args.noisy_clients:
        print(f"\n[Main] Evaluating noisy client detection ...")
        all_client_ids = list(range(args.num_clients))
        detection_metrics = compute_detection_metrics(
            shapley_values=cumulative_shapley,
            noisy_client_ids=args.noisy_clients,
            all_client_ids=all_client_ids,
        )
        logger.log_final_summary({"detection/" + k: v for k, v in detection_metrics.items()})
        print(f"  AUROC: {detection_metrics['auroc']:.4f}")
        print(f"  Precision@k: {detection_metrics['precision_at_k']:.4f}")

    # =========================================================================
    # 13. Final summary logging (including cost tracking)
    # =========================================================================
    cost_summary = cost_tracker.get_summary()
    cost_tracker.print_summary()

    summary = {
        "final_val_acc": final_metrics.get("val_acc", None),
        "final_test_acc": final_metrics.get("test_acc", None),
        "num_rounds": args.num_rounds,
        "num_clients": args.num_clients,
        "partition": args.partition,
    }
    if fidelity_metrics:
        summary.update({f"fidelity_{k}": v for k, v in fidelity_metrics.items()})

    # Add cost tracking to summary
    summary.update(cost_summary)

    logger.log_final_summary(summary)

    # =========================================================================
    # 14. Visualizations (saved into figures/ subfolder with tag prefix)
    # =========================================================================
    tag = logger.tag
    vis_dir = os.path.join(exp_output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)

    # Shapley bar chart
    if cumulative_shapley:
        plot_shapley_bar(
            shapley_values=cumulative_shapley,
            title=f"In-Run Shapley Values [{tag}]",
            save_path=os.path.join(vis_dir, f"{tag}_shapley_bar.png"),
        )

    # Fidelity scatter (in-run vs ground-truth)
    if gt_shapley_values and len(cumulative_shapley) >= 3:
        common_ids = sorted(set(cumulative_shapley.keys()) & set(gt_shapley_values.keys()))
        if len(common_ids) >= 3:
            in_run_list = [cumulative_shapley[c] for c in common_ids]
            gt_list = [gt_shapley_values[c] for c in common_ids]
            plot_fidelity_scatter(
                in_run=in_run_list,
                ground_truth=gt_list,
                title=f"In-Run vs Ground-Truth [{tag}]",
                save_path=os.path.join(vis_dir, f"{tag}_fidelity_scatter.png"),
            )

    # Partition heatmap (shows class distribution per client)
    if hasattr(train_dataset, "targets"):
        import numpy as np
        all_labels = list(train_dataset.targets)
        client_labels_for_vis = {}
        for cid, global_idxs in client_global_indices.items():
            client_labels_for_vis[cid] = [all_labels[i] for i in global_idxs[:500]]
        plot_partition_heatmap(
            client_data=client_labels_for_vis,
            num_classes=num_classes,
            save_path=os.path.join(vis_dir, f"{tag}_partition_heatmap.png"),
        )

    # =========================================================================
    # 15. Save results to disk
    # =========================================================================
    logger.save_to_disk(exp_output_dir)
    logger.finish()

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  Experiment '{args.exp_name}' completed.")
    print(f"  Output: {exp_output_dir}")
    print(f"  Final test accuracy: {final_metrics.get('test_acc', 'N/A'):.4f}")
    if fidelity_metrics:
        print(f"  Shapley fidelity (Spearman rho): {fidelity_metrics.get('spearman_r', 'N/A'):.4f}")
    # Print key timing info
    fl_time = cost_tracker.timings.get("fl_training_total", 0)
    train_time = cost_tracker.timings.get("local_training", 0)
    shap_time = cost_tracker.timings.get("shapley_computation", 0)
    print(f"  FL training total: {fl_time:.1f}s "
          f"(training: {train_time:.1f}s, shapley: {shap_time:.1f}s, "
          f"overhead: {shap_time / max(train_time, 1e-6) * 100:.1f}%)")
    if cost_tracker.timings.get("gt_shapley"):
        print(f"  Ground-truth Shapley: {cost_tracker.timings['gt_shapley']:.1f}s")
    if cost_tracker.timings.get("centralized_training"):
        print(f"  Centralized training: {cost_tracker.timings['centralized_training']:.1f}s")
    print(f"{'='*60}\n")

    # Return results dict for programmatic access
    return {
        "cumulative_shapley": cumulative_shapley,
        "ground_truth_shapley": gt_shapley_values,
        "centralized_shapley": centralized_shapley,
        "fidelity_metrics": fidelity_metrics,
        "final_metrics": final_metrics,
        "cost_summary": cost_summary,
        "output_dir": exp_output_dir,
    }


if __name__ == "__main__":
    results = main()
