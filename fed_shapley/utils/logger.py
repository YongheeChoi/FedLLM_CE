"""
utils/logger.py — Unified experiment logging to W&B, TensorBoard, and disk.

All experiment results are saved into a single output directory.
Each experiment is identified by a tag string derived from its key args,
so filenames like ``results_cifar10_c10_iid_r100_e5_s42.json`` make it
easy to distinguish runs at a glance.
"""

import json
import os
import csv
import argparse
from typing import Any, Dict, List, Optional


def make_experiment_tag(args: argparse.Namespace) -> str:
    """Build a short, filesystem-safe tag from the most important args.

    Format: ``{dataset}_c{num_clients}_k{clients_per_round}_{partition}
              [_a{alpha}]_r{rounds}_e{epochs}_lr{lr}
              [_2nd][_noisy{ids}][_qskew]_s{seed}``

    Examples:
        cifar10_c10_k10_iid_r100_e5_lr0.01_s42
        cifar10_c10_k10_dir_a0.5_r100_e5_lr0.01_2nd_s42
        cifar10_c10_k10_iid_r100_e5_lr0.01_noisy0-1_s42
    """
    parts = [
        args.dataset,
        f"c{args.num_clients}",
        f"k{args.clients_per_round}",
    ]

    if args.partition == "dirichlet":
        parts.append(f"dir_a{args.dirichlet_alpha}")
    else:
        parts.append(args.partition)

    parts.extend([
        f"r{args.num_rounds}",
        f"e{args.local_epochs}",
        f"lr{args.local_lr}",
    ])

    if args.use_second_order:
        parts.append("2nd")
    if args.quantity_skew:
        parts.append("qskew")
    if args.noisy_clients:
        ids_str = "-".join(str(i) for i in args.noisy_clients)
        parts.append(f"noisy{ids_str}")

    parts.append(f"s{args.seed}")
    return "_".join(parts)


class ExperimentLogger:
    """Unified logger supporting W&B, TensorBoard, and local file output.

    All files are written into a single ``output_dir`` folder, with
    filenames prefixed by an auto-generated experiment tag so that
    different runs are distinguishable without subdirectories.

    Attributes:
        args: Parsed experiment arguments.
        tag: Short string encoding key hyperparameters.
        round_logs: List of per-round metric dicts accumulated during training.
        shapley_history: List of per-round Shapley value dicts.
        final_summary: Dict of summary metrics set at experiment end.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize all logging backends based on args flags.

        Args:
            args: Parsed argument namespace. Checked attributes:
                - use_wandb, wandb_project, wandb_run_name, wandb_entity
                - use_tensorboard, output_dir, exp_name
        """
        self.args = args
        self.tag = make_experiment_tag(args)
        self.round_logs: List[Dict[str, Any]] = []
        self.shapley_history: List[Dict[int, float]] = []
        self.round_timing_history: List[Dict[str, float]] = []
        self.final_summary: Dict[str, Any] = {}

        # --- Weights & Biases ---
        self._wandb = None
        if getattr(args, "use_wandb", False):
            try:
                import wandb
                run_name = args.wandb_run_name or self.tag
                self._wandb = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    entity=getattr(args, "wandb_entity", None),
                    config=vars(args),
                )
            except Exception as e:
                print(f"[Logger] W&B init failed: {e}. Continuing without W&B.")

        # --- TensorBoard ---
        self._tb_writer = None
        if getattr(args, "use_tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(args.output_dir, "tensorboard", self.tag)
                os.makedirs(tb_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tb_dir)
            except Exception as e:
                print(f"[Logger] TensorBoard init failed: {e}. Continuing without TB.")

    def log_round(self, round_idx: int, metrics_dict: Dict[str, Any]) -> None:
        """Log per-round metrics to all active backends.

        Args:
            round_idx: 1-based communication round index.
            metrics_dict: Flat dict of metric_name -> scalar value.
                Example: {'val_loss': 0.5, 'val_acc': 0.8, 'test_acc': 0.75}
        """
        entry = {"round": round_idx, **metrics_dict}
        self.round_logs.append(entry)

        if self._wandb is not None:
            try:
                self._wandb.log({**metrics_dict, "round": round_idx})
            except Exception:
                pass

        if self._tb_writer is not None:
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, round_idx)

    def log_shapley(self, round_idx: int, shapley_dict: Dict[int, float]) -> None:
        """Record per-round Shapley values (keyed by client_id).

        Args:
            round_idx: Communication round index.
            shapley_dict: {client_id: shapley_value} for this round.
        """
        entry = {"round": round_idx, **{f"client_{k}": v for k, v in shapley_dict.items()}}
        self.shapley_history.append(entry)

        if self._wandb is not None:
            try:
                self._wandb.log({f"shapley/client_{k}": v for k, v in shapley_dict.items()},
                                step=round_idx)
            except Exception:
                pass

        if self._tb_writer is not None:
            for k, v in shapley_dict.items():
                self._tb_writer.add_scalar(f"shapley/client_{k}", v, round_idx)

    def log_round_timing(self, timing_history: List[Dict[str, float]]) -> None:
        """Store per-round timing data for later disk serialization.

        Args:
            timing_history: List of dicts, each containing 'round' and
                phase timing keys (e.g., 'local_training', 'shapley_computation').
        """
        self.round_timing_history = timing_history

    def log_final_summary(self, summary_dict: Dict[str, Any]) -> None:
        """Store and log experiment-level summary metrics.

        Args:
            summary_dict: High-level metrics such as final accuracy,
                Spearman correlation, AUROC, etc.
        """
        self.final_summary.update(summary_dict)

        if self._wandb is not None:
            try:
                self._wandb.summary.update(summary_dict)
            except Exception:
                pass

    def save_to_disk(self, output_dir: str) -> None:
        """Persist all logs and results to a single shared directory.

        All files are written directly into ``output_dir`` (no subdirectories).
        Filenames are prefixed with ``self.tag`` so different experiments
        are distinguishable at a glance.

        Created files:
            {tag}_results.json   — round metrics + final summary + full config
            {tag}_shapley.csv    — per-client cumulative Shapley values
            {tag}_shapley_rounds.csv — per-round Shapley breakdown
            {tag}_timing_rounds.csv — per-round wall-clock timing breakdown
            figures/{tag}_*.png  — visualisation images (created by main.py)

        Args:
            output_dir: Directory path where files will be written.
        """
        os.makedirs(output_dir, exist_ok=True)
        tag = self.tag

        # --- Build serialisable config ---
        config_dict = vars(self.args).copy()
        for k, v in config_dict.items():
            if not isinstance(v, (str, int, float, bool, list, type(None))):
                config_dict[k] = str(v)

        # --- {tag}_results.json ---
        results = {
            "experiment_tag": tag,
            "config": config_dict,
            "round_logs": self.round_logs,
            "final_summary": self.final_summary,
        }
        results_path = os.path.join(output_dir, f"{tag}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # --- {tag}_shapley.csv ---
        if self.shapley_history:
            shapley_path = os.path.join(output_dir, f"{tag}_shapley.csv")
            cumulative: Dict[str, float] = {}
            for entry in self.shapley_history:
                for k, v in entry.items():
                    if k.startswith("client_") and isinstance(v, (int, float)):
                        cumulative[k] = cumulative.get(k, 0.0) + v

            with open(shapley_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["client_id", "cumulative_shapley"])
                for k, v in sorted(cumulative.items()):
                    client_id = k.replace("client_", "")
                    writer.writerow([client_id, v])

        # --- {tag}_shapley_rounds.csv ---
        if self.shapley_history:
            rounds_path = os.path.join(output_dir, f"{tag}_shapley_rounds.csv")
            with open(rounds_path, "w", newline="") as f:
                fieldnames = list(self.shapley_history[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.shapley_history)

        # --- {tag}_timing_rounds.csv ---
        if self.round_timing_history:
            timing_path = os.path.join(output_dir, f"{tag}_timing_rounds.csv")
            with open(timing_path, "w", newline="") as f:
                # Collect all field names across all rounds
                all_timing_fields = []
                seen_fields = set()
                for entry in self.round_timing_history:
                    for k in entry.keys():
                        if k not in seen_fields:
                            all_timing_fields.append(k)
                            seen_fields.add(k)
                writer = csv.DictWriter(f, fieldnames=all_timing_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.round_timing_history)

        print(f"[Logger] Results saved to: {output_dir}/{tag}_*")

    def finish(self) -> None:
        """Close all logging backends gracefully."""
        if self._tb_writer is not None:
            self._tb_writer.close()

        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
