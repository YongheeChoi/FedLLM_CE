"""
fl/trainer.py — Main Federated Learning training loop.

Coordinates server and clients through communication rounds:
1. Sample clients for this round
2. Distribute global model state
3. Collect local weight updates Δw_c
4. Compute per-round Shapley values
5. FedAvg aggregation
6. Periodic evaluation and logging

The trainer returns a comprehensive results dict at the end of training.
"""

import argparse
import copy
import random
import time
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .server import Server
from .client import Client


class FLTrainer:
    """Orchestrates the FL training loop with Shapley value computation.

    Attributes:
        server: The global FL server.
        clients: List of all FL client instances.
        shapley_calculator: In-run Shapley engine (InRunDataShapley or compatible).
        test_loader: Test DataLoader for held-out evaluation.
        args: Experiment configuration.
        metrics_history: List of per-round metric dicts.
    """

    def __init__(
        self,
        server: Server,
        clients: List[Client],
        shapley_calculator: Any,
        test_loader: DataLoader,
        args: argparse.Namespace,
        logger: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
    ) -> None:
        """Initialize the FL trainer.

        Args:
            server: Initialized Server instance with global model.
            clients: List of Client instances (length = num_clients).
            shapley_calculator: An object with methods:
                - compute_round_shapley(client_updates, client_ids, eta, round_idx)
                - accumulate(round_shapley)
                - get_cumulative()
            test_loader: DataLoader for test set evaluation.
            args: Experiment arguments. Used fields:
                - num_rounds, clients_per_round, eval_every, log_every,
                  local_lr, aggregation, seed
            logger: Optional ExperimentLogger for W&B/TensorBoard/disk logging.
            cost_tracker: Optional CostTracker for timing and FLOPs tracking.
        """
        self.server = server
        self.clients = clients
        self.shapley_calculator = shapley_calculator
        self.test_loader = test_loader
        self.args = args
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.metrics_history: List[Dict[str, Any]] = []

    def train(self) -> Dict[str, Any]:
        """Run the complete FL training loop.

        For each round t:
        1. Randomly sample clients_per_round clients without replacement.
        2. Send global model state to each selected client.
        3. Collect local updates Δw_c from each client.
        4. Compute per-client weights (equal or data-proportional).
        5. Compute In-Run Shapley values for this round.
        6. FedAvg aggregation to update the global model.
        7. Optionally evaluate and log every eval_every rounds.

        Returns:
            Dict containing:
            - 'cumulative_shapley': {client_id: total_shapley}
            - 'round_shapley_history': list of per-round shapley dicts
            - 'metrics_history': list of per-round metric dicts
            - 'final_metrics': dict with final val/test loss/acc
        """
        rng = random.Random(self.args.seed)
        ct = self.cost_tracker  # may be None

        all_client_ids = list(range(len(self.clients)))
        clients_per_round = min(self.args.clients_per_round, len(self.clients))

        print(f"\n[FLTrainer] Starting FL training: "
              f"{self.args.num_rounds} rounds, "
              f"{clients_per_round}/{len(self.clients)} clients/round")

        final_metrics: Dict[str, float] = {}
        round_timing_history: List[Dict[str, float]] = []

        pbar = tqdm(range(1, self.args.num_rounds + 1), desc="FL Rounds")

        for round_idx in pbar:
            round_times: Dict[str, float] = {"round": round_idx}

            # ------------------------------------------------------------------
            # Step 1: Sample clients for this round
            # ------------------------------------------------------------------
            selected_ids = rng.sample(all_client_ids, clients_per_round)
            selected_clients = [self.clients[cid] for cid in selected_ids]

            # ------------------------------------------------------------------
            # Step 2: Get current global model state
            # ------------------------------------------------------------------
            global_state = self.server.get_model_state()

            # ------------------------------------------------------------------
            # Step 3: Local training — collect Δw_c from each client
            # ------------------------------------------------------------------
            if ct:
                ct.start("local_training")
            client_updates: List[Dict[str, torch.Tensor]] = []
            data_sizes: List[int] = []

            round_fwd_passes = 0
            for client in selected_clients:
                delta = client.local_train(global_state)
                client_updates.append(delta)
                ds = client.get_data_size()
                data_sizes.append(ds)
                # Estimate forward passes: epochs × ceil(data_size / batch_size)
                bs = self.args.local_batch_size
                batches_per_epoch = (ds + bs - 1) // bs
                round_fwd_passes += self.args.local_epochs * batches_per_epoch

            if ct:
                round_times["local_training"] = ct.stop("local_training")
                ct.add_forward_passes("local_training", round_fwd_passes)

            # ------------------------------------------------------------------
            # Step 4: Compute aggregation weights
            # ------------------------------------------------------------------
            total_data = sum(data_sizes)
            equal_weights = [1.0 / clients_per_round] * clients_per_round
            data_weights = [n / total_data for n in data_sizes]
            agg_weights = equal_weights

            # ------------------------------------------------------------------
            # Step 5: Compute per-round Shapley values
            # ------------------------------------------------------------------
            if ct:
                ct.start("shapley_computation")

            trainable_names = self.server.get_trainable_param_names()
            flat_updates = [
                _flatten_update_trainable(delta, trainable_names)
                for delta in client_updates
            ]

            eta = self.args.local_lr

            round_shapley = self.shapley_calculator.compute_round_shapley(
                client_updates=flat_updates,
                client_ids=selected_ids,
                eta=eta,
                round_idx=round_idx,
            )
            self.shapley_calculator.accumulate(round_shapley)

            if ct:
                round_times["shapley_computation"] = ct.stop("shapley_computation")
                # Shapley needs 1 val gradient pass (all val batches) + optional HVP
                val_size = len(self.server.val_loader.dataset)
                bs = self.args.local_batch_size
                val_batches = (val_size + bs - 1) // bs
                shapley_fwd = val_batches  # 1 forward for val gradient
                if getattr(self.args, "use_second_order", False):
                    shapley_fwd += val_batches  # HVP needs another forward+backward
                ct.add_forward_passes("shapley_computation", shapley_fwd)

            # Log per-round Shapley
            if self.logger is not None and round_idx % self.args.log_every == 0:
                self.logger.log_shapley(round_idx, round_shapley)

            # ------------------------------------------------------------------
            # Step 6: FedAvg aggregation
            # ------------------------------------------------------------------
            if ct:
                ct.start("aggregation")
            self.server.aggregate(client_updates, agg_weights)
            if ct:
                round_times["aggregation"] = ct.stop("aggregation")

            # ------------------------------------------------------------------
            # Step 7: Evaluation and logging
            # ------------------------------------------------------------------
            if round_idx % self.args.eval_every == 0:
                if ct:
                    ct.start("evaluation")
                metrics = self.server.evaluate(self.test_loader)
                if ct:
                    round_times["evaluation"] = ct.stop("evaluation")
                    eval_fwd = (len(self.server.val_loader.dataset) + bs - 1) // bs
                    eval_fwd += (len(self.test_loader.dataset) + bs - 1) // bs
                    ct.add_forward_passes("evaluation", eval_fwd)

                metrics["round"] = round_idx
                self.metrics_history.append(metrics)
                final_metrics = metrics

                if self.logger is not None and round_idx % self.args.log_every == 0:
                    self.logger.log_round(round_idx, {
                        k: v for k, v in metrics.items() if k != "round"
                    })

                pbar.set_postfix({
                    "val_acc": f"{metrics['val_acc']:.3f}",
                    "test_acc": f"{metrics['test_acc']:.3f}",
                })

            round_timing_history.append(round_times)

        cumulative_shapley = self.shapley_calculator.get_cumulative()

        # Rank clients by Shapley value
        ranked = sorted(cumulative_shapley.items(), key=lambda x: x[1], reverse=True)
        print("\n[FLTrainer] Final Cumulative Shapley Values:")
        for rank, (cid, sv) in enumerate(ranked, 1):
            marker = " [NOISY]" if cid in (self.args.noisy_clients or []) else ""
            print(f"  Rank {rank}: Client {cid:3d} -> phi = {sv:+.6f}{marker}")

        return {
            "cumulative_shapley": cumulative_shapley,
            "round_shapley_history": self.shapley_calculator.round_history,
            "metrics_history": self.metrics_history,
            "final_metrics": final_metrics,
            "round_timing_history": round_timing_history,
        }


def _flatten_update_trainable(
    state_dict: Dict[str, torch.Tensor],
    trainable_names: List[str],
) -> torch.Tensor:
    """Flatten only trainable-parameter entries of a state_dict to a 1D vector.

    Args:
        state_dict: Dict of parameter_name -> tensor (Δw_c from a client).
        trainable_names: Ordered list of parameter names that have
            requires_grad=True in the global model.  This ensures the
            flattened vector matches the validation gradient's layout.

    Returns:
        1D float tensor of shape (P,) where P = total trainable parameters.

    Notes:
        BatchNorm running_mean / running_var are excluded because they have
        no gradient and are not part of the Shapley dot-product computation.
    """
    parts = []
    for name in trainable_names:
        if name in state_dict:
            parts.append(state_dict[name].reshape(-1).float())
        else:
            raise KeyError(f"Trainable param '{name}' missing from client update")
    return torch.cat(parts) if parts else torch.tensor([])
