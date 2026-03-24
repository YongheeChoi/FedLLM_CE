"""
fl/client.py — FL Client performing local SGD training.

Each client trains on its private local dataset for E epochs and returns
the weight update Δw_c = w_local - w_global to the server.

Noisy clients optionally apply label flipping during local training:
    label → (num_classes - 1 - label)

This simulates a Byzantine attacker or data corruption scenario.
"""

import argparse
import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:
    """FL client performing local SGD and returning weight deltas.

    Attributes:
        client_id: Unique integer identifier for this client.
        train_loader: DataLoader for this client's private training data.
        device: Compute device string.
        args: Parsed experiment arguments.
        is_noisy: Whether this client applies noise during training.
        num_classes: Number of output classes (inferred from args or model).
        _model: Local model instance (created once and reused).
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        device: str,
        args: argparse.Namespace,
        model_template: nn.Module,
    ) -> None:
        """Initialize FL client.

        Args:
            client_id: Unique client identifier.
            train_loader: DataLoader wrapping this client's data subset.
            device: Target compute device ('cuda' or 'cpu').
            args: Experiment configuration. Used fields:
                - local_epochs, local_lr, noisy_clients, noise_type
            model_template: An nn.Module instance whose architecture will be
                cloned for local training. State dict is overwritten each round.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.args = args
        self.is_noisy = client_id in (args.noisy_clients or [])

        # Create a local model clone with the same architecture
        self._model = copy.deepcopy(model_template).to(device)
        self._criterion = nn.CrossEntropyLoss()

        # Infer num_classes from the model's final linear layer
        self.num_classes: Optional[int] = None
        for module in reversed(list(self._model.modules())):
            if isinstance(module, nn.Linear):
                self.num_classes = module.out_features
                break

    def local_train(
        self, global_model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform local SGD training starting from the global model state.

        Steps:
        1. Load global_model_state into the local model.
        2. Save a copy of the initial state (w_global).
        3. Train for args.local_epochs using SGD with lr=args.local_lr.
        4. If noisy and noise_type='label_flip': flip labels during training.
        5. Return Δw_c = w_local - w_global as a CPU state_dict.

        Args:
            global_model_state: Server's current global model state_dict
                (deep copy, CPU tensors).

        Returns:
            Dict[str, torch.Tensor]: Δw_c = w_local - w_global.
                All tensors are on CPU. Parameters where local == global
                will have zero tensors.

        Notes:
            - Random_update noise replaces gradients with random noise to
              simulate a client sending garbage updates.
            - Momentum state is reset each round (no carry-over between rounds).
        """
        # Step 1: Load global weights
        self._model.load_state_dict(
            {k: v.to(self.device) for k, v in global_model_state.items()}
        )

        # Step 2: Snapshot initial weights (w_global) for delta computation
        w_global = {k: v.clone().cpu() for k, v in self._model.state_dict().items()}

        if self.args.noise_type == "random_update" and self.is_noisy:
            # Return a random noise delta immediately (no real training)
            delta = {}
            for k, v in w_global.items():
                if v.dtype.is_floating_point:
                    delta[k] = torch.randn_like(v) * 0.01
                else:
                    delta[k] = torch.zeros_like(v)
            return delta

        # Step 3: Local SGD training
        self._model.train()
        optimizer = optim.SGD(
            self._model.parameters(),
            lr=self.args.local_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        for epoch in range(self.args.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                # Step 4: Apply label flip noise if this client is noisy
                if self.is_noisy and self.args.noise_type == "label_flip":
                    if self.num_classes is not None:
                        # label → (num_classes - 1 - label): full label flip
                        y = (self.num_classes - 1 - y).clamp(0, self.num_classes - 1)

                optimizer.zero_grad()
                logits = self._model(X)
                loss = self._criterion(logits, y)
                loss.backward()
                optimizer.step()

        # Step 5: Compute Δw_c = w_local - w_global
        w_local = self._model.state_dict()
        delta: Dict[str, torch.Tensor] = {}
        for name, param in w_local.items():
            if param.dtype.is_floating_point:
                delta[name] = param.cpu().float() - w_global[name].float()
            else:
                # Non-float params (e.g., batch norm num_batches_tracked)
                delta[name] = torch.zeros_like(w_global[name])

        return delta

    def get_data_size(self) -> int:
        """Return the number of training samples held by this client.

        Returns:
            Integer count of training samples in this client's dataset.
        """
        return len(self.train_loader.dataset)
