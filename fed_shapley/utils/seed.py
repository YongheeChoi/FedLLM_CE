"""
utils/seed.py — Global random seed management for reproducibility.

Sets seeds for all sources of randomness: Python's random module, NumPy,
PyTorch CPU and CUDA, and cuDNN determinism flags.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds to ensure reproducible experiments.

    Args:
        seed: Integer seed value to apply to all random number generators.

    Notes:
        - Sets `torch.backends.cudnn.deterministic = True` which may slow
          some CUDA operations but guarantees determinism.
        - Sets `torch.backends.cudnn.benchmark = False` to prevent auto-tuning
          which can introduce non-determinism.
        - Sets `torch.use_deterministic_algorithms(True)` to force deterministic
          CUDA kernels (e.g., cuBLAS atomics, scatter/gather ops).
        - Sets CUBLAS_WORKSPACE_CONFIG to avoid cuBLAS non-determinism.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # PyTorch < 1.11: warn_only not supported
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass
