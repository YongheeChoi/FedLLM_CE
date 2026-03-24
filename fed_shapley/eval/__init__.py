"""eval — Evaluation utilities: fidelity, client removal, and noisy client detection."""

from .fidelity import compute_fidelity, plot_fidelity_scatter
from .client_removal import client_removal_experiment
from .noisy_client import noisy_client_detection

__all__ = [
    "compute_fidelity",
    "plot_fidelity_scatter",
    "client_removal_experiment",
    "noisy_client_detection",
]
