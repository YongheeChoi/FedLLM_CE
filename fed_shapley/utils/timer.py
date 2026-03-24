"""
utils/timer.py — Wall-clock timing and FLOPs estimation for cost tracking.

Provides CostTracker for measuring per-phase execution time and estimating
computational cost (FLOPs) based on model forward/backward pass counts.
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn


class CostTracker:
    """Tracks wall-clock time and estimated FLOPs for experiment phases.

    Usage:
        tracker = CostTracker()
        tracker.start("local_training")
        ...
        tracker.stop("local_training")

        # Or as context manager:
        with tracker.track("shapley_computation"):
            ...

    Attributes:
        timings: Dict of phase_name -> total elapsed seconds.
        counts: Dict of phase_name -> number of times the phase was measured.
        flops_per_forward: Estimated FLOPs for one model forward pass.
        forward_pass_counts: Dict of phase_name -> total forward passes counted.
    """

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._starts: Dict[str, float] = {}
        self.flops_per_forward: Optional[int] = None
        self.forward_pass_counts: Dict[str, int] = {}

    def start(self, name: str) -> None:
        """Start timing a named phase.

        Args:
            name: Phase identifier (e.g., 'local_training', 'shapley').
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a named phase and accumulate elapsed time.

        Args:
            name: Phase identifier matching a previous start() call.

        Returns:
            Elapsed time in seconds for this single measurement.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._starts.pop(name)
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1
        return elapsed

    def track(self, name: str) -> "_TrackContext":
        """Context manager for timing a code block.

        Args:
            name: Phase identifier.

        Returns:
            Context manager that starts/stops timing automatically.
        """
        return _TrackContext(self, name)

    def add_forward_passes(self, name: str, count: int) -> None:
        """Record number of forward passes for a phase (for FLOPs estimation).

        Args:
            name: Phase identifier.
            count: Number of forward passes performed.
        """
        self.forward_pass_counts[name] = (
            self.forward_pass_counts.get(name, 0) + count
        )

    def estimate_model_flops(
        self, model: nn.Module, input_shape: tuple, device: str
    ) -> Optional[int]:
        """Estimate FLOPs for a single forward pass using PyTorch's FlopCounterMode.

        Falls back to a parameter-count-based estimate (2 * params) if
        FlopCounterMode is unavailable (PyTorch < 2.1).

        Args:
            model: The model to profile.
            input_shape: Input tensor shape including batch dim, e.g. (1, 3, 32, 32).
            device: Compute device string.

        Returns:
            Estimated FLOPs (int) or None if estimation fails.
        """
        # Try PyTorch's built-in FlopCounterMode (available since 2.1)
        try:
            from torch.utils.flop_counter import FlopCounterMode

            model.eval()
            dummy = torch.randn(*input_shape, device=device)
            with FlopCounterMode(display=False) as flop_counter:
                model(dummy)
            total_flops = flop_counter.get_total_flops()
            self.flops_per_forward = int(total_flops)
            return self.flops_per_forward
        except (ImportError, AttributeError, Exception):
            pass

        # Fallback: estimate from parameter count
        # Rough approximation: 2 * total_params per forward pass (multiply-add)
        total_params = sum(p.numel() for p in model.parameters())
        self.flops_per_forward = 2 * total_params
        return self.flops_per_forward

    def get_estimated_flops(self, name: str) -> Optional[int]:
        """Get estimated total FLOPs for a phase.

        Computed as: forward_passes * flops_per_forward * 3
        (×3 accounts for backward pass ≈ 2× forward, total ≈ 3× forward)

        Args:
            name: Phase identifier.

        Returns:
            Estimated FLOPs (int) or None if not available.
        """
        if self.flops_per_forward is None:
            return None
        fwd_count = self.forward_pass_counts.get(name, 0)
        if fwd_count == 0:
            return None
        # forward + backward ≈ 3× forward FLOPs
        return fwd_count * self.flops_per_forward * 3

    def get_summary(self) -> Dict[str, object]:
        """Return a summary dict of all tracked timings and FLOPs.

        Returns:
            Dict with keys:
            - 'time/{name}_total_sec': total time for phase
            - 'time/{name}_avg_sec': average time per invocation
            - 'time/{name}_count': invocation count
            - 'flops/{name}_forward_passes': forward pass count
            - 'flops/{name}_estimated_total': estimated total FLOPs
            - 'flops/per_forward': FLOPs per single forward pass
        """
        summary: Dict[str, object] = {}

        for name, total in self.timings.items():
            count = self.counts.get(name, 1)
            summary[f"time/{name}_total_sec"] = round(total, 4)
            summary[f"time/{name}_avg_sec"] = round(total / max(count, 1), 4)
            summary[f"time/{name}_count"] = count

        if self.flops_per_forward is not None:
            summary["flops/per_forward"] = self.flops_per_forward

        for name, fwd_count in self.forward_pass_counts.items():
            summary[f"flops/{name}_forward_passes"] = fwd_count
            est = self.get_estimated_flops(name)
            if est is not None:
                summary[f"flops/{name}_estimated_total"] = est

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of all tracked costs."""
        print(f"\n{'='*60}")
        print("  Cost Summary (Wall-Clock Time & Estimated FLOPs)")
        print(f"{'='*60}")

        # Timing summary
        if self.timings:
            print("\n  [Time]")
            for name in sorted(self.timings.keys()):
                total = self.timings[name]
                count = self.counts.get(name, 1)
                avg = total / max(count, 1)
                print(f"    {name:30s}: {total:10.2f}s total "
                      f"({avg:.4f}s avg × {count} calls)")

            total_all = sum(self.timings.values())
            print(f"    {'--- all phases ---':30s}: {total_all:10.2f}s")

        # FLOPs summary
        if self.forward_pass_counts:
            print("\n  [FLOPs]")
            if self.flops_per_forward is not None:
                print(f"    Per forward pass: {_format_flops(self.flops_per_forward)}")
            for name in sorted(self.forward_pass_counts.keys()):
                fwd = self.forward_pass_counts[name]
                est = self.get_estimated_flops(name)
                est_str = _format_flops(est) if est else "N/A"
                print(f"    {name:30s}: {fwd:>8d} fwd passes -> {est_str} (est.)")

        print(f"{'='*60}\n")


class _TrackContext:
    """Context manager returned by CostTracker.track()."""

    def __init__(self, tracker: CostTracker, name: str) -> None:
        self._tracker = tracker
        self._name = name

    def __enter__(self) -> "_TrackContext":
        self._tracker.start(self._name)
        return self

    def __exit__(self, *exc) -> None:
        self._tracker.stop(self._name)


def _format_flops(flops: int) -> str:
    """Format FLOPs into human-readable string.

    Args:
        flops: Raw FLOPs count.

    Returns:
        Formatted string like '1.23 GFLOPs' or '456.7 MFLOPs'.
    """
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    return f"{flops} FLOPs"
