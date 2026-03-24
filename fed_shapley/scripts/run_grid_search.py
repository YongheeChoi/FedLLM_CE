"""
scripts/run_grid_search.py — Grid search runner for hyperparameter sweeps.

Reads a YAML config specifying base_args and a grid of parameters to sweep,
generates all combinations, and runs them sequentially. Results are aggregated
into a CSV summary file.

Usage:
    python scripts/run_grid_search.py --config scripts/grid_configs/alpha_sweep.yaml
    python scripts/run_grid_search.py --config scripts/grid_configs/k_sweep.yaml --dry_run
    python scripts/run_grid_search.py --config scripts/grid_configs/noise_sweep.yaml --max_runs 6

Config YAML format:
    base_args:
      dataset: cifar10
      num_clients: 10
      ...
    grid:
      dirichlet_alpha: [0.1, 0.5, 1.0]
      seed: [42, 123]

Args:
    --config: Path to YAML grid config file (required).
    --dry_run: Print commands without executing them.
    --max_runs: Limit total number of experiments.
    --output_root: Root directory for experiment outputs (default: ./outputs/grid).
"""

import itertools
import subprocess
import yaml
import csv
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse


def load_config(path: str) -> dict:
    """Load YAML config file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML as dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def generate_experiments(base_args: dict, grid: dict) -> list:
    """Generate all experiment configurations from base_args and grid.

    Args:
        base_args: Default argument values shared by all experiments.
        grid: Dict of {arg_name: [value1, value2, ...]} to sweep over.

    Returns:
        List of dicts, each representing one experiment configuration.
    """
    keys = list(grid.keys())
    values = list(grid.values())
    experiments = []
    for combo in itertools.product(*values):
        exp = dict(base_args)
        exp.update(dict(zip(keys, combo)))
        experiments.append(exp)
    return experiments


def make_run_name(grid_args: dict, grid_keys: list) -> str:
    """Create a human-readable experiment identifier from grid parameters.

    Args:
        grid_args: Full experiment argument dict.
        grid_keys: Keys that vary in the grid (used to build the name).

    Returns:
        String like "dirichlet_alpha0.1_seed42".
    """
    parts = []
    for k in grid_keys:
        v = grid_args.get(k, "")
        if isinstance(v, list):
            v_str = "_".join(str(x) for x in v)
        else:
            v_str = str(v)
        # Shorten key names for readability
        short_k = k.replace("dirichlet_alpha", "alpha") \
                   .replace("local_epochs", "K") \
                   .replace("num_rounds", "R") \
                   .replace("noise_ratio", "nr") \
                   .replace("noisy_clients", "nc")
        parts.append(f"{short_k}{v_str}")
    return "_".join(parts)


def run_experiment(args_dict: dict, output_dir: str, dry_run: bool = False) -> dict:
    """Execute a single experiment as a subprocess.

    Constructs a command line from args_dict and calls main.py.
    After completion, reads results.json to extract final metrics.

    Args:
        args_dict: Complete argument dict for this experiment.
        output_dir: Where this experiment's outputs will be written.
        dry_run: If True, print the command but do not execute.

    Returns:
        Dict with keys: 'status', 'returncode', 'metrics', 'stderr'.
    """
    # Build command: python main.py --arg1 val1 --arg2 val2 ...
    cmd = [sys.executable, "main.py"]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            # Skip False booleans (store_true flags)
        elif isinstance(v, list):
            if v:  # skip empty lists
                cmd.extend([f"--{k}"] + [str(x) for x in v])
        else:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend(["--output_dir", output_dir])

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return {"status": "dry_run", "cmd": " ".join(cmd)}

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),  # Run from fed_shapley/ root
    )

    # Extract metrics from results.json if it exists
    exp_name = args_dict.get("exp_name", "exp")
    results_file = Path(output_dir) / exp_name / "results.json"
    metrics = {}
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
                metrics = data.get("final_summary", {})
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "metrics": metrics,
        "stderr": result.stderr[-1000:] if result.returncode != 0 else "",
    }


def main():
    """Grid search entry point."""
    parser = argparse.ArgumentParser(
        description="Run a grid search over FL Shapley experiment configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML grid config file."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing."
    )
    parser.add_argument(
        "--max_runs", type=int, default=None,
        help="Maximum number of experiments to run."
    )
    parser.add_argument(
        "--output_root", default="./outputs/grid",
        help="Root directory for grid search outputs."
    )
    cli = parser.parse_args()

    # Load config
    config_path = Path(cli.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {cli.config}")
        sys.exit(1)

    config = load_config(str(config_path))
    base_args = config.get("base_args", {})
    grid = config.get("grid", {})

    if not grid:
        print("Warning: No grid parameters found in config. Running single experiment.")
        grid = {}

    # Generate all experiment configurations
    experiments = generate_experiments(base_args, grid)
    if cli.max_runs is not None:
        experiments = experiments[:cli.max_runs]

    grid_keys = list(grid.keys())
    print(f"Grid search: {len(grid_keys)} varied parameters, {len(experiments)} total experiments")
    for k, v in grid.items():
        print(f"  {k}: {v}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for i, exp_args in enumerate(experiments):
        run_name = make_run_name(exp_args, grid_keys)
        run_output_dir = str(Path(cli.output_root) / f"run_{i:03d}_{run_name}")

        print(f"\n[{i+1}/{len(experiments)}] {run_name}")
        print(f"  Output: {run_output_dir}")

        result = run_experiment(exp_args, run_output_dir, dry_run=cli.dry_run)

        # Flatten metrics into result
        flat_metrics = {}
        if isinstance(result.get("metrics"), dict):
            flat_metrics = {f"metric_{k}": v for k, v in result["metrics"].items()
                            if isinstance(v, (int, float, str))}

        record = {
            "run_idx": i,
            "run_name": run_name,
            "output_dir": run_output_dir,
            "status": result.get("status", "unknown"),
            "returncode": result.get("returncode", -1),
            **{k: v for k, v in exp_args.items()
               if not isinstance(v, list)},  # scalar args only for CSV
            **flat_metrics,
        }
        if result.get("status") == "failed":
            record["stderr_tail"] = result.get("stderr", "")
            print(f"  FAILED: {result.get('stderr', '')[-200:]}")
        else:
            print(f"  Status: {result.get('status')}")
            for k, v in flat_metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

        all_results.append(record)

    # Save summary CSV
    if not cli.dry_run and all_results:
        summary_path = Path(cli.output_root) / f"grid_results_{timestamp}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all field names from all records
        all_fields = []
        seen = set()
        for record in all_results:
            for k in record.keys():
                if k not in seen:
                    all_fields.append(k)
                    seen.add(k)

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nGrid search complete. Summary saved to: {summary_path}")

        # Print summary statistics
        successful = [r for r in all_results if r.get("status") == "success"]
        print(f"Completed: {len(successful)}/{len(all_results)} experiments successful")

    elif cli.dry_run:
        print(f"\nDry run complete. {len(all_results)} experiments would be run.")


if __name__ == "__main__":
    main()
