"""
analyze_results.py — Aggregate and summarize all experiment results into tables.

Reads *_results.json files from results/exp1, exp2, exp3 and produces:
  1. Per-experiment summary tables (mean +/- std across seeds)
  2. Combined CSV exports for further plotting
  3. Console-friendly formatted tables

Usage:
    python scripts/analyze_results.py [--results_root ../results]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_tag(tag: str) -> Dict[str, Any]:
    """Extract config fields from experiment tag string.

    Examples:
        cifar10_c5_k5_iid_r50_e5_lr0.01_noisy4_s42
        cifar10_c5_k5_dir0.1_r50_e5_lr0.01_noisy4_s42
        cifar10_c5_k5_iid_r50_e5_lr0.01_2nd_noisy4_s42
    """
    info: Dict[str, Any] = {}

    # seed
    m = re.search(r"_s(\d+)$", tag)
    info["seed"] = int(m.group(1)) if m else None

    # rounds, epochs
    m = re.search(r"_r(\d+)", tag)
    info["num_rounds"] = int(m.group(1)) if m else None
    m = re.search(r"_e(\d+)", tag)
    info["local_epochs"] = int(m.group(1)) if m else None

    # partition
    if "_iid_" in tag:
        info["partition"] = "iid"
        info["dirichlet_alpha"] = None
    else:
        m = re.search(r"_dir(?:_a)?([\d.]+)_", tag)
        if m:
            info["partition"] = "dirichlet"
            info["dirichlet_alpha"] = float(m.group(1))
        else:
            info["partition"] = "unknown"
            info["dirichlet_alpha"] = None

    # second order
    info["use_second_order"] = "_2nd_" in tag

    # budget
    if info["num_rounds"] and info["local_epochs"]:
        info["budget"] = info["num_rounds"] * info["local_epochs"]

    return info


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all *_results.json from a directory."""
    records = []
    p = Path(results_dir)
    if not p.exists():
        return records
    for f in sorted(p.glob("*_results.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            tag = data.get("experiment_tag", f.stem.replace("_results", ""))
            parsed = parse_tag(tag)
            record = {"file": str(f), "tag": tag, **parsed}

            # Extract key metrics from final_summary
            summary = data.get("final_summary", data.get("summary", {}))
            config = data.get("config", {})

            # Fidelity: in-run vs exact
            record["spearman_rho"] = _get(summary, "fidelity_spearman_r", "fidelity/spearman_r")
            record["pearson_r"] = _get(summary, "fidelity_pearson_r", "fidelity/pearson_r")
            record["rmse"] = _get(summary, "fidelity_rmse", "fidelity/rmse")

            # Fidelity: centralized vs exact
            record["cent_gt_spearman"] = _get(summary, "fidelity_cent_gt/spearman_r")
            record["cent_gt_pearson"] = _get(summary, "fidelity_cent_gt/pearson_r")

            # Fidelity: in-run vs centralized
            record["inrun_cent_spearman"] = _get(summary, "fidelity_central/spearman_r")

            # Detection
            record["auroc"] = _get(summary, "detection/auroc")
            record["precision_at_k"] = _get(summary, "detection/precision_at_k")

            # Accuracy
            record["test_acc"] = _get(summary, "final_test_acc")
            record["val_acc"] = _get(summary, "final_val_acc")

            # Timing
            record["fl_total_sec"] = _get(summary, "time/fl_training_total_total_sec")
            record["shapley_sec"] = _get(summary, "time/shapley_computation_total_sec")
            record["local_train_sec"] = _get(summary, "time/local_training_total_sec")
            record["gt_shapley_sec"] = _get(summary, "time/gt_shapley_total_sec")
            record["centralized_sec"] = _get(summary, "time/centralized_training_total_sec")

            # Shapley overhead
            if record["shapley_sec"] and record["local_train_sec"]:
                record["shapley_overhead_pct"] = (
                    record["shapley_sec"] / max(record["local_train_sec"], 1e-6) * 100
                )
            else:
                record["shapley_overhead_pct"] = None

            # Config extras
            record["run_centralized"] = config.get("run_centralized", False)
            record["run_exact_shapley"] = config.get("run_exact_shapley", False)

            records.append(record)
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
    return records


def _get(d: dict, *keys) -> Optional[float]:
    """Try multiple keys, return first non-None value."""
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None


def agg_seeds(records: List[Dict], group_keys: List[str], metric_keys: List[str]
              ) -> List[Dict[str, Any]]:
    """Group records by group_keys and compute mean/std for metric_keys."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)

    rows = []
    for key, group in sorted(groups.items()):
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_seeds"] = len(group)
        for mk in metric_keys:
            vals = [r[mk] for r in group if r.get(mk) is not None and not _is_nan(r.get(mk))]
            if vals:
                row[f"{mk}_mean"] = np.mean(vals)
                row[f"{mk}_std"] = np.std(vals) if len(vals) > 1 else 0.0
                row[f"{mk}_vals"] = vals
            else:
                row[f"{mk}_mean"] = None
                row[f"{mk}_std"] = None
                row[f"{mk}_vals"] = []
        rows.append(row)
    return rows


def _is_nan(v) -> bool:
    if v is None:
        return True
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False


def fmt(v, precision=4) -> str:
    """Format a value for table display."""
    if v is None:
        return "-"
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        return f"{v:.{precision}f}"
    return str(v)


def fmt_mean_std(mean, std, precision=4) -> str:
    if mean is None:
        return "-"
    if std is not None and std > 0:
        return f"{mean:.{precision}f} +/- {std:.{precision}f}"
    return f"{mean:.{precision}f}"


def print_table(headers: List[str], rows: List[List[str]], title: str = ""):
    """Print a formatted ASCII table."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))
    print()


def save_csv(path: str, headers: List[str], rows: List[List[str]]):
    """Write a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"  -> Saved: {path}")


# ---------------------------------------------------------------------------
# Experiment-specific analysis
# ---------------------------------------------------------------------------

def analyze_exp1(records: List[Dict], output_dir: str):
    """Exp1: Non-FL vs FL Shapley accuracy comparison."""
    if not records:
        print("\n[Exp1] No results found.")
        return

    metrics = ["spearman_rho", "pearson_r", "rmse", "auroc", "test_acc",
               "cent_gt_spearman", "cent_gt_pearson", "inrun_cent_spearman",
               "fl_total_sec", "shapley_overhead_pct", "centralized_sec"]

    rows_agg = agg_seeds(records, ["use_second_order"], metrics)

    # Table 1: FL In-Run vs Exact
    headers = ["Method", "Spearman rho", "Pearson r", "RMSE", "AUROC", "Test Acc"]
    table_rows = []
    for row in rows_agg:
        method = "2nd order" if row["use_second_order"] else "1st order"
        table_rows.append([
            method,
            fmt_mean_std(row["spearman_rho_mean"], row["spearman_rho_std"]),
            fmt_mean_std(row["pearson_r_mean"], row["pearson_r_std"]),
            fmt_mean_std(row["rmse_mean"], row["rmse_std"]),
            fmt_mean_std(row["auroc_mean"], row["auroc_std"]),
            fmt_mean_std(row["test_acc_mean"], row["test_acc_std"]),
        ])

    # Add centralized row if available
    for row in rows_agg:
        if row["cent_gt_spearman_mean"] is not None:
            table_rows.append([
                "Centralized",
                fmt_mean_std(row["cent_gt_spearman_mean"], row["cent_gt_spearman_std"]),
                fmt_mean_std(row["cent_gt_pearson_mean"], row["cent_gt_pearson_std"]),
                "-", "-", "-",
            ])
            break

    print_table(headers, table_rows, "Exp1: Method Comparison (vs Exact Shapley)")

    # Table 2: Timing comparison
    headers2 = ["Method", "FL Total (s)", "Shapley Overhead (%)", "Centralized (s)"]
    time_rows = []
    for row in rows_agg:
        method = "2nd order" if row["use_second_order"] else "1st order"
        time_rows.append([
            method,
            fmt_mean_std(row["fl_total_sec_mean"], row["fl_total_sec_std"], 1),
            fmt_mean_std(row["shapley_overhead_pct_mean"], row["shapley_overhead_pct_std"], 2),
            fmt_mean_std(row["centralized_sec_mean"], row["centralized_sec_std"], 1),
        ])
    print_table(headers2, time_rows, "Exp1: Timing Comparison")

    # CSV export
    csv_headers = ["method", "seed", "spearman_rho", "pearson_r", "rmse", "auroc",
                   "test_acc", "fl_total_sec", "shapley_overhead_pct"]
    csv_rows = []
    for r in records:
        method = "2nd_order" if r["use_second_order"] else "1st_order"
        csv_rows.append([method, r["seed"], fmt(r["spearman_rho"]), fmt(r["pearson_r"]),
                         fmt(r["rmse"]), fmt(r["auroc"]), fmt(r["test_acc"]),
                         fmt(r["fl_total_sec"], 1), fmt(r["shapley_overhead_pct"], 2)])
    save_csv(os.path.join(output_dir, "exp1_summary.csv"), csv_headers, csv_rows)


def analyze_exp2(records: List[Dict], output_dir: str):
    """Exp2: Communication cost vs Shapley accuracy trade-off."""
    if not records:
        print("\n[Exp2] No results found.")
        return

    metrics = ["spearman_rho", "pearson_r", "rmse", "auroc", "test_acc",
               "fl_total_sec", "shapley_sec", "shapley_overhead_pct"]

    rows_agg = agg_seeds(records, ["num_rounds", "local_epochs"], metrics)

    # Sort by num_rounds
    rows_agg.sort(key=lambda r: r["num_rounds"] or 0)

    headers = ["Rounds", "Epochs", "Budget", "Spearman rho", "Pearson r",
               "AUROC", "Test Acc", "FL Time (s)", "Shapley OH (%)"]
    table_rows = []
    for row in rows_agg:
        budget = (row["num_rounds"] or 0) * (row["local_epochs"] or 0)
        table_rows.append([
            row["num_rounds"],
            row["local_epochs"],
            budget,
            fmt_mean_std(row["spearman_rho_mean"], row["spearman_rho_std"]),
            fmt_mean_std(row["pearson_r_mean"], row["pearson_r_std"]),
            fmt_mean_std(row["auroc_mean"], row["auroc_std"]),
            fmt_mean_std(row["test_acc_mean"], row["test_acc_std"]),
            fmt_mean_std(row["fl_total_sec_mean"], row["fl_total_sec_std"], 1),
            fmt_mean_std(row["shapley_overhead_pct_mean"], row["shapley_overhead_pct_std"], 2),
        ])

    print_table(headers, table_rows,
                "Exp2: Communication Cost vs Shapley Accuracy (budget=250)")

    # Per-seed detail table
    headers2 = ["Rounds", "Epochs", "Seed", "Spearman rho", "Pearson r", "Test Acc"]
    detail_rows = []
    for r in sorted(records, key=lambda x: (x["num_rounds"] or 0, x["seed"] or 0)):
        detail_rows.append([
            r["num_rounds"], r["local_epochs"], r["seed"],
            fmt(r["spearman_rho"]), fmt(r["pearson_r"]), fmt(r["test_acc"]),
        ])
    print_table(headers2, detail_rows, "Exp2: Per-Seed Detail")

    # CSV export
    csv_headers = ["num_rounds", "local_epochs", "budget", "seed",
                   "spearman_rho", "pearson_r", "rmse", "auroc", "test_acc",
                   "fl_total_sec", "shapley_sec", "shapley_overhead_pct"]
    csv_rows = []
    for r in sorted(records, key=lambda x: (x["num_rounds"] or 0, x["seed"] or 0)):
        budget = (r["num_rounds"] or 0) * (r["local_epochs"] or 0)
        csv_rows.append([r["num_rounds"], r["local_epochs"], budget, r["seed"],
                         fmt(r["spearman_rho"]), fmt(r["pearson_r"]),
                         fmt(r["rmse"]), fmt(r["auroc"]), fmt(r["test_acc"]),
                         fmt(r["fl_total_sec"], 1), fmt(r["shapley_sec"], 2),
                         fmt(r["shapley_overhead_pct"], 2)])
    save_csv(os.path.join(output_dir, "exp2_summary.csv"), csv_headers, csv_rows)


def analyze_exp3(records: List[Dict], output_dir: str):
    """Exp3: Non-IID level vs Shapley accuracy."""
    if not records:
        print("\n[Exp3] No results found.")
        return

    metrics = ["spearman_rho", "pearson_r", "rmse", "auroc", "test_acc",
               "fl_total_sec", "shapley_overhead_pct"]

    # Group by (partition, alpha) — for iid, alpha is None
    def partition_key(r):
        if r["partition"] == "iid":
            return ("iid", None)
        return ("dirichlet", r["dirichlet_alpha"])

    # Manual grouping since alpha can be None
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in records:
        groups[partition_key(r)].append(r)

    # Sort: IID first, then by decreasing alpha (less non-IID first)
    sorted_keys = sorted(groups.keys(),
                         key=lambda k: (0 if k[0] == "iid" else 1, -(k[1] or 999)))

    headers = ["Partition", "Alpha", "Non-IID Level", "Spearman rho", "Pearson r",
               "AUROC", "Test Acc"]
    table_rows = []

    noniid_labels = {
        None: "baseline (IID)",
        10.0: "weak",
        1.0: "moderate",
        0.5: "strong",
        0.1: "very strong",
        0.01: "extreme",
    }

    for key in sorted_keys:
        group = groups[key]
        partition, alpha = key
        alpha_str = fmt(alpha, 2) if alpha is not None else "-"
        label = noniid_labels.get(alpha, "")

        vals = {}
        for mk in metrics:
            mv = [r[mk] for r in group if r.get(mk) is not None and not _is_nan(r.get(mk))]
            vals[f"{mk}_mean"] = np.mean(mv) if mv else None
            vals[f"{mk}_std"] = np.std(mv) if len(mv) > 1 else 0.0

        table_rows.append([
            partition, alpha_str, label,
            fmt_mean_std(vals["spearman_rho_mean"], vals["spearman_rho_std"]),
            fmt_mean_std(vals["pearson_r_mean"], vals["pearson_r_std"]),
            fmt_mean_std(vals["auroc_mean"], vals["auroc_std"]),
            fmt_mean_std(vals["test_acc_mean"], vals["test_acc_std"]),
        ])

    print_table(headers, table_rows,
                "Exp3: Non-IID Level vs Shapley Accuracy (r=50, e=5)")

    # Per-seed detail
    headers2 = ["Partition", "Alpha", "Seed", "Spearman rho", "Pearson r", "AUROC", "Test Acc"]
    detail_rows = []
    for key in sorted_keys:
        for r in sorted(groups[key], key=lambda x: x["seed"] or 0):
            detail_rows.append([
                r["partition"],
                fmt(r["dirichlet_alpha"], 2) if r["dirichlet_alpha"] is not None else "-",
                r["seed"],
                fmt(r["spearman_rho"]), fmt(r["pearson_r"]),
                fmt(r["auroc"]), fmt(r["test_acc"]),
            ])
    print_table(headers2, detail_rows, "Exp3: Per-Seed Detail")

    # CSV export
    csv_headers = ["partition", "dirichlet_alpha", "seed",
                   "spearman_rho", "pearson_r", "rmse", "auroc", "test_acc",
                   "fl_total_sec", "shapley_overhead_pct"]
    csv_rows = []
    for key in sorted_keys:
        for r in sorted(groups[key], key=lambda x: x["seed"] or 0):
            csv_rows.append([
                r["partition"],
                fmt(r["dirichlet_alpha"], 2) if r["dirichlet_alpha"] is not None else "",
                r["seed"],
                fmt(r["spearman_rho"]), fmt(r["pearson_r"]),
                fmt(r["rmse"]), fmt(r["auroc"]), fmt(r["test_acc"]),
                fmt(r["fl_total_sec"], 1), fmt(r["shapley_overhead_pct"], 2),
            ])
    save_csv(os.path.join(output_dir, "exp3_summary.csv"), csv_headers, csv_rows)


def print_overall_summary(all_records: Dict[str, List[Dict]]):
    """Print a high-level overview across all experiments."""
    print(f"\n{'#'*80}")
    print(f"  OVERALL EXPERIMENT SUMMARY")
    print(f"{'#'*80}")

    total_runs = sum(len(v) for v in all_records.values())
    print(f"\n  Total runs loaded: {total_runs}")
    for exp_name, recs in all_records.items():
        n_success = len([r for r in recs if r.get("spearman_rho") is not None])
        n_fail = len(recs) - n_success
        print(f"    {exp_name}: {len(recs)} runs ({n_success} valid, {n_fail} with missing fidelity)")

    # Best configs per experiment
    print(f"\n  Best Spearman rho per experiment (averaged across seeds):")
    for exp_name, recs in all_records.items():
        if not recs:
            continue
        valid = [r for r in recs if r.get("spearman_rho") is not None
                 and not _is_nan(r.get("spearman_rho"))]
        if not valid:
            print(f"    {exp_name}: no valid results")
            continue
        # Group and average
        from itertools import groupby
        key_fn = lambda r: r["tag"].rsplit("_s", 1)[0]  # tag without seed
        groups = defaultdict(list)
        for r in valid:
            groups[key_fn(r)].append(r["spearman_rho"])
        best_key = max(groups, key=lambda k: np.mean(groups[k]))
        best_mean = np.mean(groups[best_key])
        best_std = np.std(groups[best_key]) if len(groups[best_key]) > 1 else 0
        print(f"    {exp_name}: rho={best_mean:.4f} +/- {best_std:.4f}  [{best_key}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze FedShapley experiment results")
    parser.add_argument("--results_root", default="../results",
                        help="Root directory containing exp1/, exp2/, exp3/")
    args = parser.parse_args()

    root = Path(args.results_root)
    output_dir = str(root / "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {root.resolve()}")

    all_records = {}
    for exp in ["exp1", "exp2", "exp3"]:
        exp_dir = root / exp
        records = load_results(str(exp_dir))
        all_records[exp] = records
        print(f"  {exp}: {len(records)} result files loaded")

    # Analyze each experiment
    if all_records["exp1"]:
        analyze_exp1(all_records["exp1"], output_dir)
    if all_records["exp2"]:
        analyze_exp2(all_records["exp2"], output_dir)
    if all_records["exp3"]:
        analyze_exp3(all_records["exp3"], output_dir)

    # Overall summary
    print_overall_summary(all_records)

    print(f"\n  Analysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
