# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FedLLM_CE is a research repository containing two main components:

### 1. GhostSuite (Original Framework)
A PyTorch framework for efficient per-sample gradient information computation for data-centric ML research. Based on the ICLR'25 Outstanding Paper Runner-up ["Data Shapley in One Training Run"](https://openreview.net/pdf?id=HD6bWcj87Y).

### 2. FedShapley (`fed_shapley/`)
A Federated Learning framework that applies the In-Run Data Shapley method to attribute per-client contributions in FL settings. Each FL client is treated as a "player" whose Shapley value measures their contribution to reducing global validation loss, computed via gradient dot-products during training.

**Core formula:**
```
ϕ_c^(t) = -η · <∇ℓ(w_t, D_val), Δw_c^(t)>
```

With optional second-order Hessian correction:
```
ϕ_c^(t) += (η²/2) · Δw_c^(t)ᵀ H(D_val) Σ_{c'} Δw_{c'}^(t)
```

## Installation

```bash
# GhostSuite
cd GhostSuite
pip install -r requirements.txt

# FedShapley
cd fed_shapley
pip install -r requirements.txt
```

## Running FedShapley

```bash
cd fed_shapley

# Basic IID experiment
python main.py --num_clients 10 --partition iid --num_rounds 100 --dataset cifar10

# Non-IID with Dirichlet
python main.py --partition dirichlet --dirichlet_alpha 0.1 --num_rounds 100

# Noisy client detection with exact Shapley ground-truth
python main.py --num_clients 5 --noisy_clients 4 --noise_type label_flip --run_exact_shapley

# With centralized baseline comparison
python main.py --num_clients 5 --run_exact_shapley --run_centralized

# Run pre-configured experiments (39 runs, 3 experiments)
bash scripts/run_experiments.sh              # all experiments
bash scripts/run_experiments.sh --exp 1      # exp1 only
bash scripts/run_experiments.sh --dry-run    # preview commands

# Analyze experiment results
python scripts/analyze_results.py --results_root ../results

# Grid search
python scripts/run_grid_search.py --config scripts/grid_configs/alpha_sweep.yaml
```

## FedShapley Architecture

### Entry Point
`fed_shapley/main.py` — Orchestrates: argument parsing, dataset loading, federated partitioning, FL training with in-run Shapley, optional exact/MC Shapley baseline, evaluation, and visualization.

### Core Modules

**`fl/`** — Federated Learning primitives
- `server.py`: Global model, FedAvg aggregation, validation gradient (∇ℓ) and Hessian-vector product (Hv) computation
- `client.py`: Local SGD training, Δw_c computation, label-flip/random-update noise injection
- `trainer.py`: Main FL training loop coordinating server, clients, and Shapley computation

**`shapley/`** — Three Shapley computation engines
- `in_run_shapley.py`: **Primary method** — O(1) per-round via gradient dot-products, supports 1st and 2nd order terms
- `exact_shapley.py`: Ground-truth via 2^n subset enumeration (n ≤ 10 only)
- `mc_shapley.py`: Monte Carlo permutation sampling for n > 10

**`data/`** — Dataset loading and federated partitioning
- `datasets.py`: CIFAR-10/100, Tiny ImageNet with standard augmentation
- `partition.py`: IID and Dirichlet non-IID partitioning with optional quantity skew

**`models/`** — Model factory
- `resnet.py`: Dataset-adapted ResNet-18 (modified conv1/maxpool for 32x32/64x64 inputs)

**`eval/`** — Evaluation metrics
- `fidelity.py`: RMSE, Pearson, Spearman correlation vs. ground truth
- `client_removal.py`: Remove high/low/random Shapley clients and retrain
- `noisy_client.py`: AUROC for noisy client detection

**`centralized/`** — Centralized training baseline
- `centralized_trainer.py`: Per-sample gradient attribution mapped back to clients

**`utils/`** — Logging, visualization, seed management, cost tracking
- `logger.py`: W&B, TensorBoard, and disk logging with auto-generated experiment tags; includes per-round timing CSV export
- `visualize.py`: Bar charts, scatter plots, heatmaps
- `seed.py`: Global random seed setting
- `timer.py`: `CostTracker` class — wall-clock timing per phase + FLOPs estimation via PyTorch `FlopCounterMode` (fallback: 2×params)

**`scripts/`** — Experiment runners and analysis
- `run_experiments.sh`: **Main experiment suite** — runs 3 experiments (39 total runs) with `--exp 1|2|3` and `--dry-run` options
- `analyze_results.py`: Aggregates `*_results.json` from `results/exp1,exp2,exp3` into summary tables and CSV exports
- `run_grid_search.py`: YAML-config-driven grid search with CSV summary output
- Shell scripts: `run_basic.sh`, `run_noniid.sh`, `run_k_sweep.sh`
- `grid_configs/`: Pre-configured sweeps for alpha, local epochs, and noise

### Key Design Patterns
- All Shapley calculators share the same API: `compute_round_shapley()`, `accumulate()`, `get_cumulative()`
- Client updates are flattened to 1D tensors (trainable params only, excluding BatchNorm running stats) for dot-product computation
- Experiment results are saved with auto-generated tags encoding key hyperparameters (e.g., `cifar10_c5_k5_iid_r50_e5_lr0.01_noisy4_s42`)
- `CostTracker` wraps each FL phase (local_training, shapley_computation, aggregation, evaluation) for wall-clock timing and FLOPs estimation
- Per-round timing is exported as `{tag}_timing_rounds.csv`; cost summary is included in `{tag}_results.json`

### Experiment Results (`results/`)
Pre-run experiment results are stored at the project root under `results/`:
- `results/exp1/` — Exp1: Non-FL vs FL Shapley accuracy (1st/2nd order + centralized baseline, 6 runs)
- `results/exp2/` — Exp2: Communication cost vs accuracy trade-off (budget=250, 5 round×epoch pairs, 15 runs)
- `results/exp3/` — Exp3: Non-IID level vs Shapley accuracy (IID + 5 Dirichlet alphas, 18 runs)
- `results/analysis/` — Aggregated CSV summaries (`exp1_summary.csv`, `exp2_summary.csv`, `exp3_summary.csv`)

Run analysis: `cd fed_shapley && python scripts/analyze_results.py --results_root ../results`

## GhostSuite Architecture

### Two Core Engines

**`GradDotProdEngine`** (`ghostEngines/graddotprod_engine.py`)
- Online, per-step gradient similarity via single forward/backward pass
- Key implementation: `autograd_grad_sample_dotprod.py` uses `_NamedSavedTensorManager` for tensor capture

**`GradProjLoRAEngine`** (`ghostEngines/gradProjection/gradproj_engine.py`)
- Offline, corpus-scale gradient analysis via Kronecker-structured random projection
- LoRA-style side branch preserves inner products up to J-L distortion

### Integration Interface
`ghostEngines/engine_manager.py` (`GhostEngineManager`) auto-selects engine based on config.

### Layer Support
`supported_layers_grad_samplers_dotprod.py`: Linear, Conv2D, Embedding, LayerNorm, attention layers.

### TorchTitan Integration
`examples/torchtitan/` for large-scale LLM pretraining with GhostSuite.
