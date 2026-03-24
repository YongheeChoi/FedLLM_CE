#!/usr/bin/env bash
# run_experiments.sh — Run all 3 experiments (39 total runs)
# Usage: cd fed_shapley && bash scripts/run_experiments.sh [--exp 1|2|3] [--dry-run]
#
# Must be run from the fed_shapley/ directory.
# Use --exp to run a single experiment. Use --dry-run to print commands without executing.

set -euo pipefail

SEEDS=(42 123 456)
DRY_RUN=false
RUN_EXP=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp) RUN_EXP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Common flags
COMMON="--dataset cifar10 --num_clients 5 --clients_per_round 5 --local_lr 0.01 --noisy_clients 4 --noise_type label_flip"

run_count=0
fail_count=0

run_cmd() {
    run_count=$((run_count + 1))
    echo ""
    echo "================================================================"
    echo "  Run #${run_count}: $1"
    echo "================================================================"
    shift
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        if ! "$@"; then
            fail_count=$((fail_count + 1))
            echo "[WARN] Run #${run_count} failed, continuing..."
        fi
    fi
}

# =========================================================================
# Experiment 1: Non-FL vs FL Shapley accuracy comparison
# 2 configs (1st order, 2nd order) x 3 seeds = 6 runs
# =========================================================================
run_exp1() {
    echo ""
    echo "******** EXPERIMENT 1: Non-FL vs FL Shapley Accuracy ********"
    for seed in "${SEEDS[@]}"; do
        # Config A: FL 1st order
        run_cmd "Exp1-A seed=${seed}" \
            python main.py $COMMON \
            --num_rounds 50 --local_epochs 5 --partition iid \
            --run_exact_shapley --run_centralized \
            --seed "$seed" --output_dir ../results/exp1

        # Config B: FL 2nd order
        run_cmd "Exp1-B seed=${seed}" \
            python main.py $COMMON \
            --num_rounds 50 --local_epochs 5 --partition iid \
            --run_exact_shapley --run_centralized \
            --use_second_order \
            --seed "$seed" --output_dir ../results/exp1
    done
}

# =========================================================================
# Experiment 2: Communication cost vs Shapley accuracy trade-off
# 5 configs (rounds x epochs pairs, budget=250) x 3 seeds = 15 runs
# =========================================================================
run_exp2() {
    echo ""
    echo "******** EXPERIMENT 2: Communication vs Accuracy Trade-off ********"

    # (rounds, epochs) pairs with constant budget = 250
    ROUNDS=(10 25 50 125 250)
    EPOCHS=(25 10  5   2   1)

    for seed in "${SEEDS[@]}"; do
        for i in "${!ROUNDS[@]}"; do
            r="${ROUNDS[$i]}"
            e="${EPOCHS[$i]}"
            run_cmd "Exp2-C$((i+1)) r=${r} e=${e} seed=${seed}" \
                python main.py $COMMON \
                --num_rounds "$r" --local_epochs "$e" --partition iid \
                --run_exact_shapley \
                --seed "$seed" --output_dir ../results/exp2
        done
    done
}

# =========================================================================
# Experiment 3: Non-IID level vs Shapley accuracy
# 6 configs (IID + 5 Dirichlet alphas) x 3 seeds = 18 runs
# =========================================================================
run_exp3() {
    echo ""
    echo "******** EXPERIMENT 3: Non-IID Level vs Accuracy ********"

    ALPHAS=(10.0 1.0 0.5 0.1 0.01)

    for seed in "${SEEDS[@]}"; do
        # D1: IID
        run_cmd "Exp3-D1 iid seed=${seed}" \
            python main.py $COMMON \
            --num_rounds 50 --local_epochs 5 --partition iid \
            --run_exact_shapley \
            --seed "$seed" --output_dir ../results/exp3

        # D2-D6: Dirichlet with varying alpha
        idx=2
        for alpha in "${ALPHAS[@]}"; do
            run_cmd "Exp3-D${idx} alpha=${alpha} seed=${seed}" \
                python main.py $COMMON \
                --num_rounds 50 --local_epochs 5 \
                --partition dirichlet --dirichlet_alpha "$alpha" \
                --run_exact_shapley \
                --seed "$seed" --output_dir ../results/exp3
            idx=$((idx + 1))
        done
    done
}

# =========================================================================
# Main
# =========================================================================
start_time=$(date +%s)
echo "Starting experiments at $(date)"

if [[ -z "$RUN_EXP" || "$RUN_EXP" == "1" ]]; then run_exp1; fi
if [[ -z "$RUN_EXP" || "$RUN_EXP" == "2" ]]; then run_exp2; fi
if [[ -z "$RUN_EXP" || "$RUN_EXP" == "3" ]]; then run_exp3; fi

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo ""
echo "================================================================"
echo "  All done! ${run_count} runs completed (${fail_count} failures)"
echo "  Elapsed: $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
echo "================================================================"
