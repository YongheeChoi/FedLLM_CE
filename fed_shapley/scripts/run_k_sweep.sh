#!/bin/bash
# run_k_sweep.sh — Local epoch sweep with fixed total computation budget.
#
# Fixes total local epochs = 500 (= local_epochs × num_rounds).
# This controls for total computation while varying rounds vs. epochs tradeoff.
#
# K=1 → 500 rounds (frequent communication, many gradient steps)
# K=50 → 10 rounds (rare communication, large local drift)
#
# Usage: bash scripts/run_k_sweep.sh

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TOTAL_EPOCHS=500

for K in 1 2 5 10 20 50; do
    ROUNDS=$((TOTAL_EPOCHS / K))
    echo "============================================"
    echo "Running: K=${K} epochs/round, ${ROUNDS} rounds"
    echo "============================================"
    python main.py \
        --exp_name "k_sweep_k${K}" \
        --num_clients 10 \
        --local_epochs "$K" \
        --num_rounds "$ROUNDS" \
        --local_lr 0.01 \
        --local_batch_size 64 \
        --partition iid \
        --run_exact_shapley \
        --dataset cifar10 \
        --seed 42 \
        --eval_every $((ROUNDS / 10 > 0 ? ROUNDS / 10 : 1)) \
        --log_every $((ROUNDS / 10 > 0 ? ROUNDS / 10 : 1)) \
        --output_dir ./outputs
    echo "Completed K=${K}"
done

echo "All K-sweep experiments done!"
