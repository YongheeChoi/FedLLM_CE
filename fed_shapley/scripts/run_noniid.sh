#!/bin/bash
# run_noniid.sh — Non-IID sweep over Dirichlet alpha values.
#
# Sweeps alpha in [0.1, 0.5, 1.0, 5.0, 100.0].
# Lower alpha = more non-IID label distribution across clients.
# alpha=100 approximates IID.
#
# Usage: bash scripts/run_noniid.sh

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for ALPHA in 0.1 0.5 1.0 5.0 100.0; do
    echo "============================================"
    echo "Running: alpha = ${ALPHA}"
    echo "============================================"
    python main.py \
        --exp_name "noniid_alpha${ALPHA}" \
        --num_clients 10 \
        --partition dirichlet \
        --dirichlet_alpha "$ALPHA" \
        --num_rounds 100 \
        --local_epochs 5 \
        --local_lr 0.01 \
        --local_batch_size 64 \
        --run_exact_shapley \
        --use_second_order \
        --dataset cifar10 \
        --seed 42 \
        --eval_every 10 \
        --log_every 10 \
        --output_dir ./outputs
    echo "Completed alpha=${ALPHA}"
done

echo "All non-IID experiments done!"
