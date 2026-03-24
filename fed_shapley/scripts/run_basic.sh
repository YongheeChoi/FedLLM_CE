#!/bin/bash
# run_basic.sh — Basic experiment: 10 clients, IID, CIFAR-10, 100 rounds
#
# Usage: bash scripts/run_basic.sh
# Expected runtime: ~30 min on GPU with 10 clients

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py \
    --exp_name "basic_iid_c10" \
    --num_clients 10 \
    --partition iid \
    --num_rounds 100 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --local_batch_size 64 \
    --run_exact_shapley \
    --use_second_order \
    --dataset cifar10 \
    --seed 42 \
    --eval_every 5 \
    --log_every 5 \
    --output_dir ./outputs

echo "Done! Results in ./outputs/basic_iid_c10/"
