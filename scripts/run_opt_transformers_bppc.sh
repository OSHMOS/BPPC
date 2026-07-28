#!/bin/bash
# Parallel BPPC Optimization for Transformer & SoTA backbones across left and right batters
set -e

MODELS=("vitpose_b" "vitpose_l" "dwpose" "rtmpose")
HANDED_OPTIONS=("left" "right")

cd /home/i2slab0/oshmos/bppc

for handed in "${HANDED_OPTIONS[@]}"; do
    echo "=========================================================================="
    echo "=== Launching PARALLEL BPPC Optimization for 4 models on handed: $handed ==="
    echo "=========================================================================="
    for model in "${MODELS[@]}"; do
        echo "Launching BPPC optimization process: $model ($handed)"
        PYTHONPATH=. python runners/opt/opt_bppc.py --handed "$handed" --model_name "$model" > "opt_${model}_${handed}.log" 2>&1 &
    done
    wait
    echo "=== Completed PARALLEL BPPC Optimization for handed: $handed ==="
done
