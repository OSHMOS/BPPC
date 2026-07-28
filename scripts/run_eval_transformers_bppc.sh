#!/bin/bash
# BPPC Evaluation script for 4 Transformer & SoTA models across left and right batters

MODELS=("vitpose_b" "vitpose_l" "dwpose" "rtmpose")
HANDED_OPTIONS=("left" "right")

cd /home/i2slab0/oshmos/bppc

for handed in "${HANDED_OPTIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "=== Running BPPC evaluation for $model (handed: $handed) ==="
        PYTHONPATH=. python runners/eval/eval_bppc.py --handed "$handed" --model_name "$model"
    done
done

echo "=== All BPPC evaluations completed successfully! ==="
