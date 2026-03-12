#!/bin/bash
# 단일 테스트 케이스 Optimization + Evaluation
# Usage: bash scripts/run_one_bppc.sh <handed> <folder_number> [model_name]
#   예시: bash scripts/run_one_bppc.sh left 0170 res152

HANDED=${1:-"left"}
FOLDER=${2:-"0170"}
MODEL=${3:-"all"}

echo "=== [1/2] Optimizing: handed=${HANDED}, folder=${FOLDER}, model=${MODEL} ==="
PYTHONPATH=. python runners/opt/opt_bppc.py \
    --handed ${HANDED} \
    --model_name ${MODEL} \
    --folder_number ${FOLDER}

echo "=== [2/2] Evaluating: handed=${HANDED}, folder=${FOLDER}, model=${MODEL} ==="
PYTHONPATH=. python runners/eval/eval_bppc.py \
    --handed ${HANDED} \
    --model_name ${MODEL} \
    --folder_number ${FOLDER}

echo "Done! Results saved in demo/bppc/${HANDED}/${MODEL}/${FOLDER}_results.npz"
echo "Visualization saved in demo/output/backbone/${MODEL}/${HANDED}/${FOLDER}/"
