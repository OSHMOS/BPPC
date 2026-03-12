#!/bin/bash
# 전체 파이프라인 (Opt -> Eval, left + right 전부)
# Usage: bash scripts/run_pipeline.sh

echo "=== Running Optimization ==="
bash scripts/run_opt_bppc.sh

echo "=== Running Evaluation ==="
bash scripts/run_eval_bppc.sh

echo "All done! Collecting results to Excel..."
PYTHONPATH=. python aggregate_results.py
