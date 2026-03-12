#!/bin/bash
# 전체 테스트 케이스 Evaluation
# Usage: bash scripts/run_eval_bppc.sh
PYTHONPATH=. python runners/eval/eval_bppc.py --handed left
PYTHONPATH=. python runners/eval/eval_bppc.py --handed right
