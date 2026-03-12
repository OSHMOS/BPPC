#!/bin/bash
# 전체 테스트 케이스 Optimization
# Usage: bash scripts/run_opt_bppc.sh
PYTHONPATH=. python runners/opt/opt_bppc.py --handed left
PYTHONPATH=. python runners/opt/opt_bppc.py --handed right
