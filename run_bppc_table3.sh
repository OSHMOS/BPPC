#!/bin/bash
### all baseline
CUDA_VISIBLE_DEVICES=1 python fine-tuning_rebuttal.py --handed right # for left_final
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_rebuttal.py --handed left # for left_final