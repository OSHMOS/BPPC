#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.1
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.1

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.5
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.5

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 5
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 5

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 10
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 10

# for lambda_ohkm in $(seq 0.2 0.2 0.8); do
#   echo "lambda_ohkm: $lambda_ohkm"
#   CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm "$lambda_ohkm"
#   CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm "$lambda_ohkm"
# done

# for lambda_ohkm in $(seq 2 2 8); do
#   echo "lambda_ohkm: $lambda_ohkm"
#   CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm "$lambda_ohkm"
#   CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm "$lambda_ohkm"
# done

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.01
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.01

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.2
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.2

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.5
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.5

# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 5
# CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 5

CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.02
CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.02

CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.05
CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.05

CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed left --lambda_ohkm 0.08
CUDA_VISIBLE_DEVICES=0 python fine-tuning_2ndrebuttal.py --handed right --lambda_ohkm 0.08

echo "작업 완료"