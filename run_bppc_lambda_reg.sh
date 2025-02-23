#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg 0.005
# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg 0.005

# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg 50
# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg 50

# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg 0.05
# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg 0.05

# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg 0.01
# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg 0.01

# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg 1
# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg 1

# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg 5
# CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg 5

for lambda_reg in $(seq 0.1 0.1 0.4); do
  echo "lambda_reg: $lambda_reg"
  CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg "$lambda_reg"
  CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg "$lambda_reg"
done

for lambda_reg in $(seq 10 10 40); do
  echo "lambda_reg: $lambda_reg"
  CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed left --lambda_reg "$lambda_reg"
  CUDA_VISIBLE_DEVICES=1 python fine-tuning_2ndrebuttal.py --handed right --lambda_reg "$lambda_reg"
done

echo "작업 완료"