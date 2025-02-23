#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed left --lambda_vel 0.5
# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed right --lambda_vel 0.5

# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed left --lambda_vel 1
# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed right --lambda_vel 1

# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed left --lambda_vel 10
# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed right --lambda_vel 10

# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed left --lambda_vel 50
# CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed right --lambda_vel 50

for lambda_vel in $(seq 2 4); do
  echo "lambda_vel: $lambda_vel"
  CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed left --lambda_vel "$lambda_vel"
  CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed right --lambda_vel "$lambda_vel"
done

for lambda_vel in $(seq 20 10 40); do
  echo "lambda_vel: $lambda_vel"
  CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed left --lambda_vel "$lambda_vel"
  CUDA_VISIBLE_DEVICES=2 python fine-tuning_2ndrebuttal.py --handed right --lambda_vel "$lambda_vel"
done

echo "작업 완료"