#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed left --lambda_accel 0.001
# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed right --lambda_accel 0.001

# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed left --lambda_accel 0.1
# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed right --lambda_accel 0.1

# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed left --lambda_accel 1
# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed right --lambda_accel 1

# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed left --lambda_accel 10
# CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed right --lambda_accel 10

# for lambda_accel in $(seq 2 8); do
#   echo "lambda_accel: $lambda_accel"
#   CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed left --lambda_accel "$lambda_accel"
#   CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed right --lambda_accel "$lambda_accel"
# done

CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed left --lambda_accel 9
CUDA_VISIBLE_DEVICES=3 python fine-tuning_2ndrebuttal.py --handed right --lambda_accel 9

echo "작업 완료"