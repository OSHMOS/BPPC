# CUDA_VISIBLE_DEVICES=3 python fine-tuning.py --cfg configs/config_mlb_simplebaseline_2D_h.yaml --dataset_name mlb --estimator simplebaseline --body_representation 2D
# CUDA_VISIBLE_DEVICES=7 python fine-tuning.py --cfg configs/config_mlb_simplebaseline_2D_h.yaml --dataset_name mlb --estimator simplebaseline --body_representation 2D

# CUDA_VISIBLE_DEVICES=5 python fine-tuning_res50.py # 한 번에 돌리는 용도
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_res101.py
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_res152.py
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_hw32.py
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_hw48.py
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_darkw32.py
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_darkw48.py

### all baseline
CUDA_VISIBLE_DEVICES=1 python fine-tuning_allbaseline_right.py

### test
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_hw48_t.py # conf.score 하는 중 randomness
# CUDA_VISIBLE_DEVICES=5 python fine-tuning_res50_t.py # 하이퍼파라미터 찾는 용도

