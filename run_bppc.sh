#!/bin/bash
### all baseline
CUDA_VISIBLE_DEVICES=0 python fine-tuning_allbaseline.py --handed left
CUDA_VISIBLE_DEVICES=0 python fine-tuning_allbaseline.py --handed right