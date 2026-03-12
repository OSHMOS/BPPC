import os
import sys
import gc
import cv2
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lib.dataset.bppc_dataset import hr_to_13, bppc_to_13, gt_to_bppc
from core.model import BPPC

torch.cuda.empty_cache()

# 시드 값을 고정합니다.
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def clear_gpu_memory(*args):
    """사용하지 않는 변수들을 명시적으로 삭제하고 GPU 캐시 비우기"""
    for var in args:
        del var
    torch.cuda.empty_cache()
    gc.collect()

parser = argparse.ArgumentParser(description="BPPC Optimization")
parser.add_argument("--handed", type=str, default="", help="Handed option for batters (left or right)")
parser.add_argument("--model_name", type=str, default="all", help="Model name or 'all'")
parser.add_argument("--folder_number", type=str, default="", help="Specific folder (single test case). Leave empty for all.")
parser.add_argument("--lambda_ohkm", type=float, default=1, help="lambda ohkm")
parser.add_argument("--lambda_reg", type=float, default=0.5, help="lambda reg")
parser.add_argument("--lambda_vel", type=float, default=5, help="lambda vel")
parser.add_argument("--lambda_accel", type=float, default=0.01, help="lambda accel")
args = parser.parse_args()

if __name__ == '__main__':
    handed_option = args.handed
    lambda_ohkm = args.lambda_ohkm
    lambda_reg = args.lambda_reg
    lambda_vel = args.lambda_vel
    lambda_accel = args.lambda_accel

    if args.model_name.lower() == 'all':
        model_names = ['res50', 'res101', 'res152', 'hw32', 'hw48', 'darkw32', 'darkw48']
    else:
        model_names = [args.model_name]

    input_folder = f'data/images/{handed_option}_final/'
    
    for model_name in model_names:
        if args.folder_number:
            folders = [args.folder_number]
        else:
            folders = sorted(os.listdir(input_folder))

        for folder_number in folders:
            print(f"{model_name} processing folder {folder_number} starts")
            
            images_list = sorted(os.listdir(f"{input_folder}/{folder_number}"))
            one_image = cv2.imread(f"{input_folder}/{folder_number}/{images_list[0]}")
            width = one_image.shape[1]
            height = one_image.shape[0]
            image_shape = [width, height]

            frozen_dir = f'data/frozen/{model_name}/{folder_number}'
            hrnet_pred = torch.from_numpy(np.load(f'{frozen_dir}/kpts.npz')['kpts']).cuda().type(torch.float32)
            scores = np.load(f'{frozen_dir}/scores.npz')['scores']

            print('\nRefining 2D pose...')
            bppc = BPPC(handed_option=handed_option,
                        lambda_ohkm=lambda_ohkm,
                        lambda_reg=lambda_reg,
                        lambda_vel=lambda_vel,
                        lambda_accel=lambda_accel,
                        pred=hrnet_pred, c_scores=scores)
            bppc.optimize(1000) # optimize time, projection, sm
            bppc.optimize_kp(1000) # optimize kpts
            print('Refining 2D pose successfully!')

            gt_kpts = np.load(f'data/gt_2D/{handed_option}/{folder_number}_gt.npz')["keypoints"]
            gt_kpts = gt_kpts.reshape(1, gt_kpts.shape[0], 13, 2)
            gt_kpts = gt_to_bppc(image_shape, gt_kpts)
            gt_kpts = torch.tensor(gt_kpts).cuda().reshape(-1, 13, 2)
            gt_kpts = gt_kpts.cpu().numpy()*image_shape[:2][::-1]

            kpts = hrnet_pred.clone().detach().cuda().reshape(-1, 17, 2)
            kpts = hr_to_13(kpts)
            kpts = kpts*image_shape[:2][::-1]

            bppc_kpts = bppc.bppc_kpts.reshape(-1, 17, 2)
            bppc_kpts = bppc_to_13(bppc_kpts)
            bppc_kpts = bppc_kpts*image_shape[:2][::-1]

            output_dir = f'demo/bppc/{handed_option}/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            np.savez(os.path.join(output_dir, f'{folder_number}_results.npz'), bppc_kpts=bppc_kpts, base_kpts=kpts, gt_kpts=gt_kpts)

            clear_gpu_memory(hrnet_pred, bppc)

            print(f"{folder_number} end")
            print('---------------------')
