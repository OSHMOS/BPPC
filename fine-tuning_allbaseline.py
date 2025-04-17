import os
import gc
import re
import cv2
import glob
import math
import time
import torch
import argparse # for parameter
import numpy as np
from tabulate import tabulate
from lib.bppc.utils_allbaseline import input_img2video, generate_heatmap
from lib.backbone.lib.utils.evaluate import accuracy
from lib.dataset.bppc_dataset import hrnet_to_bppc, gt_to_bppc, scores_to_13, bppc_to_13, hr_to_13
from lib.preprocess import h36m_coco_format
from bppc import BPPC
from visualize_bppc import Visualize_BPPC
from accuracy_bppc import cal_conf_acc, cal_acc, AverageMeter

torch.cuda.empty_cache()

# 시드 값을 고정합니다.
seed = 1
torch.manual_seed(seed)  # PyTorch 시드 고정
np.random.seed(seed)     # NumPy 시드 고정

# 재현성을 높이기 위해 추가적으로 설정 (x)
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

# 재현성을 높이기 위해 추가적으로 설정 (o)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def clear_gpu_memory(*args):
    """사용하지 않는 변수들을 명시적으로 삭제하고 GPU 캐시 비우기"""
    for var in args:
        del var
    torch.cuda.empty_cache()
    gc.collect()


def body_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, prefix):
    acc_h, acc_a = cal_acc(hrnet_heatmap, bppc_heatmap, gt_heatmap)
    accs_body[f"{prefix}_h"].append(acc_h.val)
    accs_body[f"{prefix}_a"].append(acc_a.val)


def conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, indices, prefix):
    for idx in indices:
        indices_dic_conf[f"{prefix}"].append(idx)
        acc_h, acc_a = cal_conf_acc(hrnet_heatmap, bppc_heatmap, gt_heatmap, idx)
        accs_conf[f"{prefix}_h"].append(acc_h.val)
        accs_conf[f"{prefix}_a"].append(acc_a.val)


parser = argparse.ArgumentParser(description="BPPC")
parser.add_argument("--handed", type=str, default="", help="Handed option for batters (left or right)")
parser.add_argument("--lambda_ohkm", type=float, default="1", help="lambda ohkm")
parser.add_argument("--lambda_reg", type=float, default="0.5", help="lambda reg")
parser.add_argument("--lambda_vel", type=float, default="5", help="lambda vel")
parser.add_argument("--lambda_accel", type=float, default="0.01", help="lambda accel")
args = parser.parse_args()

if __name__ == '__main__':
    handed_option = args.handed
    lambda_ohkm = args.lambda_ohkm
    lambda_reg = args.lambda_reg
    lambda_vel = args.lambda_vel
    lambda_accel = args.lambda_accel

    model_names = ['res50', 'res101', 'res152', 'hw32', 'hw48', 'darkw32', 'darkw48']

    input_folder = f'data/images/{handed_option}_final/'
    
    for model_name in model_names:
        # for conf score resuts
        sum_avg_u05_h = 0
        sum_avg_u05_a = 0

        sum_avg_0506_h = 0
        sum_avg_0506_a = 0

        sum_avg_0607_h = 0
        sum_avg_0607_a = 0

        sum_avg_0708_h = 0
        sum_avg_0708_a = 0

        sum_avg_0809_h = 0
        sum_avg_0809_a = 0

        sum_avg_o09_h = 0
        sum_avg_o09_a = 0

        # for basic results
        sum_avg_h = 0
        sum_avg_h_head = 0
        sum_avg_h_sho = 0
        sum_avg_h_elb = 0
        sum_avg_h_wri = 0
        sum_avg_h_hip = 0
        sum_avg_h_knee = 0
        sum_avg_h_ank = 0

        sum_avg_a = 0
        sum_avg_a_head = 0
        sum_avg_a_sho = 0
        sum_avg_a_elb = 0
        sum_avg_a_wri = 0
        sum_avg_a_hip = 0
        sum_avg_a_knee = 0
        sum_avg_a_ank = 0
        
        cnt = 0

        if model_name == 'res50':
            from lib.bppc.utils_allbaseline import get_pose2D_r50 as get_pose_2D
        if model_name == 'res101':
            from lib.bppc.utils_allbaseline import get_pose2D_r101 as get_pose_2D
        if model_name == 'res152':
            from lib.bppc.utils_allbaseline import get_pose2D_r152 as get_pose_2D
        if model_name == 'hw32':
            from lib.bppc.utils_allbaseline import get_pose2D_hw32 as get_pose_2D
        if model_name == 'hw48':
            from lib.bppc.utils_allbaseline import get_pose2D_hw48 as get_pose_2D
        if model_name == 'darkw32':
            from lib.bppc.utils_allbaseline import get_pose2D_darkw32 as get_pose_2D
        if model_name == 'darkw48':
            from lib.bppc.utils_allbaseline import get_pose2D_darkw48 as get_pose_2D
        
        folders = sorted(os.listdir(input_folder))

        for folder_number in folders:
            accs_body = {
                "head_h": [], "head_a": [],
                "sho_h": [], "sho_a": [],
                "elb_h": [], "elb_a": [],
                "wri_h": [], "wri_a": [],
                "hip_h": [], "hip_a": [],
                "knee_h": [], "knee_a": [],
                "ank_h": [], "ank_a": [],
                "avg_h": [], "avg_a": [],
            }

            accs_conf = {
                "u05_h": [], "u05_a": [],
                "0506_h": [], "0506_a": [],
                "0607_h": [], "0607_a": [],
                "0708_h": [], "0708_a": [],
                "0809_h": [], "0809_a": [],
                "o09_h": [], "o09_a": []
            }

            indices_dic_conf = {
                "u05": [],
                "0506": [],
                "0607": [],
                "0708": [],
                "0809": [],
                "o09": [],
            }

            cnt += 1

            print(f"{model_name} ing")
            print(f"{folder_number} start")
            model_name = f'{model_name}'
            images_list = sorted(os.listdir(f"{input_folder}/{folder_number}"))
            one_image = cv2.imread(f"{input_folder}/{folder_number}/{images_list[0]}")
            # print(one_image.shape) # h, w, c
            width = one_image.shape[1]
            height = one_image.shape[0]
            image_shape = [width, height]

            # if you want to generate 2D pose w/ ckeckpoints
            # all_kpts = []
            # all_scores = []
            # print('\nGenerating 2D pose...')
            # for image_path in images_list:
            #     image = cv2.imread(f"{input_folder}/{folder_number}/{image_path}")

            #     kpts, scores = get_pose_2D(image)
            #     all_kpts.append(kpts)
            #     all_scores.append(scores)
            # print('Generating 2D pose successfully!')
            
            # # f, b, k, 2 -> b, f, k, 2
            # all_kpts = np.array(all_kpts).transpose(1, 0, 2, 3)
            # all_scores = np.array(all_scores).transpose(1, 0, 2)

            # kpts, scores, valid_frames = h36m_coco_format(all_kpts, all_scores) # all
            # kpts, scores = hrnet_to_bppc(image_shape, kpts, scores)
            # hrnet_pred = torch.tensor(kpts).cuda().type(torch.float32)
            #
            
            # I recommend to save baseline kpts and scores into 'data/frozen'
            # frozen_dir = f'data/frozen/{model_name}/{folder_number}'
            # if not os.path.exists(frozen_dir):
            #     os.makedirs(frozen_dir)

            # np.savez(f'{frozen_dir}/kpts.npz', kpts=hrnet_pred.detach().cpu())
            # np.savez(f'{frozen_dir}/scores.npz', scores=scores)

            # print(f"{folder_number} end")
            # print('---------------------')

            # for save baseline kpts and scores (frozen)
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

            ####
            gt_kpts = np.load(f'data/gt_2D/{handed_option}/{folder_number}_gt.npz')["keypoints"]
            
            gt_kpts = gt_kpts.reshape(1, gt_kpts.shape[0], 13, 2)
            gt_kpts = gt_to_bppc(image_shape, gt_kpts)
            gt_kpts = torch.tensor(gt_kpts).cuda().reshape(-1, 13, 2)
            gt_kpts = np.array(gt_kpts.cpu())*image_shape[:2][::-1]

            kpts = hrnet_pred.clone().detach().cuda().reshape(-1, 17, 2)
            kpts = hr_to_13(kpts)
            kpts = np.array(kpts)*image_shape[:2][::-1]

            bppc_kpts = bppc.bppc_kpts.reshape(-1, 17, 2)
            bppc_kpts = bppc_to_13(bppc_kpts)
            bppc_kpts = np.array(bppc_kpts)*image_shape[:2][::-1]

            # generate gt_heatmap # ..._kpts의 shape은 F, K, 2
            gt_heatmap = generate_heatmap(height, width, gt_kpts).clone().detach().cuda()
            hrnet_heatmap = generate_heatmap(height, width, kpts).clone().detach().cuda()
            bppc_heatmap = generate_heatmap(height, width, bppc_kpts).clone().detach().cuda()

            # visualize
            # visualizer = Visualize_BPPC(handed_option, bppc.bppc_kpts, hrnet_pred, bppc.sm_kpts, gt_kpts, folder_number=folder_number)
            # visualizer.visualize(images_list, image_shape, model_name, folder_number)
            
            # for basic results
            # head
            body_accuracy(hrnet_heatmap[:,0:1,:,:], bppc_heatmap[:,0:1,:,:], gt_heatmap[:,0:1,:,:], "head")
            # sholder
            body_accuracy(hrnet_heatmap[:,1:3,:,:], bppc_heatmap[:,1:3,:,:], gt_heatmap[:,1:3,:,:], "sho")
            # elbow
            body_accuracy(hrnet_heatmap[:,3:5,:,:], bppc_heatmap[:,3:5,:,:], gt_heatmap[:,3:5,:,:], "elb")
            # wrist
            body_accuracy(hrnet_heatmap[:,5:7,:,:], bppc_heatmap[:,5:7,:,:], gt_heatmap[:,5:7,:,:], "wri")
            # hip
            body_accuracy(hrnet_heatmap[:,7:9,:,:], bppc_heatmap[:,7:9,:,:], gt_heatmap[:,7:9,:,:], "hip")
            # knee
            body_accuracy(hrnet_heatmap[:,9:11,:,:], bppc_heatmap[:,9:11,:,:], gt_heatmap[:,9:11,:,:], "knee")
            # ankle
            body_accuracy(hrnet_heatmap[:,11:13,:,:], bppc_heatmap[:,11:13,:,:], gt_heatmap[:,11:13,:,:], "ank")
            # avg
            body_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, "avg")

            avg_values_body = {key: (sum(val) / len(val) if len(val) != 0 else 0) for key, val in accs_body.items()}

            sum_avg_h_head += avg_values_body["head_h"]
            sum_avg_a_head += avg_values_body["head_a"]

            sum_avg_h_sho += avg_values_body["sho_h"]
            sum_avg_a_sho += avg_values_body["sho_a"]

            sum_avg_h_elb += avg_values_body["elb_h"]
            sum_avg_a_elb += avg_values_body["elb_a"]

            sum_avg_h_wri += avg_values_body["wri_h"]
            sum_avg_a_wri += avg_values_body["wri_a"]

            sum_avg_h_hip += avg_values_body["hip_h"]
            sum_avg_a_hip += avg_values_body["hip_a"]

            sum_avg_h_knee += avg_values_body["knee_h"]
            sum_avg_a_knee += avg_values_body["knee_a"]

            sum_avg_h_ank += avg_values_body["ank_h"]
            sum_avg_a_ank += avg_values_body["ank_a"]

            sum_avg_h += avg_values_body["avg_h"]
            sum_avg_a += avg_values_body["avg_a"]

            # for conf results
            under_05_indices = np.where(scores < 0.5)
            between_05_06_indices = np.where((scores >= 0.5) & (scores < 0.6))
            between_06_07_indices = np.where((scores >= 0.6) & (scores < 0.7))
            between_07_08_indices = np.where((scores >= 0.7) & (scores < 0.8))
            between_08_09_indices = np.where((scores >= 0.8) & (scores < 0.9))
            over_09_indices = np.where(scores >= 0.9)

            conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, scores_to_13(under_05_indices), "u05")
            conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, scores_to_13(between_05_06_indices), "0506")
            conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, scores_to_13(between_06_07_indices), "0607")
            conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, scores_to_13(between_07_08_indices), "0708")
            conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, scores_to_13(between_08_09_indices), "0809")
            conf_accuracy(hrnet_heatmap, bppc_heatmap, gt_heatmap, scores_to_13(over_09_indices), "o09")

            avg_values_conf = {key: (sum(val) / len(val) if len(val) != 0 else 0) for key, val in accs_conf.items()}

            sum_avg_u05_h += avg_values_conf["u05_h"]
            sum_avg_u05_a += avg_values_conf["u05_a"]

            sum_avg_0506_h += avg_values_conf["0506_h"]
            sum_avg_0506_a += avg_values_conf["0506_a"]

            sum_avg_0607_h += avg_values_conf["0607_h"]
            sum_avg_0607_a += avg_values_conf["0607_a"]

            sum_avg_0708_h += avg_values_conf["0708_h"]
            sum_avg_0708_a += avg_values_conf["0708_a"]

            sum_avg_0809_h += avg_values_conf["0809_h"]
            sum_avg_0809_a += avg_values_conf["0809_a"]

            sum_avg_o09_h += avg_values_conf["o09_h"]
            sum_avg_o09_a += avg_values_conf["o09_a"]

            # 메모리 해제 및 캐시 정리
            clear_gpu_memory(hrnet_pred, kpts, bppc_kpts, gt_heatmap, hrnet_heatmap, bppc_heatmap)

            print(f"{folder_number} end")
            print('---------------------')

            # break
        
        # 최종 평균 계산 및 저장
        final_avg_conf = [
            [model_name, sum_avg_u05_h/cnt, sum_avg_0506_h/cnt, sum_avg_0607_h/cnt, sum_avg_0708_h/cnt, sum_avg_0809_h/cnt, sum_avg_o09_h/cnt],
            [model_name+' + bppc', sum_avg_u05_a/cnt, sum_avg_0506_a/cnt, sum_avg_0607_a/cnt, sum_avg_0708_a/cnt, sum_avg_0809_a/cnt, sum_avg_o09_a/cnt]
        ]
        table = tabulate(final_avg_conf, headers=['model', 'under 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', 'over 0.9'])
        print('conf results')
        print(table)
        
        os.makedirs(f'demo/output/conf_results/{model_name}/{handed_option}', exist_ok=True)
        with open(f'demo/output/conf_results/{model_name}/{handed_option}/{lambda_ohkm}_{lambda_reg}_{lambda_vel}_{lambda_accel}_results.txt', 'w') as f:
            f.write(table)

        final_avg_body = [
            [model_name, sum_avg_h_head/cnt, sum_avg_h_sho/cnt, sum_avg_h_elb/cnt, sum_avg_h_wri/cnt, sum_avg_h_hip/cnt, sum_avg_h_knee/cnt, sum_avg_h_ank/cnt, sum_avg_h/cnt],
            [model_name+'+bppc', sum_avg_a_head/cnt, sum_avg_a_sho/cnt, sum_avg_a_elb/cnt, sum_avg_a_wri/cnt, sum_avg_a_hip/cnt, sum_avg_a_knee/cnt, sum_avg_a_ank/cnt, sum_avg_a/cnt],
            ]
        
        table = tabulate(final_avg_body, headers=['model', 'head', 'sho', 'elb', 'wri', 'hip', 'knee', 'ank', 'avg'])
        print('\nbasic results')
        print(table)

        os.makedirs(f'demo/output/basic_results/{model_name}/{handed_option}', exist_ok=True)
        with open(f'demo/output/basic_results/{model_name}/{handed_option}/{lambda_ohkm}_{lambda_reg}_{lambda_vel}_{lambda_accel}_results.txt', 'w') as f:
            f.write(table)
        
        print('---------------------')
        print('complete to save the results') 