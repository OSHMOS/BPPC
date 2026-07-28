import os
import sys
import gc
import cv2
import argparse
import numpy as np
import torch
from tabulate import tabulate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lib.bppc.utils_allbaseline import generate_heatmap
from lib.dataset.bppc_dataset import scores_to_13
from utils_bppc.accuracy_bppc import cal_conf_acc, cal_acc
from utils_bppc.visualize_bppc import Visualize_BPPC

torch.cuda.empty_cache()

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

parser = argparse.ArgumentParser(description="BPPC Evaluation")
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
        sum_avg_u05_h = sum_avg_u05_a = 0
        sum_avg_0506_h = sum_avg_0506_a = 0
        sum_avg_0607_h = sum_avg_0607_a = 0
        sum_avg_0708_h = sum_avg_0708_a = 0
        sum_avg_0809_h = sum_avg_0809_a = 0
        sum_avg_o09_h = sum_avg_o09_a = 0

        sum_avg_h = sum_avg_h_head = sum_avg_h_sho = sum_avg_h_elb = sum_avg_h_wri = sum_avg_h_hip = sum_avg_h_knee = sum_avg_h_ank = 0
        sum_avg_a = sum_avg_a_head = sum_avg_a_sho = sum_avg_a_elb = sum_avg_a_wri = sum_avg_a_hip = sum_avg_a_knee = sum_avg_a_ank = 0
        
        cnt = 0
        if args.folder_number:
            folders = [args.folder_number]
        else:
            folders = sorted(os.listdir(input_folder))

        for folder_number in folders:
            accs_body = {
                "head_h": [], "head_a": [],"sho_h": [], "sho_a": [],"elb_h": [], "elb_a": [],
                "wri_h": [], "wri_a": [],"hip_h": [], "hip_a": [],"knee_h": [], "knee_a": [],
                "ank_h": [], "ank_a": [],"avg_h": [], "avg_a": [],
            }
            accs_conf = {
                "u05_h": [], "u05_a": [],"0506_h": [], "0506_a": [],
                "0607_h": [], "0607_a": [],"0708_h": [], "0708_a": [],
                "0809_h": [], "0809_a": [],"o09_h": [], "o09_a": []
            }
            indices_dic_conf = {
                "u05": [],"0506": [],"0607": [],"0708": [],"0809": [],"o09": [],
            }
            cnt += 1

            images_list = sorted(os.listdir(f"{input_folder}/{folder_number}"))
            one_image = cv2.imread(f"{input_folder}/{folder_number}/{images_list[0]}")
            width = one_image.shape[1]
            height = one_image.shape[0]

            frozen_dir = f'data/frozen/{model_name}/{folder_number}'
            scores = np.load(f'{frozen_dir}/scores.npz')['scores']

            results_path = f'demo/bppc/{handed_option}/{model_name}/{folder_number}_results.npz'
            results = np.load(results_path)
            bppc_kpts = results['bppc_kpts']
            base_kpts = results['base_kpts']
            gt_kpts = results['gt_kpts']

            print(f"{model_name} starting evaluation for folder {folder_number}")

            gt_heatmap = generate_heatmap(height, width, gt_kpts).clone().detach().cpu()
            hrnet_heatmap = generate_heatmap(height, width, base_kpts).clone().detach().cpu()
            bppc_heatmap = generate_heatmap(height, width, bppc_kpts).clone().detach().cpu()

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

            # Visualize outputs
            # Provide sm_kpts placeholder (since we don't return it currently)
            sm_kpts_placeholder = torch.zeros_like(torch.tensor(bppc_kpts)).cuda()
            visualizer = Visualize_BPPC(handed_option, 
                                      torch.tensor(bppc_kpts).cuda().unsqueeze(0), 
                                      torch.tensor(base_kpts).cuda().unsqueeze(0), 
                                      sm_kpts_placeholder.unsqueeze(0), 
                                      torch.tensor(gt_kpts).unsqueeze(0), 
                                      folder_number)
            # visualizer.visualize(images_list, [width, height], model_name, folder_number)

            avg_values_body = {key: (sum(val) / len(val) if len(val) != 0 else 0) for key, val in accs_body.items()}
            sum_avg_h_head += avg_values_body["head_h"]; sum_avg_a_head += avg_values_body["head_a"]
            sum_avg_h_sho += avg_values_body["sho_h"]; sum_avg_a_sho += avg_values_body["sho_a"]
            sum_avg_h_elb += avg_values_body["elb_h"]; sum_avg_a_elb += avg_values_body["elb_a"]
            sum_avg_h_wri += avg_values_body["wri_h"]; sum_avg_a_wri += avg_values_body["wri_a"]
            sum_avg_h_hip += avg_values_body["hip_h"]; sum_avg_a_hip += avg_values_body["hip_a"]
            sum_avg_h_knee += avg_values_body["knee_h"]; sum_avg_a_knee += avg_values_body["knee_a"]
            sum_avg_h_ank += avg_values_body["ank_h"]; sum_avg_a_ank += avg_values_body["ank_a"]
            sum_avg_h += avg_values_body["avg_h"]; sum_avg_a += avg_values_body["avg_a"]

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
            sum_avg_u05_h += avg_values_conf["u05_h"]; sum_avg_u05_a += avg_values_conf["u05_a"]
            sum_avg_0506_h += avg_values_conf["0506_h"]; sum_avg_0506_a += avg_values_conf["0506_a"]
            sum_avg_0607_h += avg_values_conf["0607_h"]; sum_avg_0607_a += avg_values_conf["0607_a"]
            sum_avg_0708_h += avg_values_conf["0708_h"]; sum_avg_0708_a += avg_values_conf["0708_a"]
            sum_avg_0809_h += avg_values_conf["0809_h"]; sum_avg_0809_a += avg_values_conf["0809_a"]
            sum_avg_o09_h += avg_values_conf["o09_h"]; sum_avg_o09_a += avg_values_conf["o09_a"]

            clear_gpu_memory(gt_heatmap, hrnet_heatmap, bppc_heatmap)
            print(f"{folder_number} evaluation finished.")
            print('---------------------')
        
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
        print(f'complete to save {model_name} {handed_option} results')

