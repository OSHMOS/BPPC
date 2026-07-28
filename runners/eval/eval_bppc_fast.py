import os
import sys
import gc
import cv2
import argparse
import numpy as np
import torch
from tabulate import tabulate
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lib.bppc.utils_allbaseline import generate_heatmap
from utils_bppc.accuracy_bppc import cal_acc

def process_folder(args_tuple):
    folder_number, handed_option, model_name, input_folder = args_tuple
    images_list = sorted(os.listdir(f"{input_folder}/{folder_number}"))
    one_image = cv2.imread(f"{input_folder}/{folder_number}/{images_list[0]}")
    if one_image is None:
        return None
    width = one_image.shape[1]
    height = one_image.shape[0]

    results_path = f'demo/bppc/{handed_option}/{model_name}/{folder_number}_results.npz'
    if not os.path.exists(results_path):
        return None

    results = np.load(results_path)
    bppc_kpts = results['bppc_kpts']
    base_kpts = results['base_kpts']
    gt_kpts = results['gt_kpts']

    gt_heatmap = generate_heatmap(height, width, gt_kpts)
    hrnet_heatmap = generate_heatmap(height, width, base_kpts)
    bppc_heatmap = generate_heatmap(height, width, bppc_kpts)

    def get_acc(hr_h, bp_h, gt_h):
        acc_h, acc_a = cal_acc(hr_h, bp_h, gt_h)
        return acc_h.val, acc_a.val

    body_res = {}
    body_res["head_h"], body_res["head_a"] = get_acc(hrnet_heatmap[:,0:1,:,:], bppc_heatmap[:,0:1,:,:], gt_heatmap[:,0:1,:,:])
    body_res["sho_h"], body_res["sho_a"] = get_acc(hrnet_heatmap[:,1:3,:,:], bppc_heatmap[:,1:3,:,:], gt_heatmap[:,1:3,:,:])
    body_res["elb_h"], body_res["elb_a"] = get_acc(hrnet_heatmap[:,3:5,:,:], bppc_heatmap[:,3:5,:,:], gt_heatmap[:,3:5,:,:])
    body_res["wri_h"], body_res["wri_a"] = get_acc(hrnet_heatmap[:,5:7,:,:], bppc_heatmap[:,5:7,:,:], gt_heatmap[:,5:7,:,:])
    body_res["hip_h"], body_res["hip_a"] = get_acc(hrnet_heatmap[:,7:9,:,:], bppc_heatmap[:,7:9,:,:], gt_heatmap[:,7:9,:,:])
    body_res["knee_h"], body_res["knee_a"] = get_acc(hrnet_heatmap[:,9:11,:,:], bppc_heatmap[:,9:11,:,:], gt_heatmap[:,9:11,:,:])
    body_res["ank_h"], body_res["ank_a"] = get_acc(hrnet_heatmap[:,11:13,:,:], bppc_heatmap[:,11:13,:,:], gt_heatmap[:,11:13,:,:])
    body_res["avg_h"], body_res["avg_a"] = get_acc(hrnet_heatmap, bppc_heatmap, gt_heatmap)

    return body_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--handed", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    handed_option = args.handed
    model_name = args.model_name
    input_folder = f'data/images/{handed_option}_final/'
    folders = sorted(os.listdir(input_folder))

    tasks = [(f, handed_option, model_name, input_folder) for f in folders]

    with Pool(processes=min(16, cpu_count())) as pool:
        results = pool.map(process_folder, tasks)

    results = [r for r in results if r is not None]
    cnt = len(results)

    if cnt > 0:
        keys = ["head_h", "head_a", "sho_h", "sho_a", "elb_h", "elb_a", "wri_h", "wri_a",
                "hip_h", "hip_a", "knee_h", "knee_a", "ank_h", "ank_a", "avg_h", "avg_a"]
        sums = {k: sum(r[k] for r in results) / cnt for k in keys}

        final_avg_body = [
            [model_name, sums["head_h"], sums["sho_h"], sums["elb_h"], sums["wri_h"], sums["hip_h"], sums["knee_h"], sums["ank_h"], sums["avg_h"]],
            [model_name+'+bppc', sums["head_a"], sums["sho_a"], sums["elb_a"], sums["wri_a"], sums["hip_a"], sums["knee_a"], sums["ank_a"], sums["avg_a"]],
        ]

        table = tabulate(final_avg_body, headers=['model', 'head', 'sho', 'elb', 'wri', 'hip', 'knee', 'ank', 'avg'])
        print(f"\n=== Evaluation Result: {model_name} ({handed_option}) ===")
        print(table)

        output_dir = f'demo/output/basic_results/{model_name}/{handed_option}'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/1_0.5_5_0.01_results.txt', 'w') as f:
            f.write(table)

        print(f'Successfully saved evaluation results for {model_name} ({handed_option})')
