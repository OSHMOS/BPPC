import os
import gc
import re
import cv2
import glob
import math
import time
import torch
import numpy as np
from tabulate import tabulate
from lib.bppc.utils_allbaseline import input_img2video, img2video, generate_heatmap
from lib.backbone.lib.utils.evaluate import accuracy
from lib.dataset.bppc_dataset import hrnet_to_bppc, gt_to_bppc, scores_to_13, bppc_to_13, hr_to_13
from lib.preprocess import h36m_coco_format
from bppc_left import BPPC
from visualize_bppc_left import Visualize_BPPC
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

if __name__ == '__main__':
    # for cal time
    start = time.time()

    model_names = ['res50', 'res101', 'res152', 'hw32', 'hw48', 'darkw32', 'darkw48']
    # model_names = ['res101'] # more exp
    # model_names = ['hw32'] # for rebuttal
    # model_names = ['darkw48'] # for rebuttal

    # # left
    input_folder = 'data/images/left_final/'
    # # left before for reproduce              
    
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
        sum_h = 0
        sum_h_head = 0
        sum_h_shol = 0
        sum_h_elb = 0
        sum_h_wri = 0
        sum_h_hip = 0
        sum_h_knee = 0
        sum_h_ank = 0

        sum_a = 0
        sum_a_head = 0
        sum_a_shol = 0
        sum_a_elb = 0
        sum_a_wri = 0
        sum_a_hip = 0
        sum_a_knee = 0
        sum_a_ank = 0

        avg_h = 0
        avg_h_head = 0
        avg_h_shol = 0
        avg_h_elb = 0
        avg_h_wri = 0
        avg_h_hip = 0
        avg_h_knee = 0
        avg_h_ank = 0

        avg_a = 0
        avg_a_head = 0
        avg_a_shol = 0
        avg_a_elb = 0
        avg_a_wri = 0
        avg_a_hip = 0
        avg_a_knee = 0
        avg_a_ank = 0
        
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
        # print(folders)
        for folder_number in folders:
            cnt += 1
            # folder_number = '0264'
            print(f"{model_name} ing")
            print(f"{folder_number} start")
            model_name = f'{model_name}'
            images_list = sorted(os.listdir(f"{input_folder}/{folder_number}"))
            one_image = cv2.imread(f"{input_folder}/{folder_number}/{images_list[0]}")
            # print(one_image.shape) # h, w, c
            width = one_image.shape[1]
            height = one_image.shape[0]
            image_shape = [width, height]

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

            # for save baseline kpts and scores (frozen)
            frozen_dir = f'data/frozen/{model_name}/{folder_number}'
            hrnet_pred = torch.from_numpy(np.load(f'{frozen_dir}/kpts.npz')['kpts']).cuda().type(torch.float32)
            scores = np.load(f'{frozen_dir}/scores.npz')['scores']

            # for save baseline kpts and scores (frozen)
            # frozen_dir = f'data/frozen/{model_name}/{folder_number}'
            # if not os.path.exists(frozen_dir):
            #     os.makedirs(frozen_dir)

            # np.savez(f'{frozen_dir}/kpts.npz', kpts=hrnet_pred.detach().cpu())
            # np.savez(f'{frozen_dir}/scores.npz', scores=scores)

            # print(f"{folder_number} end")
            # print('---------------------')
            # end = time.time()

            # continue

            print('\nRefining 2D pose...')
            bppc = BPPC(pred=hrnet_pred, c_scores=scores)
            bppc.optimize(1000) # optimize time, projection, sm
            bppc.optimize_kp(1000) # optimize kpts
            print('Refining 2D pose successfully!')

            ####
            gt_kpts = np.load(f'data/gt_2D/left/{folder_number}_gt.npz')["keypoints"]
            
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
            visualizer = Visualize_BPPC(bppc.bppc_kpts, bppc.sm_kpts, gt_kpts, folder_number=folder_number)
            visualizer.visualize(images_list, image_shape, model_name, folder_number)
            
            # img2video(video_path, number, model_name)

            # accuracy
            acc_h = AverageMeter()
            acc_a = AverageMeter()
            
            # head
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,0:1,:,:].cpu().numpy(), gt_heatmap[:,0:1,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,0:1,:,:].cpu().numpy(), gt_heatmap[:,0:1,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_head += acc_h.val
            sum_a_head += acc_a.val

            avg_h_head = acc_h.avg
            avg_a_head = acc_a.avg

            # sholder
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,1:3,:,:].cpu().numpy(), gt_heatmap[:,1:3,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,1:3,:,:].cpu().numpy(), gt_heatmap[:,1:3,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_shol += acc_h.val
            sum_a_shol += acc_a.val

            avg_h_shol = acc_h.avg
            avg_a_shol = acc_a.avg

            # elbow
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,3:5,:,:].cpu().numpy(), gt_heatmap[:,3:5,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,3:5,:,:].cpu().numpy(), gt_heatmap[:,3:5,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_elb += acc_h.val
            sum_a_elb += acc_a.val

            avg_h_elb = acc_h.avg
            avg_a_elb = acc_a.avg

            # wrist
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,5:7,:,:].cpu().numpy(), gt_heatmap[:,5:7,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,5:7,:,:].cpu().numpy(), gt_heatmap[:,5:7,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_wri += acc_h.val
            sum_a_wri += acc_a.val

            avg_h_wri = acc_h.avg
            avg_a_wri = acc_a.avg

            # hip
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,7:9,:,:].cpu().numpy(), gt_heatmap[:,7:9,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,7:9,:,:].cpu().numpy(), gt_heatmap[:,7:9,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_hip += acc_h.val
            sum_a_hip += acc_a.val

            avg_h_hip = acc_h.avg
            avg_a_hip = acc_a.avg

            # knee
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,9:11,:,:].cpu().numpy(), gt_heatmap[:,9:11,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,9:11,:,:].cpu().numpy(), gt_heatmap[:,9:11,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_knee += acc_h.val
            sum_a_knee += acc_a.val

            avg_h_knee = acc_h.avg
            avg_a_knee = acc_a.avg

            # ankle
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap[:,11:13,:,:].cpu().numpy(), gt_heatmap[:,11:13,:,:].cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap[:,11:13,:,:].cpu().numpy(), gt_heatmap[:,11:13,:,:].cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h_ank += acc_h.val
            sum_a_ank += acc_a.val

            avg_h_ank = acc_h.avg
            avg_a_ank = acc_a.avg

            # all avg
            _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap.cpu().numpy(), gt_heatmap.cpu().numpy())
            _, avg_acc_a, cnt_a, pred_a = accuracy(bppc_heatmap.cpu().numpy(), gt_heatmap.cpu().numpy())
            acc_h.update(avg_acc_h, cnt_h)
            acc_a.update(avg_acc_a, cnt_a)

            sum_h += acc_h.val
            sum_a += acc_a.val

            avg_h = acc_h.avg
            avg_a = acc_a.avg

            # 정확도 계산
            # under_05_indices = np.where(scores < 0.5)
            # between_05_06_indices = np.where((scores >= 0.5) & (scores < 0.6))
            # between_06_07_indices = np.where((scores >= 0.6) & (scores < 0.7))
            # between_07_08_indices = np.where((scores >= 0.7) & (scores < 0.8))
            # between_08_09_indices = np.where((scores >= 0.8) & (scores < 0.9))
            # over_09_indices = np.where(scores >= 0.9)

            # accs = {
            #     "u05_h": [], "u05_a": [],
            #     "0506_h": [], "0506_a": [],
            #     "0607_h": [], "0607_a": [],
            #     "0708_h": [], "0708_a": [],
            #     "0809_h": [], "0809_a": [],
            #     "o09_h": [], "o09_a": []
            # }

            # indices_dic = {
            #     "u05": [],
            #     "0506": [],
            #     "0607": [],
            #     "0708": [],
            #     "0809": [],
            #     "o09": [],
            # }

            # for conf results
        #     def calculate_accuracy(indices, prefix):
        #         for idx in indices:
        #             indices_dic[f"{prefix}"].append(idx)
        #             acc_h, acc_a = cal_conf_acc(hrnet_heatmap, bppc_heatmap, gt_heatmap, idx)
        #             accs[f"{prefix}_h"].append(acc_h.val)
        #             accs[f"{prefix}_a"].append(acc_a.val)

        #     calculate_accuracy(scores_to_13(under_05_indices), "u05")
        #     calculate_accuracy(scores_to_13(between_05_06_indices), "0506")
        #     calculate_accuracy(scores_to_13(between_06_07_indices), "0607")
        #     calculate_accuracy(scores_to_13(between_07_08_indices), "0708")
        #     calculate_accuracy(scores_to_13(between_08_09_indices), "0809")
        #     calculate_accuracy(scores_to_13(over_09_indices), "o09")

        #     results_dir = f'demo/output/conf_results/left/{model_name}/{folder_number}/'
        #     os.makedirs(results_dir, exist_ok=True)

        #     # 결과 저장
        #     with open(f'{results_dir}/details.txt', 'w') as f:
        #         for key in accs.keys():
        #             f.write(f'Accuracies {key} : {accs[key]}\n\n')
        #         for key in indices_dic.keys():
        #             f.write(f'Indices {key} : {indices_dic[key]}\n\n')

        #     avg_values = {key: (sum(val) / len(val) if len(val) != 0 else 0) for key, val in accs.items()}
            
        #     data = [
        #         [model_name, avg_values["u05_h"], avg_values["0506_h"], avg_values["0607_h"], avg_values["0708_h"], avg_values["0809_h"], avg_values["o09_h"]],
        #         [model_name+'+bppe', avg_values["u05_a"], avg_values["0506_a"], avg_values["0607_a"], avg_values["0708_a"], avg_values["0809_a"], avg_values["o09_a"]]
        #     ]
        #     table = tabulate(data, headers=['model', 'under 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', 'over 0.9'])

        #     with open(f'{results_dir}/results.txt', 'w') as f:
        #         f.write(table)

        #     sum_avg_u05_h += avg_values["u05_h"]
        #     sum_avg_u05_a += avg_values["u05_a"]

        #     sum_avg_0506_h += avg_values["0506_h"]
        #     sum_avg_0506_a += avg_values["0506_a"]

        #     sum_avg_0607_h += avg_values["0607_h"]
        #     sum_avg_0607_a += avg_values["0607_a"]

        #     sum_avg_0708_h += avg_values["0708_h"]
        #     sum_avg_0708_a += avg_values["0708_a"]

        #     sum_avg_0809_h += avg_values["0809_h"]
        #     sum_avg_0809_a += avg_values["0809_a"]

        #     sum_avg_o09_h += avg_values["o09_h"]
        #     sum_avg_o09_a += avg_values["o09_a"]

        #     # 메모리 해제 및 캐시 정리
        #     clear_gpu_memory(hrnet_pred, kpts, bppc_kpts, gt_heatmap, hrnet_heatmap, bppc_heatmap)

        #     print(f"{folder_number} end")
        #     print('---------------------')
        #     end = time.time()
        
        # # 최종 평균 계산 및 저장
        # final_avg_data = [
        #     [model_name, sum_avg_u05_h/cnt, sum_avg_0506_h/cnt, sum_avg_0607_h/cnt, sum_avg_0708_h/cnt, sum_avg_0809_h/cnt, sum_avg_o09_h/cnt],
        #     [model_name+' + bppc', sum_avg_u05_a/cnt, sum_avg_0506_a/cnt, sum_avg_0607_a/cnt, sum_avg_0708_a/cnt, sum_avg_0809_a/cnt, sum_avg_o09_a/cnt]
        # ]
        # table = tabulate(final_avg_data, headers=['model', 'under 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', 'over 0.9'])
        
        # last_results_dir = f'demo/output/conf_results/left/{model_name}'
        # with open(f'{last_results_dir}/last_results.txt', 'w') as f:
        #     f.write(table)
        # print(cnt) # 52
        # print(end - start)
        sum_h = sum_h / cnt
        sum_h_head = sum_h_head / cnt
        sum_h_shol = sum_h_shol / cnt
        sum_h_elb = sum_h_elb / cnt
        sum_h_wri = sum_h_wri / cnt
        sum_h_hip = sum_h_hip / cnt
        sum_h_knee = sum_h_knee / cnt
        sum_h_ank = sum_h_ank / cnt

        sum_a = sum_a / cnt
        sum_a_head = sum_a_head / cnt
        sum_a_shol = sum_a_shol / cnt
        sum_a_elb = sum_a_elb / cnt
        sum_a_wri = sum_a_wri / cnt
        sum_a_hip = sum_a_hip / cnt
        sum_a_knee = sum_a_knee / cnt
        sum_a_ank = sum_a_ank / cnt

        data = [
            [model_name+'(val)', sum_h_head, sum_h_shol, sum_h_elb, sum_h_wri, sum_h_hip, sum_h_knee, sum_h_ank, sum_h],
            [model_name+'+bppc(val)', sum_a_head, sum_a_shol, sum_a_elb, sum_a_wri, sum_a_hip, sum_a_knee, sum_a_ank, sum_a],
            ]
        
        table = tabulate(data, headers=['model', 'head', 'shol', 'elb', 'wri', 'hip', 'knee', 'ank', 'avg'])
        print(table)

        os.makedirs(f'demo/output/basic_results/left/{model_name}', exist_ok=True)
        with open(f'demo/output/basic_results/left/{model_name}/results.txt', 'w') as f:
            f.write(table)
        
        print('save the results complete')