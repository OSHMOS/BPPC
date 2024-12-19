import cv2
import torch
import numpy as np

video_path = 'data/video/input_video.mp4'

def hrnet_to_bppc(image_shape, hrnet_kpts, hrnet_scores):
  hrnet_data_len = len(hrnet_kpts[0])
  hrnet_pred_data = hrnet_kpts[0].reshape(hrnet_data_len, -1)

  hrnet_pred_data = hrnet_pred_data.reshape(-1, 2) / image_shape[:2][::-1]
  hrnet_pred_data = hrnet_pred_data.reshape(-1, 34) # 17 keypoints 2(x, y)


  hrnet_pred_data_ = []
  hrnet_pred_data_.append(hrnet_pred_data)
  hrnet_kpts = np.array(np.array(hrnet_pred_data_))

  np.savez('data/hrnet_2D/hrnet_2D.npz', keypoints=hrnet_kpts, scores=hrnet_scores)

  return hrnet_kpts, hrnet_scores


def gt_to_bppc(image_shape, gt_kpts):
  gt_data_len = len(gt_kpts[0])
  gt_data = gt_kpts[0].reshape(gt_data_len, -1)

  gt_data = gt_data.reshape(-1, 2) / image_shape[:2][::-1]
  # gt_data = gt_data.reshape(-1, 26) # 13 keypoints 2(x, y)
  
  # mlb
  # gt_data = gt_data.reshape(-1, 2) / data_imageshape[:2][::-1]

  gt_data_ = []
  gt_data_.append(gt_data)
  gt_kpts = np.array(np.array(gt_data_))

  # np.savez('data/gt_2D/gt_2D.npz', keypoints=gt_kpts)

  return gt_kpts

def scores_to_13(scores_idx, new_idx_list=None):
  new_idx_list = []
  # 1 - 7, 2 - 9, 3 - 11, 4 - 8, 5 - 10, 6 - 12, 10 - 0, 11 - 2, 12 - 4, 13 - 6, 14 - 1, 15 - 3, 16 - 5
  for idx in zip(*scores_idx):
    idx_list = list(idx)
    if idx[-1] in {0, 7, 8, 9}:
        continue
    if idx[-1] == 1:
        idx_list[-1] = 7
    if idx[-1] == 2:
        idx_list[-1] = 9
    if idx[-1] == 3:
        idx_list[-1] = 11
    if idx[-1] == 4:
        idx_list[-1] = 8
    if idx[-1] == 5:
        idx_list[-1] = 10
    if idx[-1] == 6:
        idx_list[-1] = 12
    if idx[-1] == 10:
        idx_list[-1] = 0
    if idx[-1] == 11:
        idx_list[-1] = 2
    if idx[-1] == 12:
        idx_list[-1] = 4
    if idx[-1] == 13:
        idx_list[-1] = 6
    if idx[-1] == 14:
        idx_list[-1] = 1
    if idx[-1] == 15:
        idx_list[-1] = 3
    if idx[-1] == 16:
        idx_list[-1] = 5
    new_idx_list.append(tuple(idx_list))
  return new_idx_list

def bppc_to_13(bppc_kpts): # bppc_kpts (16, 17, 2)
  bppc_temp = bppc_kpts.detach().cpu()
  bppc_result = []
  for i in range(bppc_temp.shape[0]):
    bppc_return = []
    bppc_return.append(bppc_temp[i][10])
    bppc_return.append(bppc_temp[i][14])
    bppc_return.append(bppc_temp[i][11])
    bppc_return.append(bppc_temp[i][15])
    bppc_return.append(bppc_temp[i][12])
    bppc_return.append(bppc_temp[i][16])
    bppc_return.append(bppc_temp[i][13])
    bppc_return.append(bppc_temp[i][1])
    bppc_return.append(bppc_temp[i][4])
    bppc_return.append(bppc_temp[i][2])
    bppc_return.append(bppc_temp[i][5])
    bppc_return.append(bppc_temp[i][3])
    bppc_return.append(bppc_temp[i][6])
    bppc_result.append(bppc_return)
  bppc_result = np.array(bppc_result)
  
  return torch.tensor(bppc_result)


def hr_to_13(hrnet_kpts):
  hrnet_temp = hrnet_kpts.detach().cpu()
  hrnet_result = []
  for i in range(hrnet_temp.shape[0]):
    hrnet_return = []
    hrnet_return.append(hrnet_temp[i][10])
    hrnet_return.append(hrnet_temp[i][14])
    hrnet_return.append(hrnet_temp[i][11])
    hrnet_return.append(hrnet_temp[i][15])
    hrnet_return.append(hrnet_temp[i][12])
    hrnet_return.append(hrnet_temp[i][16])
    hrnet_return.append(hrnet_temp[i][13])
    hrnet_return.append(hrnet_temp[i][1])
    hrnet_return.append(hrnet_temp[i][4])
    hrnet_return.append(hrnet_temp[i][2])
    hrnet_return.append(hrnet_temp[i][5])
    hrnet_return.append(hrnet_temp[i][3])
    hrnet_return.append(hrnet_temp[i][6])
    hrnet_result.append(hrnet_return)
  hrnet_result = np.array(hrnet_result)
  
  return hrnet_result