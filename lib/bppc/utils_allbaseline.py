import os
import re
import cv2
import glob
import math
import torch
from scipy.ndimage import gaussian_filter
import numpy as np
from lib.backbone.gen_kpts_resnet50 import gen_image_kpts as resnet50_pose
from lib.backbone.gen_kpts_resnet101 import gen_image_kpts as resnet101_pose
from lib.backbone.gen_kpts_resnet152 import gen_image_kpts as resnet152_pose
from lib.backbone.gen_kpts_hw32 import gen_image_kpts as hw32_pose
from lib.backbone.gen_kpts_hw48 import gen_image_kpts as hw48_pose
from lib.backbone.gen_kpts_darkw32 import gen_image_kpts as darkw32_pose
from lib.backbone.gen_kpts_darkw48 import gen_image_kpts as darkw48_pose
# from lib.backbone.gen_kpts_darkw48 import gen_video_kpts as darkw48_pose # input : video

torch.cuda.empty_cache()


def input_img2video(image_folder, video_name):
    # images = sorted(os.listdir(image_folder), key=sort_key)
    images = sorted(os.listdir(image_folder))

    # 이미지를 기준으로 비디오 크기와 프레임 속도를 결정합니다.
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))
    # video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def sort_key(s):
    # 이미지 파일명에서 숫자를 추출하여 정렬 기준으로 사용
    return int(re.search(r'\d+', s).group())


def generate_heatmap(height, width, all_keypoints, sigma=3):
    num_images, num_keypoints, _ = all_keypoints.shape
    heatmaps = torch.zeros((num_images, num_keypoints, height, width), dtype=torch.float32)

    for i in range(num_images):
        for j in range(num_keypoints):
            x, y = int(all_keypoints[i, j][0]), int(all_keypoints[i, j][1])

            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            heatmaps[i, j, y, x] = 1
            heatmaps[i, j] = torch.from_numpy(gaussian_filter(heatmaps[i, j].numpy(), sigma, mode='constant'))
    
    return heatmaps


def get_pose2D_r50(image):
    with torch.no_grad():
        keypoints, scores = resnet50_pose(image, det_dim=416, num_person=1)

    return keypoints, scores


def get_pose2D_r101(image):
    with torch.no_grad():
        keypoints, scores = resnet101_pose(image, det_dim=416, num_person=1)

    return keypoints, scores


def get_pose2D_r152(image):
    with torch.no_grad():
        keypoints, scores = resnet152_pose(image, det_dim=416, num_person=1)

    return keypoints, scores


def get_pose2D_hw32(image):
    with torch.no_grad():
        keypoints, scores = hw32_pose(image, det_dim=416, num_person=1)

    return keypoints, scores


def get_pose2D_hw48(image):
    with torch.no_grad():
        keypoints, scores = hw48_pose(image, det_dim=416, num_person=1)

    return keypoints, scores


def get_pose2D_darkw32(image):
    with torch.no_grad():
        keypoints, scores = darkw32_pose(image, det_dim=416, num_person=1)

    return keypoints, scores


def get_pose2D_darkw48(image):
    with torch.no_grad():
        # keypoints, scores, output = darkw48_pose(video_path, det_dim=416, num_person=1, gen_output=True) # input : video with gen_video_kpts
        keypoints, scores = darkw48_pose(image, det_dim=416, num_person=1)

    return keypoints, scores