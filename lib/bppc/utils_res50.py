import os
import re
import cv2
import glob
import math
import torch
from scipy.ndimage import gaussian_filter
import numpy as np
from lib.preprocess import h36m_coco_format
from lib.backbone.gen_kpts_resnet50 import gen_video_kpts as resnet50_pose
from lib.backbone.lib.utils.evaluate import accuracy

torch.cuda.empty_cache()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def input_img2video(image_folder, video_name, number):
    # images = sorted(os.listdir(image_folder), key=sort_key)
    images = sorted(os.listdir(image_folder))

    # 첫 번째 이미지를 기준으로 비디오 크기와 프레임 속도를 결정합니다.
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def get_pose2D(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    with torch.no_grad():
        # the first frame of the video should be detected a person
        keypoints, scores, output = resnet50_pose(video_path, det_dim=416, num_person=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    print('Generating 2D pose successfully!')

    return keypoints, scores, output # why output num_images : 1?

    # output_dir += 'input_2D/'
    # os.makedirs(output_dir, exist_ok=True)

    # output_npz = output_dir + 'keypoints.npz'
    # np.savez_compressed(output_npz, reconstruction=keypoints)

def sort_key(s):
    # 이미지 파일명에서 숫자를 추출하여 정렬 기준으로 사용
    return int(re.search(r'\d+', s).group())

def img2video(video_path, number, model_name):
    output_dir='demo/output/backbone/{model_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    fps = 10

    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(f'{output_dir}{number}/output.mp4', fourcc, fps, (width * 4, height))

    folders = [f'demo/output/backbone/{model_name}/{number}/{model_name}/', f'demo/output/backbone/{model_name}/{number}/apc/', f'demo/output/backbone/{model_name}/{number}/sm/', f'demo/output/backbone/{model_name}/{number}/gt/']

    # mlb
    # folders = [f'demo/sm_output/hrnet/', f'demo/sm_output/apc/', f'demo/sm_output/sm/', f'demo/sm_output/gt/']

    # 각 폴더의 이미지 리스트 가져오기 & 정렬
    image_list = [sorted(os.listdir(folder), key=sort_key) for folder in folders]

    max_images = min([len(images) for images in image_list])
    
    labels = [f"{model_name}", "apc", "SM", "GT"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (0, 0, 0)

    for i in range(max_images):
        frames = []
        for j, folder in enumerate(folders):
            if not os.path.exists(folder):
                os.makedirs(folder)
            img_path = os.path.join(folder, image_list[j][i])
            img = cv2.imread(img_path)
            
            # 이미지 크기 검사 및 조정 (필요한 경우)
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            
            # 텍스트 추가
            cv2.putText(img, labels[j], (10, height - 10), font, font_scale, color, font_thickness)

            frames.append(img)
        
        # 모든 이미지를 수평으로 연결
        combined_frame = cv2.hconcat(frames)
        
        # 비디오에 프레임 추가
        videoWrite.write(combined_frame)

    videoWrite.release()
    print("Generating Output video successfully!")


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