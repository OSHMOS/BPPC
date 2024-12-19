import os
import re
import cv2
import copy
import torch
import numpy as np
from math import sqrt

torch.cuda.empty_cache()

class Visualize_BPPC():
    def __init__(self, bppc_kpts, sm_kpts, gt_kpts, folder_number):
        super().__init__()
        self.hrnet_kpts = np.load('data/hrnet_2D/hrnet_2D.npz')
        self.bppc_kpts = bppc_kpts
        self.sm_kpts = sm_kpts
        # if list(gt_kpts) == None: # gt is Not None
        # # if gt_kpts == None:
        #     self.gt_kpts = None
        self.gt_kpts = gt_kpts

        self.folder_number = folder_number

    
    def sort_key(self, s):
        # 이미지 파일명에서 숫자를 추출하여 정렬 기준으로 사용
        return int(re.search(r'\d+', s).group())
    
    
    def show2Dpose(self, kps, img):
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=int) # real left : 0, real right : 1, center : 2

        # color : (blue, green, red)
        lcolor = (255, 0, 0) # blue
        rcolor = (0, 0, 255) # red
        ccolor = (0, 255, 0) # green
        thickness = 3

        for j,c in enumerate(connections):
            start = map(int, kps[c[0]])
            end = map(int, kps[c[1]])
            start = list(start)
            end = list(end)
            
            if LR[j] == 0:
                color = lcolor
            elif LR[j] == 1:
                color = rcolor
            else:
                color = ccolor

            cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=(255, 180, 0), radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=(255, 180, 0), radius=3)

        return img
    

    def show2Dpose_gt(self, kps, img):
        connections = [[1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12]] # 8개

        LR = np.array([2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int) # real left : 0, real right : 1, center : 2

        # color : (blue, green, red)
        lcolor = (255, 0, 0) # blue
        rcolor = (0, 0, 255) # red
        ccolor = (0, 255, 0) # green
        thickness = 3

        for j,c in enumerate(connections):
            start = map(int, kps[c[0]])
            end = map(int, kps[c[1]])
            start = list(start)
            end = list(end)
            
            if LR[j] == 0:
                color = lcolor
            elif LR[j] == 1:
                color = rcolor
            else:
                color = ccolor

            cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=(255, 180, 0), radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=(255, 180, 0), radius=3)

        return img

    def visualize(self, images_list, image_shape, model_name, number):
        output_dir1 = f'demo/output/backbone/{model_name}/left/{number}/{model_name}/'
        output_dir2 = f'demo/output/backbone/{model_name}/left/{number}/bppc/'
        output_dir3 = f'demo/output/backbone/{model_name}/left/{number}/sm/'
        output_dir4 = f'demo/output/backbone/{model_name}/left/{number}/gt/'

        # output_dir1 = f'demo/output/backbone/{model_name}/right/{number}/{model_name}/'
        # output_dir2 = f'demo/output/backbone/{model_name}/right/{number}/bppc/'
        # output_dir3 = f'demo/output/backbone/{model_name}/right/{number}/sm/'
        # output_dir4 = f'demo/output/backbone/{model_name}/right/{number}/gt/'
    
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        if not os.path.exists(output_dir3):
            os.makedirs(output_dir3)
        if not os.path.exists(output_dir4):
            os.makedirs(output_dir4)

        gt_keypoint_number = 13
        keypoint_number = 17

        hrnet_kpts = self.hrnet_kpts['keypoints'][0]
        hrnet_kpts = torch.tensor(hrnet_kpts).cuda()
        bppc_kpts = self.bppc_kpts
        sm_kpts = self.sm_kpts
        if self.gt_kpts is None:
            gt_kpts = self.hrnet_kpts['keypoints'][0]
            gt_kpts = torch.tensor(gt_kpts).cuda()
        if self.gt_kpts is not None:
            gt_kpts = self.gt_kpts
            gt_kpts = torch.tensor(gt_kpts).cuda()

        hrnet_kpts = hrnet_kpts.reshape(-1, keypoint_number, 2)
        bppc_kpts = bppc_kpts.reshape(-1, keypoint_number, 2)
        sm_kpts = sm_kpts.reshape(-1, keypoint_number, 2)
        gt_kpts = gt_kpts.reshape(-1, gt_keypoint_number, 2)

        # print(hrnet_kpts.shape)
        # print(gt_kpts.shape)
    
        hrnet_kpts = np.array(hrnet_kpts.cpu())*image_shape[:2][::-1]
        bppc_kpts = np.array(bppc_kpts.detach().cpu())*image_shape[:2][::-1]
        sm_kpts = np.array(sm_kpts.detach().cpu())*image_shape[:2][::-1]
        # gt_kpts = np.array(gt_kpts.cpu())*image_shape[:2][::-1]
        # print(hrnet_kpts.shape) # 16, 17, 2, for문 돌려서 show2Dpose 해보기
        # print(bppc_kpts.shape)
        # print(sm_kpts.shape)
        # print(gt_kpts.shape)

        # images = sorted(os.listdir(image_folder), key=self.sort_key)
        # mlb
        # images = sorted(os.listdir(image_folder))[1:]

        hrnet_images = []
        bppc_images = []
        sm_images = []
        gt_images = []

        for i in range(hrnet_kpts.shape[0]):
            try:
                img = cv2.imread(f'data/images/left_final/{self.folder_number}/{images_list[i]}')
                # img = cv2.imread(f'data/images/right/{self.folder_number}/{images_list[i]}')
                # img = cv2.imread(f'data/mlb_images/{number}/{images_list[i]}')
                # print(img)
                # height, width, channels = img.shape
                hrnet_img = self.show2Dpose(hrnet_kpts[i], copy.deepcopy(img))
                # print(hrnet_kpts[i])
                # hrnet_img = self.show2Dpose(hrnet_kpts[i], copy.deepcopy(np.ones((height, width, 3), np.uint8) * 255))
                bppc_img = self.show2Dpose(bppc_kpts[i], copy.deepcopy(img))
                sm_img = self.show2Dpose(sm_kpts[i], copy.deepcopy(img))
                gt_img = self.show2Dpose_gt(gt_kpts[i], copy.deepcopy(img))

                # print(hrnet_img)
                cv2.imwrite(f'demo/output/backbone/{model_name}/left/{number}/{model_name}/{i}.png', hrnet_img)
                cv2.imwrite(f'demo/output/backbone/{model_name}/left/{number}/bppc/{i}.png', bppc_img)
                cv2.imwrite(f'demo/output/backbone/{model_name}/left/{number}/sm/{i}.png', sm_img)
                cv2.imwrite(f'demo/output/backbone/{model_name}/left/{number}/gt/{i}.png', gt_img)

                # cv2.imwrite(f'demo/output/backbone/{model_name}/right/{number}/{model_name}/{i}.png', hrnet_img)
                # cv2.imwrite(f'demo/output/backbone/{model_name}/right/{number}/bppc/{i}.png', bppc_img)
                # cv2.imwrite(f'demo/output/backbone/{model_name}/right/{number}/sm/{i}.png', sm_img)
                # cv2.imwrite(f'demo/output/backbone/{model_name}/right/{number}/gt/{i}.png', gt_img)
                                
                # mlb
                # gt_img = self.show2Dpose(gt_kpts[i], copy.deepcopy(img))
                # cv2.imwrite(f'demo/output/backbone/{model_name}/{number}/gt/{i}.png', gt_img)
                # cv2.imwrite(f'demo/output/hrnet/{i}.png', hrnet_img)
                # cv2.imwrite(f'demo/output/bppc/{i}.png', bppc_img)
                # cv2.imwrite(f'demo/output/sm/{i}.png', sm_img)
                # cv2.imwrite(f'demo/output/backbone/gt/{i}.png', gt_img)
            except IndexError:
                continue
