import os
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm

edges = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)
]

# We only draw the standard 8 connections for evaluation comparison to match SPC
connections = [[1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12]]

def draw_skeleton(img, kps, LR, thickness=3):
    # Connections color mapping based on Left/Right/Center
    lcolor = (255, 0, 0) # blue
    rcolor = (0, 0, 255) # red
    ccolor = (0, 255, 0) # green
    
    for j, c in enumerate(connections):
        start = list(map(int, kps[c[0]]))
        end = list(map(int, kps[c[1]]))
        
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

def main():
    parser = argparse.ArgumentParser(description="Visualize BPPC PA Results")
    parser.add_argument("--pose_type", type=str, default="baseball_swing")
    parser.add_argument("--model_name", type=str, default="hw48")
    parser.add_argument("--folder_number", type=str, default="0330")
    parser.add_argument("--output_video", action="store_true", default=True)
    args = parser.parse_args()
    
    pose_type = args.pose_type
    model_name = args.model_name
    folder_number = args.folder_number
    
    # 1. Search for results.npz in left or right folders of demo/bppc/
    handed_option = None
    res_file = None
    for h in ["left", "right"]:
        path = f"demo/bppc/{h}/{model_name}/{folder_number}_results.npz"
        if os.path.exists(path):
            handed_option = h
            res_file = path
            break
            
    if not res_file:
        print(f"Error: Results file not found for {model_name}/{folder_number} in left or right folders.")
        return
        
    print(f"Found results file at: {res_file} (handed_option: {handed_option})")
    
    # Load coordinates
    results = np.load(res_file)
    gt_kpts = results['gt_kpts']
    base_kpts = results['base_kpts']
    bppc_kpts = results['bppc_kpts'] # BPPC result key
    
    # 2. Image path setup (use full frames from SPC repository)
    image_folder = f"/home/i2slab0/oshmos/spc/data/penn_action/{pose_type}/frames/{folder_number}"
    if not os.path.exists(image_folder):
        image_folder = f"data/penn_action/{pose_type}/{folder_number}"
        
    if not os.path.exists(image_folder):
        print(f"Error: Image folder {image_folder} not found.")
        return
        
    images_list = sorted(os.listdir(image_folder))
    if not images_list:
        print("Error: No images found in folder.")
        return
        
    first_img_path = os.path.join(image_folder, images_list[0])
    image = cv2.imread(first_img_path)
    height, width = image.shape[:2]
    
    # Output dir setup
    out_dir = f"demo/bppc/{handed_option}/{model_name}/{folder_number}/visualize"
    os.makedirs(out_dir, exist_ok=True)
    
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_path = os.path.join(out_dir, "comparison.mp4")
        video_writer = cv2.VideoWriter(vid_path, fourcc, 15, (width * 3, height))
        
    # LR index connections mapping (same as Visualize_BPPC)
    LR = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=int)
    
    for i in tqdm(range(min(len(images_list), bppc_kpts.shape[0])), desc="Visualizing frames"):
        img_path = os.path.join(image_folder, images_list[i])
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        b_kp = base_kpts[i]
        bp_kp = bppc_kpts[i]
        g_kp = gt_kpts[i]
        
        # Scale to image dimensions if normalized
        if np.max(b_kp) <= 1.0:
            b_kp = b_kp * [width, height]
        if np.max(bp_kp) <= 1.0:
            bp_kp = bp_kp * [width, height]
        if g_kp is not None and np.max(g_kp) <= 1.0:
            g_kp = g_kp * [width, height]
            
        canvas_base = img.copy()
        canvas_bppc = img.copy()
        canvas_gt = img.copy()
        
        # Draw skeletons
        canvas_base = draw_skeleton(canvas_base, b_kp, LR)
        canvas_bppc = draw_skeleton(canvas_bppc, bp_kp, LR)
        if g_kp is not None:
            canvas_gt = draw_skeleton(canvas_gt, g_kp, LR)
            
        # Add text labels
        cv2.putText(canvas_base, model_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas_bppc, "BPPC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas_gt, "GT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save individual frames
        dir_backbone = os.path.join(out_dir, model_name)
        dir_bppc = os.path.join(out_dir, 'bppc')
        dir_gt = os.path.join(out_dir, 'gt')
        
        os.makedirs(dir_backbone, exist_ok=True)
        os.makedirs(dir_bppc, exist_ok=True)
        if g_kp is not None:
            os.makedirs(dir_gt, exist_ok=True)
            
        cv2.imwrite(os.path.join(dir_backbone, f"frame_{i:04d}.png"), canvas_base)
        cv2.imwrite(os.path.join(dir_bppc, f"frame_{i:04d}.png"), canvas_bppc)
        if g_kp is not None:
            cv2.imwrite(os.path.join(dir_gt, f"frame_{i:04d}.png"), canvas_gt)
            
        # Concat side-by-side
        combined = np.hstack((canvas_base, canvas_bppc, canvas_gt))
        if args.output_video:
            video_writer.write(combined)
            
    if args.output_video:
        video_writer.release()
        
    # Copy evaluation results if they exist
    import shutil
    eval_dir = f"demo/bppc/{handed_option}/{model_name}/eval/{folder_number}"
    if os.path.exists(eval_dir):
        for txt_file in ['body_results.txt', 'conf_results.txt']:
            src_file = os.path.join(eval_dir, txt_file)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(out_dir, txt_file))
                
    print(f"Visualization successfully generated in: {out_dir}")

if __name__ == "__main__":
    main()
