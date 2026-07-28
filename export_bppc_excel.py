import os
import glob
import numpy as np
import pandas as pd

# List of all 11 backbones
BACKBONES = [
    "res50", "res101", "res152",
    "hw32", "hw48", "darkw32", "darkw48",
    "vitpose_b", "vitpose_l", "dwpose", "rtmpose"
]

HANDED_OPTIONS = ["left", "right"]

def evaluate_npz_files(handed, model_name):
    path = f"demo/bppc/{handed}/{model_name}"
    files = sorted(glob.glob(f"{path}/*_results.npz"))
    if not files:
        return None, None
    
    # We will compute keypoint distances normalized by diagonal/box
    base_accs = []
    bppc_accs = []
    
    # Standard PCK evaluation matching bppc accuracy
    for f in files:
        data = np.load(f)
        bppc_k = data['bppc_kpts'] # (N, 13, 2)
        base_k = data['base_kpts'] # (N, 13, 2)
        gt_k   = data['gt_kpts']   # (N, 13, 2)
        
        # W=640, H=480 standard frame size in PA
        w, h = 640.0, 480.0
        norm = np.array([w, h]) / 10.0
        
        dist_base = np.linalg.norm((base_k - gt_k) / norm, axis=-1)
        dist_bppc = np.linalg.norm((bppc_k - gt_k) / norm, axis=-1)
        
        # PCK @ 0.5
        acc_b = np.mean(dist_base <= 0.5, axis=0) # (13,)
        acc_p = np.mean(dist_bppc <= 0.5, axis=0) # (13,)
        
        base_accs.append(acc_b)
        bppc_accs.append(acc_p)
        
    base_avg = np.mean(base_accs, axis=0) # (13,)
    bppc_avg = np.mean(bppc_accs, axis=0) # (13,)
    
    # 13 joints: 0:head, 1..2:sho, 3..4:elb, 5..6:wri, 7..8:hip, 9..10:knee, 11..12:ank
    def group_body(arr):
        head = arr[0]
        sho  = np.mean(arr[1:3])
        elb  = np.mean(arr[3:5])
        wri  = np.mean(arr[5:7])
        hip  = np.mean(arr[7:9])
        knee = np.mean(arr[9:11])
        ank  = np.mean(arr[11:13])
        avg  = np.mean(arr)
        return [head, sho, elb, wri, hip, knee, ank, avg]

    return group_body(base_avg), group_body(bppc_avg)

def parse_txt_file(txt_path):
    if not os.path.exists(txt_path):
        return None, None
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Find table lines
    table_lines = [line for line in lines if not line.startswith('-') and not line.startswith('model')]
    if len(table_lines) >= 2:
        parts_base = table_lines[0].split()
        parts_bppc = table_lines[1].split()
        
        vals_base = [float(x) for x in parts_base[1:]]
        vals_bppc = [float(x) for x in parts_bppc[1:]]
        return vals_base, vals_bppc
    return None, None

def build_data(handed):
    rows = []
    cols = ["Model", "Method", "Head", "Sho", "Elb", "Wri", "Hip", "Knee", "Ank", "Avg Accuracy"]
    
    for m in BACKBONES:
        txt_files = glob.glob(f"demo/output/basic_results/{m}/{handed}/*.txt")
        vals_base, vals_bppc = None, None
        
        if txt_files:
            vals_base, vals_bppc = parse_txt_file(txt_files[0])
            
        if vals_base is None or m in ["vitpose_b", "vitpose_l", "dwpose", "rtmpose"]:
            v_b, v_p = evaluate_npz_files(handed, m)
            if v_b is not None:
                vals_base, vals_bppc = v_b, v_p
                
        if vals_base is not None:
            rows.append([m, "Baseline", *[round(x * 100, 2) for x in vals_base]])
            rows.append([f"{m}+BPPC", "+BPPC", *[round(x * 100, 2) for x in vals_bppc]])
            
    df = pd.DataFrame(rows, columns=cols)
    return df

def main():
    os.chdir("/home/i2slab0/oshmos/bppc")
    df_left  = build_data("left")
    df_right = build_data("right")
    
    # Calculate Overall Average (47 left + 118 right = 165 total)
    df_avg = df_left.copy()
    num_cols = ["Head", "Sho", "Elb", "Wri", "Hip", "Knee", "Ank", "Avg Accuracy"]
    
    for c in num_cols:
        df_avg[c] = np.round((df_left[c] * 47 + df_right[c] * 118) / 165.0, 2)
        
    excel_path = "eval_bppc_results_all.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_left.to_excel(writer, sheet_name="Left Batter (좌타자)", index=False)
        df_right.to_excel(writer, sheet_name="Right Batter (우타자)", index=False)
        df_avg.to_excel(writer, sheet_name="Overall Average (전체)", index=False)
        
    # Also overwrite eval_bppc_results.xlsx
    with pd.ExcelWriter("eval_bppc_results.xlsx", engine='openpyxl') as writer:
        df_left.to_excel(writer, sheet_name="Left Batter (좌타자)", index=False)
        df_right.to_excel(writer, sheet_name="Right Batter (우타자)", index=False)
        df_avg.to_excel(writer, sheet_name="Overall Average (전체)", index=False)

    print(f"Successfully generated {excel_path} and eval_bppc_results.xlsx!")
    print("\n=== Overall Average Table ===")
    print(df_avg.to_string(index=False))

if __name__ == "__main__":
    main()
