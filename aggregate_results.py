import os
import glob
import pandas as pd

base_dir = '/home/i2slab2/oshmos/bppc/demo/output'
body_files = glob.glob(os.path.join(base_dir, 'basic_results', '*', '*', '*_results.txt'))
conf_files = glob.glob(os.path.join(base_dir, 'conf_results', '*', '*', '*_results.txt'))

print(f"Found {len(body_files)} basic result files and {len(conf_files)} conf result files")

body_headers = ['model', 'head', 'sho', 'elb', 'wri', 'hip', 'knee', 'ank', 'avg']
conf_headers = ['model', 'under 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', 'over 0.9']

def parse_file(filepath, headers):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if len(lines) < 4:
            return []
        
        # Line 3 (index 2) - original backbone
        val3 = lines[2].strip().split()
        nums3 = []
        for x in val3:
            try:
                nums3.append(float(x))
            except:
                pass
        model3 = " ".join(val3[:-len(nums3)])
        
        # Line 4 (index 3) - backbone + bppc
        val4 = lines[3].strip().split()
        nums4 = []
        for x in val4:
            try:
                nums4.append(float(x))
            except:
                pass
        model4 = " ".join(val4[:-len(nums4)])
        
        return [
            [model3] + nums3,
            [model4] + nums4
        ]
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

all_body_data = []
for f in body_files:
    # f is like: /home/i2slab2/oshmos/bppc/demo/output/basic_results/res152/left/1.0_0.5_5.0_0.01_results.txt
    parts = f.split('/')
    handed = parts[-2]
    model_name = parts[-3]
    parsed = parse_file(f, body_headers)
    for row in parsed:
        all_body_data.append([handed] + row)

all_conf_data = []
for f in conf_files:
    parts = f.split('/')
    handed = parts[-2]
    model_name = parts[-3]
    parsed = parse_file(f, conf_headers)
    for row in parsed:
        all_conf_data.append([handed] + row)

# If no data found yet, exit gracefully
if len(all_body_data) == 0 and len(all_conf_data) == 0:
    print("No valid data found yet. Please make sure eval_bppc.py generated results.")
    exit(0)

df_body = pd.DataFrame(all_body_data, columns=['handed'] + body_headers)
df_conf = pd.DataFrame(all_conf_data, columns=['handed'] + conf_headers)

df_body = df_body.sort_values(by=['handed', 'model'])
df_conf = df_conf.sort_values(by=['handed', 'model'])

output_path = '/home/i2slab2/oshmos/bppc/eval_bppc_results.xlsx'

with pd.ExcelWriter(output_path) as writer:
    if not df_body.empty:
        df_body.to_excel(writer, sheet_name='Body Results', index=False)
    if not df_conf.empty:
        df_conf.to_excel(writer, sheet_name='Conf Results', index=False)

print(f"Results successfully saved to {output_path}")
