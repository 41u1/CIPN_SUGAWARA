import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 0. Configuration (設定)
# ==============================================================================

# ★重要: グループフォルダ(STUDENT, CIPNなど)が格納されている親フォルダを指定してください
INPUT_ROOT = r"C:\Users\Kei15\CIPN\CIPN_SUGAWARA\data\1_processed\main_research"

# 出力先
OUTPUT_ROOT = r"C:\Users\Kei15\CIPN\CIPN_SUGAWARA\data/2_time_series_feature/main_research/AoJ"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ------------------------------------------------------------------
# ▼ フィルタリング設定 (空リスト [] なら「全て」実行)
# ------------------------------------------------------------------

# 解析したいグループを指定 (例: ["STUDENT"] や ["CIPN", "NOCIPN"])
TARGET_GROUPS = [] 

# 解析したい被験者を指定 (例: ["P001", "P002"])
TARGET_SUBJECTS = [] 

# ==============================================================================
# 1. Joint definitions (2D)
# ==============================================================================

JOINTS = {
    "left_shoulder":  ["left_shoulder_X", "left_shoulder_Y"],
    "right_shoulder": ["right_shoulder_X", "right_shoulder_Y"],
    "left_elbow":     ["left_elbow_X", "left_elbow_Y"],
    "right_elbow":    ["right_elbow_X", "right_elbow_Y"],
    "left_wrist":     ["left_wrist_X", "left_wrist_Y"],
    "right_wrist":    ["right_wrist_X", "right_wrist_Y"],
    "left_hip":       ["left_hip_X", "left_hip_Y"],
    "right_hip":      ["right_hip_X", "right_hip_Y"],
    "left_knee":      ["left_knee_X", "left_knee_Y"],
    "right_knee":     ["right_knee_X", "right_knee_Y"],
    "left_ankle":     ["left_ankle_X", "left_ankle_Y"],
    "right_ankle":    ["right_ankle_X", "right_ankle_Y"],
}

ANGLES_TO_CALCULATE = {
    # 下肢
    'left_knee':  {'p1': 'left_hip',  'center': 'left_knee',  'p2': 'left_ankle'},
    'right_knee': {'p1': 'right_hip', 'center': 'right_knee', 'p2': 'right_ankle'},
    'left_hip':   {'p1': 'left_shoulder', 'center': 'left_hip', 'p2': 'left_knee'},
    'right_hip':  {'p1': 'right_shoulder', 'center': 'right_hip', 'p2': 'right_knee'},
    # 上肢
    'left_elbow': {'p1': 'left_shoulder', 'center': 'left_elbow', 'p2': 'left_wrist'},
    'right_elbow':{'p1': 'right_shoulder', 'center': 'right_elbow', 'p2': 'right_wrist'},
    'left_shoulder': {'p1': 'left_elbow', 'center': 'left_shoulder', 'p2': 'left_hip'},
    'right_shoulder':{'p1': 'right_elbow', 'center': 'right_shoulder', 'p2': 'right_hip'},
}

# ==============================================================================
# 2. Vectorized Angle Calculation (高速化)
# ==============================================================================

def calculate_angle_2d_vectorized(df, p1_cols, center_cols, p2_cols):
    """ NumPyを用いた高速2D角度計算 """
    p1 = df[p1_cols].values
    center = df[center_cols].values
    p2 = df[p2_cols].values

    v1 = p1 - center
    v2 = p2 - center

    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    dot_product = np.einsum('ij,ij->i', v1, v2)

    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_theta))
    
    return angles

# ==============================================================================
# 3. Main Processing
# ==============================================================================

def main():
    print(f"\n📂 Root Data Folder: {INPUT_ROOT}")

    # 1. 親フォルダ内のグループフォルダを取得
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Input Root not found: {INPUT_ROOT}")
        return

    all_groups = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    # グループフィルタリング
    if TARGET_GROUPS:
        target_groups = [g for g in all_groups if g in TARGET_GROUPS]
    else:
        target_groups = all_groups

    print(f"🎯 Target Groups: {target_groups}")

    total_files_processed = 0

    for group_name in target_groups:
        group_path = os.path.join(INPUT_ROOT, group_name)
        
        # 2. グループ内の被験者フォルダを取得
        all_subjects = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        
        # 被験者フィルタリング
        if TARGET_SUBJECTS:
            target_subjects = [s for s in all_subjects if s in TARGET_SUBJECTS]
        else:
            target_subjects = all_subjects

        if not target_subjects:
            continue

        print(f"\n=== Processing Group: {group_name} ({len(target_subjects)} subjects) ===")

        for subject_name in tqdm(target_subjects, desc=f"Subjects in {group_name}"):
            subject_path = os.path.join(group_path, subject_name)
            
            # 3. CSV探索
            csv_files = glob.glob(os.path.join(subject_path, "**", "*.csv"), recursive=True)
            
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                    if "TIME" not in df.columns: continue

                    angle_cols = []
                    
                    # --- 角度計算 (高速版) ---
                    for angle_name, pts in ANGLES_TO_CALCULATE.items():
                        p1_cols = JOINTS[pts['p1']]
                        c_cols  = JOINTS[pts['center']]
                        p2_cols = JOINTS[pts['p2']]

                        if not all(col in df.columns for col in p1_cols + c_cols + p2_cols):
                            continue

                        col_name = f"{angle_name}_angle_2d"
                        df[col_name] = calculate_angle_2d_vectorized(df, p1_cols, c_cols, p2_cols)
                        angle_cols.append(col_name)

                    # --- 角速度計算 ---
                    dt = df["TIME"].diff() / 1000.0
                    vel_cols = []
                    for angle_col in angle_cols:
                        vel_col = angle_col.replace("angle_2d", "angvel_2d")
                        df[vel_col] = df[angle_col].diff() / dt
                        vel_cols.append(vel_col)

                    # --- 保存 (パス構造を維持) ---
                    # 例: INPUT/Group/Subject/Task/File.csv -> OUTPUT/Group/Subject/Task/File_2d.csv
                    
                    # Groupフォルダからの相対パスを取得 (Subject/Task/File.csv)
                    rel_path_from_group = os.path.relpath(csv_path, group_path)
                    
                    # 保存先パス構築
                    save_path = os.path.join(OUTPUT_ROOT, group_name, rel_path_from_group)
                    save_path = save_path.replace(".csv", "_angle_velocity_2d.csv")
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    out_cols = ["TIME"] + angle_cols + vel_cols
                    df[out_cols].to_csv(save_path, index=False)
                    total_files_processed += 1

                except Exception as e:
                    tqdm.write(f"❌ Error: {os.path.basename(csv_path)} - {e}")
                    continue

    print(f"\n✅ All Done! Processed {total_files_processed} files.")
    print(f"📂 Output saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()