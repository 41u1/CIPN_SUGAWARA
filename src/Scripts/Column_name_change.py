# VITの列名をMPに一致させるためのコード

import pandas as pd
import glob
import os

# === 設定 ===
input_dir =r"C:\Users\yuich\python_project\project_analysis\data\1_processed\csv\ViTPose\hospital\3" # 処理したいフォルダを指定
output_dir = input_dir  # 出力フォルダ（同じにしたいなら input_dir に変更）

os.makedirs(output_dir, exist_ok=True)

# 列名変換マッピング
rename_dict = {
    "Nose_X": "nose_X", "Nose_Y": "nose_Y",
    "L_Eye_X": "left_eye_X", "L_Eye_Y": "left_eye_Y",
    "R_Eye_X": "right_eye_X", "R_Eye_Y": "right_eye_Y",
    "L_Ear_X": "left_ear_X", "L_Ear_Y": "left_ear_Y",
    "R_Ear_X": "right_ear_X", "R_Ear_Y": "right_ear_Y",
    "L_Shoulder_X": "left_shoulder_X", "L_Shoulder_Y": "left_shoulder_Y",
    "R_Shoulder_X": "right_shoulder_X", "R_Shoulder_Y": "right_shoulder_Y",
    "L_Elbow_X": "left_elbow_X", "L_Elbow_Y": "left_elbow_Y",
    "R_Elbow_X": "right_elbow_X", "R_Elbow_Y": "right_elbow_Y",
    "L_Wrist_X": "left_wrist_X", "L_Wrist_Y": "left_wrist_Y",
    "R_Wrist_X": "right_wrist_X", "R_Wrist_Y": "right_wrist_Y",
    "L_Hip_X": "left_hip_X", "L_Hip_Y": "left_hip_Y",
    "R_Hip_X": "right_hip_X", "R_Hip_Y": "right_hip_Y",
    "L_Knee_X": "left_knee_X", "L_Knee_Y": "left_knee_Y",
    "R_Knee_X": "right_knee_X", "R_Knee_Y": "right_knee_Y",
    "L_Ankle_X": "left_ankle_X", "L_Ankle_Y": "left_ankle_Y",
    "R_Ankle_X": "right_ankle_X", "R_Ankle_Y": "right_ankle_Y",
}

print("📂 検出されたCSVファイル:")
print(glob.glob(os.path.join(input_dir, "*.csv")))

# === すべてのCSVファイルを処理 ===
for file_path in glob.glob(os.path.join(input_dir, "*.csv")):
    try:
        df = pd.read_csv(file_path)
        df.rename(columns=rename_dict, inplace=True)

        output_path = os.path.join(output_dir, os.path.basename(file_path))
        df.to_csv(output_path, index=False)
        print(f"✅ {os.path.basename(file_path)} → 保存完了")

    except Exception as e:
        print(f"⚠️ {os.path.basename(file_path)} の処理中にエラー: {e}")
