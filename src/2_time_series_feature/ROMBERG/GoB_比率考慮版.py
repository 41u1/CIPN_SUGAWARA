import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === パス設定 ===
base_path = r"C:/Users/yuich/python_project/project_analysis_main_research"
input_root = os.path.join(base_path, "data", "1_processed", "main_research")
output_root = os.path.join(base_path, r"data\2_time_series_feature\main_research\CoG")
target_condition = "NOCIPN"

# 被験者IDの指定 
# 特定の被験者のみ処理したい場合はここにIDを記入（例: "P002"）
# ※ 全員分処理したい場合は None または "" (空文字) にしてください
target_subject_id = "" 


input_dir = os.path.join(input_root, target_condition)
output_dir = os.path.join(output_root, target_condition)
os.makedirs(output_dir, exist_ok=True)

# === スケール設定 ===
SCALE_MM_PER_PX = 1680 / 1080  # ≒ 1.556 mm/px

# === 日本語フォント設定 ===
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# === ViTPoseの関節定義 ===
JOINTS = {
    "nose": {"x": "nose_X", "y": "nose_Y"},
    "left_eye": {"x": "left_eye_X", "y": "left_eye_Y"},
    "right_eye": {"x": "right_eye_X", "y": "right_eye_Y"},
    "left_ear": {"x": "left_ear_X", "y": "left_ear_Y"},
    "right_ear": {"x": "right_ear_X", "y": "right_ear_Y"},
    "left_shoulder": {"x": "left_shoulder_X", "y": "left_shoulder_Y"},
    "right_shoulder": {"x": "right_shoulder_X", "y": "right_shoulder_Y"},
    "left_elbow": {"x": "left_elbow_X", "y": "left_elbow_Y"},
    "right_elbow": {"x": "right_elbow_X", "y": "right_elbow_Y"},
    "left_wrist": {"x": "left_wrist_X", "y": "left_wrist_Y"},
    "right_wrist": {"x": "right_wrist_X", "y": "right_wrist_Y"},
    "left_hip": {"x": "left_hip_X", "y": "left_hip_Y"},
    "right_hip": {"x": "right_hip_X", "y": "right_hip_Y"},
    "left_knee": {"x": "left_knee_X", "y": "left_knee_Y"},
    "right_knee": {"x": "right_knee_X", "y": "right_knee_Y"},
    "left_ankle": {"x": "left_ankle_X", "y": "left_ankle_Y"},
    "right_ankle": {"x": "right_ankle_X", "y": "right_ankle_Y"},
}

# === 1. 質量比 (Winter, 2009 / Dempster) ===
BODY_SEGMENTS_MASS_RATIO = {
    "head": 0.081, "trunk": 0.497,
    "right_upper_arm": 0.028, "left_upper_arm": 0.028,
    "right_forearm": 0.016, "left_forearm": 0.016,
    "right_hand": 0.006, "left_hand": 0.006,
    "right_thigh": 0.100, "left_thigh": 0.100,
    "right_shin": 0.0465, "left_shin": 0.0465,
    "right_foot": 0.0145, "left_foot": 0.0145,
}

# === 2. 重心位置の比率 (Proximal Ratio) ===
COM_PROXIMAL_RATIOS = {
    "upper_arm": 0.436,
    "forearm": 0.430,
    "thigh": 0.433,
    "shin": 0.433,
}

# === 3. セグメントの構造定義 ===
SEGMENT_PAIRS = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shin": ("left_knee", "left_ankle"),
    "right_shin": ("right_knee", "right_ankle"),
}

SEGMENT_OTHERS = {
    "head": ("nose", "left_ear", "right_ear"),
    "trunk": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "left_hand": ("left_wrist",),
    "right_hand": ("right_wrist",),
    "left_foot": ("left_ankle",),
    "right_foot": ("right_ankle",),
}

# === 処理開始 ===
all_csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)

# ▼▼▼【変更点】被験者IDでのフィルタリング ▼▼▼
if target_subject_id:
    # ファイルパスの中に target_subject_id (例: "P002") が含まれているものだけ抽出
    csv_files = [f for f in all_csv_files if target_subject_id in f]
    print(f"フィルタ設定: ID '{target_subject_id}' を含むファイルのみ処理します。")
else:
    csv_files = all_csv_files
    print("フィルタ設定: 全ファイルを処理します。")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

if not csv_files:
    print(f"対象のCSVファイルが '{input_dir}' に見つかりませんでした。")
    if target_subject_id:
        print(f"（ID '{target_subject_id}' に一致するファイルがありませんでした）")
else:
    print(f"{len(csv_files)} 件のCSVファイルを検出しました。")

for csv_path in csv_files:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    rel_dir = os.path.relpath(os.path.dirname(csv_path), input_dir)
    out_subdir = os.path.join(output_dir, rel_dir)
    os.makedirs(out_subdir, exist_ok=True)

    print(f"\n▶ {rel_dir}\\{base_name}.csv を処理中...")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ 読み込みエラー: {e}")
        continue

    if "left_shoulder_X" not in df.columns:
        print(f"  ⚠ ViTPose形式でないためスキップ")
        continue

    # 欠損・0値フレーム除去
    df = df.replace(0, np.nan).dropna(subset=["left_shoulder_X", "right_shoulder_X", "left_hip_X", "right_hip_X"])
    if df.empty:
        print(f"  ⚠ 有効なフレームがありません")
        continue

    # --- 重心計算 ---
    df["CoG_X"] = 0.0
    df["CoG_Y"] = 0.0
    
    # 1. 四肢
    for segment, (prox_name, dist_name) in SEGMENT_PAIRS.items():
        if JOINTS[prox_name]["x"] not in df.columns or JOINTS[dist_name]["x"] not in df.columns:
            continue
        
        ratio_key = segment.replace("left_", "").replace("right_", "")
        ratio = COM_PROXIMAL_RATIOS.get(ratio_key, 0.5)

        prox_x = df[JOINTS[prox_name]["x"]]
        prox_y = df[JOINTS[prox_name]["y"]]
        dist_x = df[JOINTS[dist_name]["x"]]
        dist_y = df[JOINTS[dist_name]["y"]]

        cx = prox_x + (dist_x - prox_x) * ratio
        cy = prox_y + (dist_y - prox_y) * ratio

        m = BODY_SEGMENTS_MASS_RATIO[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m

    # 2. その他
    for segment, landmarks in SEGMENT_OTHERS.items():
        valid = [name for name in landmarks if JOINTS[name]["x"] in df.columns]
        if not valid:
            continue

        if segment == "trunk":
            s_cx = (df["left_shoulder_X"] + df["right_shoulder_X"]) / 2
            h_cx = (df["left_hip_X"] + df["right_hip_X"]) / 2
            cx = (s_cx + h_cx) / 2
            
            s_cy = (df["left_shoulder_Y"] + df["right_shoulder_Y"]) / 2
            h_cy = (df["left_hip_Y"] + df["right_hip_Y"]) / 2
            cy = (s_cy + h_cy) / 2
        else:
            cx = sum(df[JOINTS[n]["x"]] for n in valid) / len(valid)
            cy = sum(df[JOINTS[n]["y"]] for n in valid) / len(valid)
        
        m = BODY_SEGMENTS_MASS_RATIO[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m

    # --- スケール変換（px → mm） ---
    df["CoG_X_mm"] = df["CoG_X"] * SCALE_MM_PER_PX
    df["CoG_Y_mm"] = df["CoG_Y"] * SCALE_MM_PER_PX

    # --- 出力CSV ---
    out_csv = os.path.join(out_subdir, f"{base_name}_CoG.csv")
    df[["TIME", "CoG_X_mm", "CoG_Y_mm"]].to_csv(out_csv, index=False)
    print(f"  ✅ 出力: {out_csv}")

    # --- プロット ---
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"重心(CoG)軌跡 [mm換算] - {base_name}", fontsize=16)
    t = df["TIME"]

    ax[0].plot(t, df["CoG_X_mm"], color="r", label="X軸 (左右)")
    ax[0].set_ylabel("X [mm]"); ax[0].grid(True); ax[0].legend()

    ax[1].plot(t, df["CoG_Y_mm"], color="g", label="Y軸 (上下)")
    ax[1].set_ylabel("Y [mm]"); ax[1].set_xlabel("時間 (ms)")
    ax[1].grid(True); ax[1].legend()

    out_plot = os.path.join(out_subdir, f"{base_name}_CoG.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_plot)
    plt.close(fig)
    print(f"  📈 グラフ保存: {out_plot}")

print("\n=== 指定ファイルの処理が完了しました ===")