import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定・定数定義
# ==========================================

# === パス設定 ===
# 基本パス（環境に合わせて変更してください）
base_path = "C:/Users/yuich/python_project/project_analysis_main_research"

# 入力・出力フォルダ
input_root = os.path.join(base_path, "data", "1_processed", "3D_Result")
output_root = os.path.join(base_path, r"data\2_time_series_feature\main_research/CoB_3D") # 日付は今日にしています
target_condition = "NOCIPN"
# 解析対象の設定
target_subject_id = None  # 特定のIDだけやるなら "Subject01" など指定
target_tasks = ["4MWALK", "TUG"]  # 処理したいタスク名の一部

# ディレクトリ作成
input_dir = os.path.join(input_root, target_condition)
output_dir = os.path.join(output_root, target_condition)
os.makedirs(output_dir, exist_ok=True)

# === グラフ設定 ===
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# === 関節定義 (3次元対応) ===
JOINTS = {
    "nose": {"x": "nose_X", "y": "nose_Y", "z": "nose_Z"},
    "left_eye": {"x": "left_eye_X", "y": "left_eye_Y", "z": "left_eye_Z"},
    "right_eye": {"x": "right_eye_X", "y": "right_eye_Y", "z": "right_eye_Z"},
    "left_ear": {"x": "left_ear_X", "y": "left_ear_Y", "z": "left_ear_Z"},
    "right_ear": {"x": "right_ear_X", "y": "right_ear_Y", "z": "right_ear_Z"},
    "left_shoulder": {"x": "left_shoulder_X", "y": "left_shoulder_Y", "z": "left_shoulder_Z"},
    "right_shoulder": {"x": "right_shoulder_X", "y": "right_shoulder_Y", "z": "right_shoulder_Z"},
    "left_elbow": {"x": "left_elbow_X", "y": "left_elbow_Y", "z": "left_elbow_Z"},
    "right_elbow": {"x": "right_elbow_X", "y": "right_elbow_Y", "z": "right_elbow_Z"},
    "left_wrist": {"x": "left_wrist_X", "y": "left_wrist_Y", "z": "left_wrist_Z"},
    "right_wrist": {"x": "right_wrist_X", "y": "right_wrist_Y", "z": "right_wrist_Z"},
    "left_hip": {"x": "left_hip_X", "y": "left_hip_Y", "z": "left_hip_Z"},
    "right_hip": {"x": "right_hip_X", "y": "right_hip_Y", "z": "right_hip_Z"},
    "left_knee": {"x": "left_knee_X", "y": "left_knee_Y", "z": "left_knee_Z"},
    "right_knee": {"x": "right_knee_X", "y": "right_knee_Y", "z": "right_knee_Z"},
    "left_ankle": {"x": "left_ankle_X", "y": "left_ankle_Y", "z": "left_ankle_Z"},
    "right_ankle": {"x": "right_ankle_X", "y": "right_ankle_Y", "z": "right_ankle_Z"},
}

# === 身体セグメント質量比 (Winter / Dempster) ===
BODY_SEGMENTS_MASS_RATIO = {
    "head": 0.081, "trunk": 0.497,
    "right_upper_arm": 0.028, "left_upper_arm": 0.028,
    "right_forearm": 0.016, "left_forearm": 0.016,
    "right_hand": 0.006, "left_hand": 0.006,
    "right_thigh": 0.100, "left_thigh": 0.100,
    "right_shin": 0.0465, "left_shin": 0.0465,
    "right_foot": 0.0145, "left_foot": 0.0145,
}

# === 重心位置の比率 (Proximal Ratio: 近位端からの距離率) ===
# 例: 大腿は股関節から43.3%の位置に重心がある
COM_PROXIMAL_RATIOS = {
    "upper_arm": 0.436,
    "forearm": 0.430,
    "thigh": 0.433,
    "shin": 0.433,
}

# === セグメント定義 ===
# A. 比率計算を行うペア (四肢)
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

# B. 幾何学的中心を使用する部位 (頭・体幹・手足)
SEGMENT_OTHERS = {
    "head": ("nose", "left_ear", "right_ear"),
    "trunk": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "left_hand": ("left_wrist",),
    "right_hand": ("right_wrist",),
    "left_foot": ("left_ankle",),
    "right_foot": ("right_ankle",),
}


# ==========================================
# 2. ファイル収集とフィルタリング
# ==========================================

print(f"検索ディレクトリ: {input_dir}")
all_csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
csv_files = []

for f in all_csv_files:
    filename = os.path.basename(f)
    
    # 1. タスクフィルタ (4MWALK または ROMBERG が含まれているか)
    is_target_task = any(task in filename for task in target_tasks)
    
    # 2. 被験者IDフィルタ
    if target_subject_id:
        is_target_subject = target_subject_id in filename
    else:
        is_target_subject = True

    if is_target_task and is_target_subject:
        csv_files.append(f)

if not csv_files:
    print(f"対象のCSVファイル（タスク: {target_tasks}）が見つかりませんでした。")
    exit()
else:
    print(f"{len(csv_files)} 件のファイルを処理対象として検出しました。")


# ==========================================
# 3. メイン処理ループ
# ==========================================

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

    # 必須カラムチェック (左肩Xがあるか)
    if "left_shoulder_X" not in df.columns:
        print(f"  ⚠ 必要なカラムが含まれていないためスキップ")
        continue

    # 欠損・0値フレーム除去 (主要な関節が欠損しているフレームは削除)
    required_cols = ["left_shoulder_X", "right_shoulder_X", "left_hip_X", "right_hip_X"]
    df = df.replace(0, np.nan).dropna(subset=required_cols)
    
    if df.empty:
        print(f"  ⚠ 有効なフレームがありません")
        continue

    # --- 重心計算初期化 ---
    df["CoG_X"] = 0.0
    df["CoG_Y"] = 0.0
    df["CoG_Z"] = 0.0
    
    # A. 四肢 (Proximal Ratioを使用)
    for segment, (prox_name, dist_name) in SEGMENT_PAIRS.items():
        # 両方の関節が存在するかチェック
        if (JOINTS[prox_name]["x"] not in df.columns or 
            JOINTS[dist_name]["x"] not in df.columns):
            continue
        
        # 比率の取得
        ratio_key = segment.replace("left_", "").replace("right_", "")
        ratio = COM_PROXIMAL_RATIOS.get(ratio_key, 0.5)

        # 近位端 (Proximal)
        prox_x = df[JOINTS[prox_name]["x"]]
        prox_y = df[JOINTS[prox_name]["y"]]
        prox_z = df[JOINTS[prox_name]["z"]] if JOINTS[prox_name]["z"] in df.columns else 0

        # 遠位端 (Distal)
        dist_x = df[JOINTS[dist_name]["x"]]
        dist_y = df[JOINTS[dist_name]["y"]]
        dist_z = df[JOINTS[dist_name]["z"]] if JOINTS[dist_name]["z"] in df.columns else 0

        # 重心位置 = Prox + (Dist - Prox) * Ratio
        cx = prox_x + (dist_x - prox_x) * ratio
        cy = prox_y + (dist_y - prox_y) * ratio
        cz = prox_z + (dist_z - prox_z) * ratio

        m = BODY_SEGMENTS_MASS_RATIO[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m
        df["CoG_Z"] += cz * m

    # B. その他 (単純平均を使用)
    for segment, landmarks in SEGMENT_OTHERS.items():
        valid = [name for name in landmarks if JOINTS[name]["x"] in df.columns]
        if not valid:
            continue

        if segment == "trunk":
            # 肩の中点
            s_cx = (df["left_shoulder_X"] + df["right_shoulder_X"]) / 2
            s_cy = (df["left_shoulder_Y"] + df["right_shoulder_Y"]) / 2
            s_cz = (df["left_shoulder_Z"] + df["right_shoulder_Z"]) / 2 if "left_shoulder_Z" in df.columns else 0

            # 腰の中点
            h_cx = (df["left_hip_X"] + df["right_hip_X"]) / 2
            h_cy = (df["left_hip_Y"] + df["right_hip_Y"]) / 2
            h_cz = (df["left_hip_Z"] + df["right_hip_Z"]) / 2 if "left_hip_Z" in df.columns else 0

            # その中間
            cx = (s_cx + h_cx) / 2
            cy = (s_cy + h_cy) / 2
            cz = (s_cz + h_cz) / 2
        else:
            # 各点の単純平均
            cx = sum(df[JOINTS[n]["x"]] for n in valid) / len(valid)
            cy = sum(df[JOINTS[n]["y"]] for n in valid) / len(valid)
            
            # Z軸の平均 (存在しない場合は0)
            cz_list = []
            for n in valid:
                col_z = JOINTS[n]["z"]
                if col_z in df.columns:
                    cz_list.append(df[col_z])
                else:
                    cz_list.append(0)
            cz = sum(cz_list) / len(valid)
        
        m = BODY_SEGMENTS_MASS_RATIO[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m
        df["CoG_Z"] += cz * m

    # --- 出力CSV ---
    out_csv = os.path.join(out_subdir, f"{base_name}_CoG.csv")
    df[["TIME", "CoG_X", "CoG_Y", "CoG_Z"]].to_csv(out_csv, index=False)
    print(f"  ✅ CSV保存: {out_csv}")

    # --- プロット (3段) ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"重心軌跡 (CoG) - {base_name}\nTask: {target_tasks}", fontsize=14)
    t = df["TIME"]

    # X軸 (Medio-Lateral / 左右)
    ax[0].plot(t, df["CoG_X"], color="r", label="X")
    ax[0].set_ylabel("X (raw)")
    ax[0].grid(True, linestyle=":")
    ax[0].legend(loc='upper right')

    # Y軸 (Vertical / 高さ ※座標系による)
    ax[1].plot(t, df["CoG_Y"], color="g", label="Y")
    ax[1].set_ylabel("Y (raw)")
    ax[1].grid(True, linestyle=":")
    ax[1].legend(loc='upper right')

    # Z軸 (Depth / 進行方向 ※座標系による)
    ax[2].plot(t, df["CoG_Z"], color="b", label="Z")
    ax[2].set_ylabel("Z (raw)")
    ax[2].set_xlabel("Time (ms)")
    ax[2].grid(True, linestyle=":")
    ax[2].legend(loc='upper right')

    out_plot = os.path.join(out_subdir, f"{base_name}_CoG.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_plot)
    plt.close(fig)
    print(f"  📈 グラフ保存: {out_plot}")

print("\n=== 全ファイルの処理が完了しました ===")