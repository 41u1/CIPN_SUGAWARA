import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === パス設定 ===
base_path = r"C:\Users\yuich\python_project\project_analysis_main_research"
csv_root = os.path.join(base_path, r"data/1_processed/main_research/NOCIPN/P002/ROMBERG")
output_root = os.path.join(base_path, r"data\2_time_series_feature\main_research\CoG\NOCIPN\P002\ROMBERG")
os.makedirs(output_root, exist_ok=True)

# === スケール設定（C2撮影条件: 1080px = 168cm = 1680mm） ===
SCALE_MM_PER_PX = 1680 / 1080  # ≒ 1.556 mm/px

# === 日本語フォント設定 ===
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# === ViTPoseの関節対応 ===
JOINTS = {
    "nose": {"x": "nose_X", "y": "nose_Y"},
    "left_eye": {"x": "left_eye_X", "y": "left_eye_Y"},
    "left_ear": {"x": "left_ear_X", "y": "left_ear_Y"},
    "left_shoulder": {"x": "left_shoulder_X", "y": "left_shoulder_Y"},
    "left_elbow": {"x": "left_elbow_X", "y": "left_elbow_Y"},
    "left_wrist": {"x": "left_wrist_X", "y": "left_wrist_Y"},
    "left_hip": {"x": "left_hip_X", "y": "left_hip_Y"},
    "left_knee": {"x": "left_knee_X", "y": "left_knee_Y"},
    "left_ankle": {"x": "left_ankle_X", "y": "left_ankle_Y"},
}

# === 1. 質量比（半身モデル: Winter/Dempster参考） ===
# 左半身だけの計算のため、全身比率の半分を使用（頭や体幹も半分とする）
BODY_SEGMENTS_MASS_RATIO_LEFT = {
    "head": 0.081 / 2,
    "trunk": 0.497 / 2,
    "left_upper_arm": 0.028,
    "left_forearm": 0.016,
    "left_hand": 0.006,
    "left_thigh": 0.100,
    "left_shin": 0.0465,
    "left_foot": 0.0145,
}

# === 2. 重心位置の比率 (Proximal Ratio) ===
# Winter Table 4.1 "Center of Mass / Segment Length (Proximal)"
# 近位関節（体幹に近い方）から何割の位置に重心があるか
COM_PROXIMAL_RATIOS = {
    "upper_arm": 0.436,  # 肩から43.6%
    "forearm": 0.430,    # 肘から43.0%
    "thigh": 0.433,      # 腰から43.3%
    "shin": 0.433,       # 膝から43.3%
}

# === 3. セグメントの構造定義（左側） ===
# A. 比率を使って計算する部位: (近位関節, 遠位関節)
SEGMENT_PAIRS_LEFT = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_shin": ("left_knee", "left_ankle"),
}

# B. そのまま座標を使う、または単純平均をとる部位
SEGMENT_OTHERS_LEFT = {
    "head": ("left_ear",),          # 側面視では耳を頭部中心と仮定
    "trunk": ("left_shoulder", "left_hip"), # 体幹は肩と腰の中点とする（簡易モデル）
    "left_hand": ("left_wrist",),   # 手首で代用
    "left_foot": ("left_ankle",),   # 足首で代用
}


# =====================================================================
# メイン処理
# =====================================================================

# ① CSV探索
all_csv_files = glob.glob(os.path.join(csv_root, "**", "*.csv"), recursive=True)

# ② C1 のみをフィルタ (ファイルパスにC1が含まれるか)
csv_files = [
    f for f in all_csv_files
    if "C1" in os.path.normpath(f).split(os.sep)
]

if not csv_files:
    print("⚠ C1 の CSV が見つかりませんでした。")
else:
    print(f"🔍 {len(csv_files)} 件の C1 CSV を検出しました。")

for csv_path in csv_files:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    rel_dir = os.path.relpath(os.path.dirname(csv_path), csv_root)
    out_subdir = os.path.join(output_root, rel_dir)
    os.makedirs(out_subdir, exist_ok=True)

    print(f"\n▶ {rel_dir}\\{base_name}.csv を処理中...")

    df = pd.read_csv(csv_path)

    # 必須カラムチェック
    if "left_shoulder_X" not in df.columns:
        print(f"  ⚠ ViTPose形式でないためスキップします。")
        continue

    # --- 欠損除去 ---
    # 計算に必要な主要関節がない行は削除
    required_joints = ["left_shoulder_X", "left_hip_X", "left_knee_X", "left_ankle_X"]
    # 存在しないカラムがあればスキップ
    if not all(col in df.columns for col in required_joints):
        print("  ⚠ 必須関節カラムが不足しています。スキップ。")
        continue

    df = df.replace(0, np.nan).dropna(subset=required_joints)
    if df.empty:
        print(f"  ⚠ 有効なデータがありません。スキップ。")
        continue

    # --- 全質量（分母用） ---
    total_mass = sum(BODY_SEGMENTS_MASS_RATIO_LEFT.values())

    # --- 重心計算 ---
    df["CoG_X"] = 0.0
    df["CoG_Y"] = 0.0

    # 1. 四肢 (Proximal Ratio を使用して計算)
    for segment, (prox_name, dist_name) in SEGMENT_PAIRS_LEFT.items():
        # キーポイントが存在するか確認
        if JOINTS[prox_name]["x"] not in df.columns or JOINTS[dist_name]["x"] not in df.columns:
            continue
        
        # 比率のキーを取得 (left_thigh -> thigh)
        ratio_key = segment.replace("left_", "").replace("right_", "")
        ratio = COM_PROXIMAL_RATIOS.get(ratio_key, 0.5)

        # 座標取得
        prox_x = df[JOINTS[prox_name]["x"]]
        prox_y = df[JOINTS[prox_name]["y"]]
        dist_x = df[JOINTS[dist_name]["x"]]
        dist_y = df[JOINTS[dist_name]["y"]]

        # 重心 = 近位 + (遠位 - 近位) * 比率
        cx = prox_x + (dist_x - prox_x) * ratio
        cy = prox_y + (dist_y - prox_y) * ratio

        # 加算 (位置 * 質量)
        m = BODY_SEGMENTS_MASS_RATIO_LEFT[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m

    # 2. その他 (Head, Trunk, Hand, Foot - 平均または単一点)
    for segment, landmarks in SEGMENT_OTHERS_LEFT.items():
        valid = [n for n in landmarks if JOINTS[n]["x"] in df.columns]
        if not valid:
            continue

        # 平均座標を計算 (Trunkの場合は肩と腰の中点になる)
        cx = sum(df[JOINTS[n]["x"]] for n in valid) / len(valid)
        cy = sum(df[JOINTS[n]["y"]] for n in valid) / len(valid)
        
        m = BODY_SEGMENTS_MASS_RATIO_LEFT[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m

    # 最後に総質量で割る
    df["CoG_X"] /= total_mass
    df["CoG_Y"] /= total_mass

    # --- スケール変換 (mm単位) ---
    df["CoG_X_mm"] = df["CoG_X"] * SCALE_MM_PER_PX
    df["CoG_Y_mm"] = df["CoG_Y"] * SCALE_MM_PER_PX

    # --- 出力CSV ---
    output_csv_path = os.path.join(out_subdir, f"{base_name}_CoG.csv")
    df[["TIME", "CoG_X_mm", "CoG_Y_mm"]].to_csv(output_csv_path, index=False)
    print(f"  ✅ 出力: {output_csv_path}")

    # --- プロット ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"重心推移 (左半身モデル/Winter比率) - {base_name}", fontsize=15)
    t = df["TIME"]

    axes[0].plot(t, df["CoG_X_mm"], color="r", label="X軸（前後方向, mm）")
    axes[0].set_ylabel("X [mm]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(t, df["CoG_Y_mm"], color="g", label="Y軸（上下方向, mm）")
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_xlabel("時間 [ms]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)
    axes[1].invert_yaxis()  # 画像座標系(上が0)の場合、Y軸反転で見やすくする

    output_plot_path = os.path.join(out_subdir, f"{base_name}_CoG.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_plot_path)
    plt.close(fig)
    print(f"  📈 グラフ保存: {output_plot_path}")

print("\n=== 全フォルダのCSV処理が完了しました (C2: Left-side, Proximal Ratio適用) ===")