import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import label
import os
import re

# ======================================
# 設定
# ======================================
real_length_cm_upper_arm = 30.0

scaling_joint_start = ("left_shoulder_X", "left_shoulder_Y")
scaling_joint_end   = ("left_elbow_X",    "left_elbow_Y")

video_width_px  = 1920
video_height_px = 1080

stride_threshold_cm = 20          # 歩幅として採用する最小距離 [cm]
stop_velocity_threshold = 1.0     # cm/frame：停止とみなす速度
min_stop_distance_cm = 40         # 停止点間の移動がこれ未満なら「偽停止」として除外

# ======================================
# ディレクトリ
# ======================================
base_path = r"C:\Users\yuich\python_project\project_analysis_main_research"

# 解析対象グループのルート（ここを STUDENT / CIPN / NOCIPN で切り替え）
csv_root = os.path.join(base_path, r"data/1_processed/main_research/CIPN")

output_root = os.path.join(base_path, r"data/3_summary_feature/Walking_gait/CIPN")
os.makedirs(output_root, exist_ok=True)

# ======================================
# ① グループ配下の CSV をすべて探索
# ======================================
all_csv_files = glob.glob(os.path.join(csv_root, "**", "*.csv"), recursive=True)

# ======================================
# ② C1 の TUG と C1 の 4MWALK だけをフィルタ
# ======================================
csv_files = []
for f in all_csv_files:
    parts = os.path.normpath(f).split(os.sep)

    if "C1" in parts and ("TUG" in parts or "4MWALK" in parts):
        csv_files.append(f)

print(f"対象 CSV: {len(csv_files)} 個")

# ======================================
# 停止点検出（速度ベース）
# ======================================
def detect_stops_by_velocity(position_cm):
    """
    位置の時系列（cm）から、速度が閾値以下の区間を検出し、
    5フレーム以上続いた区間の中央フレームを「停止点」として返す。
    """
    smooth = savgol_filter(position_cm, window_length=7, polyorder=2)
    velocity = np.abs(np.gradient(smooth))

    stop_mask = velocity < stop_velocity_threshold

    label_map, num = label(stop_mask)

    stop_frames = []
    for i in range(1, num + 1):
        frames = np.where(label_map == i)[0]
        if len(frames) >= 5:  # 5フレーム以上持続した場合だけ採用
            stop_frames.append(frames[len(frames)//2])

    return smooth, stop_frames


def filter_stop_frames_by_distance(smooth_position, stop_frames, min_distance_cm=40):
    """
    stop_frames: detect_stops_by_velocity() が返した停止点の index リスト
    smooth_position: 速度フィルタ後の滑らかな足位置
    min_distance_cm: この距離未満なら偽停止点として除外
    """
    if len(stop_frames) < 2:
        return stop_frames  # 比較対象がないのでそのまま返す
    
    filtered = [stop_frames[0]]  # 最初のポイントは残す
    
    for i in range(1, len(stop_frames)):
        prev = filtered[-1]
        curr = stop_frames[i]
        
        dist = abs(smooth_position[curr] - smooth_position[prev])

        # 変化が min_distance_cm 未満なら “偽物” として除外
        if dist >= min_distance_cm:
            filtered.append(curr)

    return filtered


# ======================================
# 集計用リスト
# ======================================
summary_rows = []

# ======================================
# CSV処理ループ
# ======================================
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    parts = os.path.normpath(file_path).split(os.sep)

    # --------- グループ名抽出 ---------
    group = None
    for g in ["STUDENT", "CIPN", "NOCIPN"]:
        if g in parts:
            group = g
            break
    if group is None:
        group = "UNKNOWN"

    # --------- 被験者番号抽出 ---------
    m = re.search(r"P\d{3}", file_name)
    subject_id = m.group() if m else "UNKNOWN"

    # --------- タスク名抽出 ---------
    task = None
    for t in ["TUG", "4MWALK", "4M"]:
        if t in parts:
            task = t
            break

    # --------- カメラ抽出 ---------
    camera = None
    for c in ["C1", "C2"]:
        if c in parts:
            camera = c
            break

    print(f"\n--- 処理中: {file_name} ---")
    print(f"  被験者: {subject_id}, グループ: {group}, タスク: {task}, カメラ: {camera}")

    df = pd.read_csv(file_path)

    # 必要なカラムチェック
    required_cols = {
        "left_shoulder_X", "left_shoulder_Y",
        "left_elbow_X", "left_elbow_Y",
        "left_ankle_X", "right_ankle_X"
    }
    if not required_cols.issubset(df.columns):
        print("  必要なカラムが不足しています。スキップします。")
        continue

    # ======================================
    # 全関節が検出されているフレームのみに制限
    # ======================================
    joint_cols = [c for c in df.columns if c.endswith("_X") or c.endswith("_Y")]

    # 0 → NaN （欠損として扱う）
    df[joint_cols] = df[joint_cols].replace(0, np.nan)

    # NaN を含むフレーム除外
    df = df.dropna(subset=joint_cols).reset_index(drop=True)

    if len(df) < 10:
        print("  ⚠ 有効フレームが少ないためスキップ")
        continue

    # ======================================
    # スケーリング（上腕長で cm 換算）
    # ======================================
    dx = (df[scaling_joint_end[0]] - df[scaling_joint_start[0]]) * video_width_px
    dy = (df[scaling_joint_end[1]] - df[scaling_joint_start[1]]) * video_height_px
    d_px = np.sqrt(dx**2 + dy**2)

    cm_per_px = real_length_cm_upper_arm / np.median(d_px)
    x_scale_cm = cm_per_px * video_width_px

    # ======================================
    # 足の X 位置（cm）
    # ======================================
    left_ankle_cm  = (df["left_ankle_X"]  - df["left_ankle_X"].min())  * x_scale_cm
    right_ankle_cm = (df["right_ankle_X"] - df["right_ankle_X"].min()) * x_scale_cm

    # ======================================
    # 停止点検出 + 距離フィルタ
    # ======================================
    # --- タスク別 min_distance_cm 設定 ---
    if task in ["4MWALK", "4M"]:
        task_min_distance = 90
    else:
        task_min_distance = 30

    # --- 右足 ---
    right_smooth, right_stops = detect_stops_by_velocity(right_ankle_cm)
    right_stops = filter_stop_frames_by_distance(
        right_smooth, right_stops, min_distance_cm=task_min_distance
    )

    # --- 左足 ---
    left_smooth, left_stops = detect_stops_by_velocity(left_ankle_cm)
    left_stops = filter_stop_frames_by_distance(
        left_smooth, left_stops, min_distance_cm=task_min_distance
    )

    # ======================================
    # 歩幅計算（右足）
    # ======================================
    right_stride_list = []
    for i in range(1, len(right_stops)):
        start = right_stops[i - 1]
        end   = right_stops[i]
        stride = abs(right_smooth[end] - right_smooth[start])
        if stride >= stride_threshold_cm:
            right_stride_list.append({
                "start_frame": start,
                "end_frame": end,
                "stride_cm": stride
            })

    # ======================================
    # 歩幅計算（左足）
    # ======================================
    left_stride_list = []
    for i in range(1, len(left_stops)):
        start = left_stops[i - 1]
        end   = left_stops[i]
        stride = abs(left_smooth[end] - left_smooth[start])
        if stride >= stride_threshold_cm:
            left_stride_list.append({
                "start_frame": start,
                "end_frame": end,
                "stride_cm": stride
            })

    # ======================================
    # 結果表示（ターミナル）
    # ======================================
    right_steps = len(right_stride_list)
    left_steps  = len(left_stride_list)

    print("  【右足歩幅】")
    for s in right_stride_list:
        print(f"    Frame {s['start_frame']} → {s['end_frame']}：{s['stride_cm']:.2f} cm")
    if right_steps:
        print(f"    右平均: {np.mean([s['stride_cm'] for s in right_stride_list]):.2f} cm")

    print("  【左足歩幅】")
    for s in left_stride_list:
        print(f"    Frame {s['start_frame']} → {s['end_frame']}：{s['stride_cm']:.2f} cm")
    if left_steps:
        print(f"    左平均: {np.mean([s['stride_cm'] for s in left_stride_list]):.2f} cm")

    print(f"  → 歩数（右）: {right_steps} 歩")
    print(f"  → 歩数（左）: {left_steps} 歩")

    # ======================================
    # 指標の集計（歩数・平均歩幅など）
    # ======================================
    right_mean = np.mean([s["stride_cm"] for s in right_stride_list]) if right_steps else np.nan
    left_mean  = np.mean([s["stride_cm"] for s in left_stride_list])  if left_steps  else np.nan

    right_max = np.max([s["stride_cm"] for s in right_stride_list]) if right_steps else np.nan
    left_max  = np.max([s["stride_cm"] for s in left_stride_list])  if left_steps  else np.nan

    stride_ratio = (right_mean / left_mean) if (right_steps and left_steps and left_mean > 0) else np.nan
    mean_stride  = np.nanmean([right_mean, left_mean])

    summary_rows.append({
        "subject": subject_id,
        "group": group,
        "task": task,
        "camera": camera,
        "file_name": file_name,
        "right_num_steps": right_steps,
        "left_num_steps": left_steps,
        "right_mean_stride_cm": right_mean,
        "left_mean_stride_cm": left_mean,
        "right_max_stride_cm": right_max,
        "left_max_stride_cm": left_max,
        "stride_ratio_R_over_L": stride_ratio,
        "mean_stride_cm": mean_stride,
    })

    # ======================================
    # グラフ描画
    # ======================================
    plt.figure(figsize=(12, 5))
    plt.plot(right_smooth, label="Right Ankle X (cm)", color="blue")
    plt.plot(left_smooth,  label="Left Ankle X (cm)",  color="red")

    # 右足
    for s in right_stride_list:
        plt.plot(s['start_frame'], right_smooth[s['start_frame']], 'o', color='blue')
        plt.plot(s['end_frame'],   right_smooth[s['end_frame']],   'o', color='blue')
        plt.plot([s['start_frame'], s['end_frame']],
                 [right_smooth[s['start_frame']], right_smooth[s['end_frame']]],
                 linestyle='--', color='blue', alpha=0.7)

    # 左足
    for s in left_stride_list:
        plt.plot(s['start_frame'], left_smooth[s['start_frame']], 'o', color='red')
        plt.plot(s['end_frame'],   left_smooth[s['end_frame']],   'o', color='red')
        plt.plot([s['start_frame'], s['end_frame']],
                 [left_smooth[s['start_frame']], left_smooth[s['end_frame']]],
                 linestyle='--', color='red', alpha=0.7)

    plt.xlabel("Frame")
    plt.ylabel("Ankle X Position (cm)")
    plt.title(f"Stride Detection (velocity-based): {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_root, file_name.replace(".csv", "_stride_plot.png"))
    plt.savefig(save_path)
    plt.close()

# ======================================
# 全ファイルの集計結果を CSV に保存
# ======================================
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_root, "stride_summary_CIPN_C1_TUG_4MWALK.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n=== Summary saved ===")
    print(summary_path)
else:
    print("\n=== 有効なデータがなかったため、Summary CSV は作成されませんでした ===")

print("\n=== All processing complete! ===")
