import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter

# =========================================================
# 1. FFmpegの設定 (ここが重要)
# =========================================================
# 指定されたパスを設定
FFMPEG_PATH = r"C:\Users\yuich\python_project\project_analysis_main_research\data\etc\ffmpeg\bin\ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

# =========================================================
# 2. 処理対象の設定
# =========================================================
TARGET_GROUP = "CIPN"      # グループ (NOCIPN, CIPN, STUDENT)
TARGET_SUBJECT = "P003"    # 被験者ID
TARGET_TASK = "ROMBERG"    # タスク (ROMBERG, ONELEG)
TARGET_COND = "EO"         # 条件 (EO, EC, L, R)
TARGET_TRIAL = "T1"        # 試行 (T1, T2, T3)

# 動画の設定
FPS = 60                   # 動画のフレームレート
VIDEO_DURATION_SEC = 10    # 作成する秒数 (None なら全データ)
FRAME_START = 200          # 解析開始フレーム

# パス設定
BASE_ROOT = r"C:\Users\yuich\python_project\project_analysis_main_research"
INPUT_ROOT = os.path.join(BASE_ROOT, r"data\2_time_series_feature\main_research\CoG")
OUTPUT_VIDEO_DIR = os.path.join(BASE_ROOT, r"data\4_videos")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# ファイル名パターン
pattern = re.compile(
    r"detected_(STUDENT|CIPN|NOCIPN)-(ROMBERG|ONELEG)-(P\d{3})-\d{8}-(EO|EC|L|R)-(T[123])-(C1|C2)_trim_CoG\.csv",
    re.IGNORECASE
)

# =========================================================
# 3. 関数定義
# =========================================================
def apply_sg_filter(df, win=21, poly=3):
    """SGフィルター適用"""
    df = df.copy()
    # 欠損補間してから配列化
    vals = df["CoG_X_mm"].interpolate().to_numpy()
    return savgol_filter(vals, window_length=win, polyorder=poly)

def find_files(group, subject, task, cond, trial):
    """C1(X)とC2(Y)のファイルペアを探す"""
    search_dir = os.path.join(INPUT_ROOT, group, subject, task)
    c1_path, c2_path = None, None
    
    # C1 (X座標用)
    for f in glob.glob(os.path.join(search_dir, "C1", "*.csv")):
        m = pattern.search(os.path.basename(f))
        if m and m.groups()[2] == subject and m.groups()[3] == cond and m.groups()[4] == trial:
            c1_path = f
            break
            
    # C2 (Y座標用)
    for f in glob.glob(os.path.join(search_dir, "C2", "*.csv")):
        m = pattern.search(os.path.basename(f))
        if m and m.groups()[2] == subject and m.groups()[3] == cond and m.groups()[4] == trial:
            c2_path = f
            break
            
    return c1_path, c2_path

# =========================================================
# 4. メイン処理
# =========================================================
print(f"Target: {TARGET_SUBJECT} - {TARGET_TASK} - {TARGET_COND}")
file_c1, file_c2 = find_files(TARGET_GROUP, TARGET_SUBJECT, TARGET_TASK, TARGET_COND, TARGET_TRIAL)

if not file_c1 or not file_c2:
    print(f"Error: Files not found.\nC1: {file_c1}\nC2: {file_c2}")
    exit()

# データ読み込み
print("Loading data...")
try:
    # C1ファイルの "CoG_X_mm" 列を X座標として使用
    data_x = apply_sg_filter(pd.read_csv(file_c1))
    # C2ファイルの "CoG_X_mm" 列を Y座標として使用 (※ファイル構造に依存する既存ロジック)
    data_y = apply_sg_filter(pd.read_csv(file_c2))
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# データ長を揃えてカット
n_len = min(len(data_x), len(data_y))
data_x = data_x[FRAME_START:n_len]
data_y = data_y[FRAME_START:n_len]

# 指定秒数のみにトリミング
if VIDEO_DURATION_SEC:
    max_frames = int(FPS * VIDEO_DURATION_SEC)
    data_x = data_x[:max_frames]
    data_y = data_y[:max_frames]

print(f"Frames to render: {len(data_x)}")

# --- アニメーション設定 ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlabel("Medial-Lateral (mm)")
ax.set_ylabel("Anterior-Posterior (mm)")
title_str = f"CoG Trajectory\n{TARGET_SUBJECT} ({TARGET_COND})"
ax.set_title(title_str)

# 軸の固定（動きが見やすいようにマージンを確保）
margin = 10
ax.set_xlim(np.min(data_x) - margin, np.max(data_x) + margin)
ax.set_ylim(np.min(data_y) - margin, np.max(data_y) + margin)

# プロット要素
line, = ax.plot([], [], '-', color='royalblue', lw=1, alpha=0.6, label='Path')
point, = ax.plot([], [], 'o', color='tomato', ms=10, markeredgecolor='white', label='CoG')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    point.set_data([], [])
    time_text.set_text('')
    return line, point, time_text

def update(frame):
    # 軌跡（現在まで）
    line.set_data(data_x[:frame], data_y[:frame])
    # 現在点
    point.set_data([data_x[frame]], [data_y[frame]])
    # 時間表示
    time_text.set_text(f'{frame/FPS:.2f} s')
    return line, point, time_text

# 動画生成
print("Generating MP4...")
ani = FuncAnimation(fig, update, frames=len(data_x), init_func=init, blit=True, interval=1000/FPS)

# 保存
save_filename = f"Video_{TARGET_SUBJECT}_{TARGET_TASK}_{TARGET_COND}_{TARGET_TRIAL}.mp4"
save_path = os.path.join(OUTPUT_VIDEO_DIR, save_filename)

try:
    # writer='ffmpeg' を明示的に指定
    ani.save(save_path, writer='ffmpeg', fps=FPS, dpi=150)
    print(f"\nSUCCESS: Video saved to:\n{save_path}")
except Exception as e:
    print(f"\nERROR: {e}")
    print("Check if the FFMPEG_PATH is correct.")

plt.close()