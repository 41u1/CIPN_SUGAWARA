import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ---------------------------------------------------------
# 1. Base Root
# ---------------------------------------------------------
BASE_ROOT = r"C:\Users\yuich\python_project\project_analysis_main_research"

# CoG の位置（あなたの構造）
INPUT_ROOT = os.path.join(BASE_ROOT, r"data\2_time_series_feature\main_research\CoG")
OUTPUT_ROOT = os.path.join(BASE_ROOT, r"data\3_summary_feature\ROMBERG_ratio")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------------------------------------
# 2. 定義
# ---------------------------------------------------------
GROUPS = ["NOCIPN"]
TASKS = ["ROMBERG"]

ROMBERG_CONDS = ["EO", "EC"]
ONELEG_CONDS = ["L", "R"]
TRIALS = ["T1", "T2", "T3"]

FRAME_START = 200
FRAME_END = 1700

# ファイル名パターン
pattern = re.compile(
    r"detected_(STUDENT|CIPN|NOCIPN)-(ROMBERG|ONELEG)-(P\d{3})-\d{8}-(EO|EC|L|R)-(T[123])-(C1|C2)_trim_CoG\.csv",
    re.IGNORECASE
)

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def compute_path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def compute_convex_area(x, y):
    try:
        hull = ConvexHull(np.column_stack((x, y)))
        return hull.volume
    except:
        return np.nan

# ---------------------------------------------------------
# 3. メイン処理
# ---------------------------------------------------------
for group in GROUPS:

    group_dir = os.path.join(INPUT_ROOT, group)
    if not os.path.exists(group_dir):
        print(f"\n✖ Missing group folder: {group_dir}")
        continue

    subjects = sorted([d for d in os.listdir(group_dir) if d.startswith("P")])

    print(f"\n==============================")
    print(f"▶ GROUP = {group}")
    print("==============================")

    for sid in subjects:

        print(f"\n--- Subject {sid} ---")
        subject_dir = os.path.join(group_dir, sid)

        for task in TASKS:

            task_dir = os.path.join(subject_dir, task)
            if not os.path.exists(task_dir):
                print(f"  ✖ Missing task folder: {task_dir}")
                continue

            print(f"  ▶ TASK = {task}")

            c1_dir = os.path.join(task_dir, "C1")
            c2_dir = os.path.join(task_dir, "C2")

            if not (os.path.exists(c1_dir) and os.path.exists(c2_dir)):
                print("    ✖ Missing C1 or C2 folder")
                continue

            files_c1 = glob.glob(os.path.join(c1_dir, "*.csv"))
            files_c2 = glob.glob(os.path.join(c2_dir, "*.csv"))

            index_c1 = {}
            index_c2 = {}

            # C1
            for f in files_c1:
                m = pattern.search(os.path.basename(f))
                if m:
                    g, t, pid, cond, trial, cam = m.groups()
                    if t == task:
                        index_c1[(pid, cond, trial)] = f

            # C2
            for f in files_c2:
                m = pattern.search(os.path.basename(f))
                if m:
                    g, t, pid, cond, trial, cam = m.groups()
                    if t == task:
                        index_c2[(pid, cond, trial)] = f

            # プロット準備
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            fig.suptitle(f"{group} - {sid} - {task}", fontsize=16)

            all_x, all_y = [], []

            CONDITIONS = ROMBERG_CONDS if task == "ROMBERG" else ONELEG_CONDS
            metrics = {c: {"path": [], "area": []} for c in CONDITIONS}

            # 各条件 × 試行
            for i, cond in enumerate(CONDITIONS):
                for j, trial in enumerate(TRIALS):

                    ax = axes[i, j]
                    key = (sid, cond, trial)

                    f1 = index_c1.get(key)
                    f2 = index_c2.get(key)

                    if not f1 or not f2:
                        ax.set_visible(False)
                        continue

                    df1 = pd.read_csv(f1)
                    df2 = pd.read_csv(f2)

                    n = min(len(df1), len(df2))
                    if n < FRAME_START:
                        ax.set_visible(False)
                        continue

                    x = df1["CoG_X_mm"].iloc[FRAME_START:n].to_numpy()
                    y = df2["CoG_X_mm"].iloc[FRAME_START:n].to_numpy()

                    all_x.extend(x)
                    all_y.extend(y)

                    path_len = compute_path_length(x, y)
                    area = compute_convex_area(x, y)

                    metrics[cond]["path"].append(path_len)
                    metrics[cond]["area"].append(area)

                    # plot
                    ax.plot(x, y, lw=1.4, color="royalblue")

                    # convex hull
                    try:
                        hull = ConvexHull(np.column_stack((x, y)))
                        pts = np.column_stack((x, y))[hull.vertices]
                        ax.fill(pts[:, 0], pts[:, 1], color="salmon", alpha=0.25)
                        ax.plot(pts[:, 0], pts[:, 1], color="tomato", lw=1.2)
                    except:
                        pass

                    ax.text(
                        0.98, 0.92,
                        f"{path_len:.1f} mm\n{area:.1f} mm²",
                        ha="right", va="top",
                        transform=ax.transAxes,
                        fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6)
                    )

                    ax.set_title(f"{cond}-{trial}")
                    ax.grid(True)

            # 軸揃え
            if len(all_x) > 0:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)

                for row in axes:
                    for ax in row:
                        if ax.get_visible():
                            ax.set_xlim(xmin, xmax)
                            ax.set_ylim(ymin, ymax)
                            ax.set_aspect("equal", "box")

            # 保存
            plot_dir = os.path.join(OUTPUT_ROOT, group, task, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            out_path = os.path.join(plot_dir, f"{sid}_{task}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close(fig)

            # summary（条件別平均を保存）
            summary_dir = os.path.join(OUTPUT_ROOT, group, task, "summary")
            os.makedirs(summary_dir, exist_ok=True)

            row = {"subject": sid}
            for c in CONDITIONS:
                row[f"{c}_path"] = np.nanmean(metrics[c]["path"])
                row[f"{c}_area"] = np.nanmean(metrics[c]["area"])

            summary_path = os.path.join(summary_dir, f"{group}_{task}_summary.csv")

            # 追記 or 新規作成
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])

            df.to_csv(summary_path, index=False)

print("\n=== All Processing Complete ===")
