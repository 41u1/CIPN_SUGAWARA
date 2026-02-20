import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # ★追加

# =========================================
# PATH
# =========================================
BASE = r"C:/Users/yuich/python_project/project_analysis_main_research"

INPUT_ROOT  = os.path.join(BASE, r"data/2_time_series_feature/main_research/CoG")
OUTPUT_ROOT = os.path.join(BASE, r"daily_results\20260210\ROMBERG_Variance")

OUT_TS      = os.path.join(OUTPUT_ROOT, "ts_plots")
OUT_SUMMARY = os.path.join(OUTPUT_ROOT, "summary")

os.makedirs(OUT_TS, exist_ok=True)
os.makedirs(OUT_SUMMARY, exist_ok=True)

# =========================================
# SETTINGS
# =========================================
# ▼▼▼ 被験者IDの指定 ▼▼▼
target_subject_id = ""
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

GROUPS = ["CIPN","NOCIPN","STUDENT"]
CONDS  = ["EO", "EC"]
TRIALS = ["T1", "T2", "T3"]

FRAME_START, FRAME_END = 0, 1800
WINDOW = 60
FPS = 60
COG_COL = "CoG_X_mm"

# ★追加：SGフィルターの設定
SG_WIN = 61   # window length (奇数である必要があります)
SG_POLY = 3   # polyorder

# =========================================
# ファイル名完全一致 pattern
# =========================================
pattern = re.compile(
    r"detected_(CIPN|STUDENT|NOCIPN)-ROMBERG-(P\d{3})-(\d{8})-(EO|EC)-(T[123])-(C1|C2)_trim_CoG\.csv",
    re.IGNORECASE
)

# =========================================
# FUNCTIONS
# =========================================

# ★追加：SGフィルター適用関数
def apply_sg_filter(df, win=SG_WIN, poly=SG_POLY):
    """
    df : 読み込んだCSV（CoG_X_mm, CoG_Y_mm を含む）
    win: window length（必ず奇数）
    """
    df = df.copy()

    # 欠損値があっても補間して処理
    # (CoG_Y_mm もついでに処理していますが、今回の解析で使うのは X だけなら X だけでも可)
    if "CoG_X_mm" in df.columns:
        x = df["CoG_X_mm"].interpolate().to_numpy()
        df["CoG_X_mm_sg"] = savgol_filter(x, window_length=win, polyorder=poly)
    
    if "CoG_Y_mm" in df.columns:
        y = df["CoG_Y_mm"].interpolate().to_numpy()
        df["CoG_Y_mm_sg"] = savgol_filter(y, window_length=win, polyorder=poly)

    return df


def rolling_variance(series, win):
    local_var  = series.rolling(win, center=True).var(ddof=0)
    mu         = series.mean()
    global_var = ((series - mu)**2).rolling(win, center=True).mean()
    return local_var, global_var


def scan_subjects():
    subject_paths = []
    subject_ids   = []
    subject_groups = []

    for group in GROUPS:
        group_dir = os.path.join(INPUT_ROOT, group)
        if not os.path.exists(group_dir):
            continue

        for d in sorted(os.listdir(group_dir)):
            if re.fullmatch(r"P\d{3}", d):
                subject_paths.append(os.path.join(group_dir, d))
                subject_ids.append(d)
                subject_groups.append(group)

    return subject_paths, subject_ids, subject_groups


def index_romberg_files(subject_dir, sid):
    idx = {}
    rom_dir = os.path.join(subject_dir, "ROMBERG")

    files = []
    for cam in ["C1", "C2"]:
        cam_dir = os.path.join(rom_dir, cam)
        if os.path.exists(cam_dir):
            files += glob.glob(os.path.join(cam_dir, "*.csv"))

    print(f"\n--- File scan for {sid} ---")

    for f in sorted(files):
        base = os.path.basename(f)
        m = pattern.match(base)
        if m:
            group, pid, date, cond, trial, cam = m.groups()
            key = (pid, cond, trial, cam)
            idx[key] = f
            print(f"  ✓ Recognized: {base} → key={key}")
        else:
            print(f"  - Skip (pattern mismatch): {base}")

    return idx


# ============================================================
# 6面プロット共通関数
# ============================================================

def plot_6panel(data_dict, sid, cam, label, ylabel, outname, group):
    subject_out_dir = os.path.join(OUT_TS, group , sid)
    os.makedirs(subject_out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle(f"{sid} [{group}]  ROMBERG - {label} ({cam}) [Filtered SG]") # タイトルにFilteredと追記

    all_vals = []
    eo_vals = []
    ec_vals = []

    for (cond, trial), item in data_dict.items():
        i = 0 if cond == "EO" else 1
        j = TRIALS.index(trial)
        ax = axes[i][j]

        if isinstance(item, tuple):
            t, vals = item
            ax.plot(t, vals, lw=1.4)
            mean_val = np.nanmean(vals)
            max_val = np.nanmax(vals)
            all_vals.append(max_val)
            ax.text(
                0.98, 0.95,
                f"{label}_mean={mean_val:.2f}\n{label}_max={max_val:.2f}",
                ha="right", va="top",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8)
            )
        else:
            mean_val = item
            max_val  = item
            ax.text(0.5, 0.5, f"{ylabel} = {mean_val:.2f}",
                    ha="center", va="center", fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7))

        ax.set_title(f"{cond}-{trial}")
        ax.grid(alpha=0.3)

        all_vals.append(max_val)
        if cond == "EO": eo_vals.append(mean_val)
        else: ec_vals.append(mean_val)

    if len(all_vals) > 0:
        ymax = max(all_vals)
        for row in axes:
            for ax in row:
                if ax.get_visible():
                    ax.set_ylim(0, ymax * 1.05)

    eo_mean = np.nanmean(eo_vals)
    ec_mean = np.nanmean(ec_vals)
    ratio = ec_mean / eo_mean if eo_mean > 0 else np.nan

    fig.text(0.5, 0.01, f"Romberg ratio (EC / EO) = {ratio:.3f}",
             ha="center", fontsize=12)

    save_path = os.path.join(subject_out_dir, f"{sid}_{group}_ROMBERG_{cam}_{outname}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

    return eo_mean, ec_mean, ratio


# ============================================================
# Subject processing
# ============================================================

def process_subject(subject_dir, sid, group,
                    summary_local, summary_global, summary_total):

    idx = index_romberg_files(subject_dir, sid)
    cameras = ["C1", "C2"]

    for cam in cameras:
        files_for_cam = [k for k in idx.keys() if k[3] == cam]
        if not files_for_cam:
            continue
        
        unique_key = f"{group}_{sid}_{cam}"

        # -------- LOCAL --------
        local_dict = {}
        for cond in CONDS:
            for trial in TRIALS:
                key = (sid, cond, trial, cam)
                if key not in idx: continue
                
                # ★変更：読み込み後にフィルター適用
                df = pd.read_csv(idx[key])
                df = apply_sg_filter(df, win=SG_WIN, poly=SG_POLY)
                
                # ★変更：フィルター済みカラム(_sg)を使用
                target_col = f"{COG_COL}_sg" 
                
                s = df[target_col].iloc[FRAME_START:FRAME_END].reset_index(drop=True)
                v_local, _ = rolling_variance(s, WINDOW)
                t = np.arange(len(s)) / FPS
                local_dict[(cond, trial)] = (t, v_local)

        if local_dict:
            eo_L, ec_L, rr_L = plot_6panel(local_dict, sid, cam, "Local Variance", "LocalVar", "LocalVariance", group)
            summary_local[unique_key] = {"Group": group,"EO": eo_L, "EC": ec_L, "RR": rr_L}

        # -------- GLOBAL --------
        global_dict = {}
        for cond in CONDS:
            for trial in TRIALS:
                key = (sid, cond, trial, cam)
                if key not in idx: continue

                df = pd.read_csv(idx[key])
                df = apply_sg_filter(df, win=SG_WIN, poly=SG_POLY) # ★フィルター適用
                target_col = f"{COG_COL}_sg" # ★フィルター済みカラム使用

                s = df[target_col].iloc[FRAME_START:FRAME_END].reset_index(drop=True)
                _, v_global = rolling_variance(s, WINDOW)
                t = np.arange(len(s)) / FPS
                global_dict[(cond, trial)] = (t, v_global)

        if global_dict:
            eo_G, ec_G, rr_G = plot_6panel(global_dict, sid, cam, "Global Variance", "GlobalVar", "GlobalVariance", group)
            summary_global[unique_key] = {"EO": eo_G, "EC": ec_G, "RR": rr_G}

        # -------- TOTAL --------
        total_dict = {}
        for cond in CONDS:
            for trial in TRIALS:
                key = (sid, cond, trial, cam)
                if key not in idx: continue

                df = pd.read_csv(idx[key])
                df = apply_sg_filter(df, win=SG_WIN, poly=SG_POLY) # ★フィルター適用
                target_col = f"{COG_COL}_sg" # ★フィルター済みカラム使用

                s = df[target_col].iloc[FRAME_START:FRAME_END].reset_index(drop=True)
                total_var = float(np.nanvar(s))
                total_dict[(cond, trial)] = total_var

        if total_dict:
            eo_T, ec_T, rr_T = plot_6panel(total_dict, sid, cam, "Total Variance", "TotalVar", "TotalVariance", group)
            summary_total[unique_key] = {"EO": eo_T, "EC": ec_T, "RR": rr_T}


# =========================================
# EXECUTION
# =========================================

subject_paths, subject_ids, subject_groups = scan_subjects()

# フィルタリング
if target_subject_id:
    indices = [i for i, x in enumerate(subject_ids) if x == target_subject_id]
    if not indices:
        print(f"WARNING: Subject ID '{target_subject_id}' not found.")
        subject_paths, subject_ids, subject_groups = [], [], []
    else:
        subject_paths  = [subject_paths[i] for i in indices]
        subject_ids    = [subject_ids[i] for i in indices]
        subject_groups = [subject_groups[i] for i in indices]
        print(f"FILTER APPLIED: Processing only {target_subject_id}")
else:
    print(f"FILTER NONE: Processing all {len(subject_ids)} subjects")


summary_local  = {}
summary_global = {}
summary_total  = {}

# 解析実行
for subdir, sid ,group in zip(subject_paths, subject_ids, subject_groups):
    print(f"\n=== Processing {sid} ===")
    process_subject(subdir, sid, group, summary_local, summary_global, summary_total)


# ============================================================
# CSVへの追記・更新ロジック (Group対応)
# ============================================================
if summary_local: 
    # 1. 今回の処理結果をDataFrame化
    df_local  = pd.DataFrame(summary_local).T.add_prefix("Local_")
    df_global = pd.DataFrame(summary_global).T.add_prefix("Global_")
    df_total  = pd.DataFrame(summary_total).T.add_prefix("Total_")
    
    df_new = pd.concat([df_local, df_global, df_total], axis=1)
    df_new.index.name = "Key_ID"
    df_new.reset_index(inplace=True)

    # キーを分解: "Group_Subject_Camera"
    df_new["Group"]   = df_new["Key_ID"].apply(lambda x: x.split("_")[0])
    df_new["subject"] = df_new["Key_ID"].apply(lambda x: x.split("_")[1])
    df_new["camera"]  = df_new["Key_ID"].apply(lambda x: x.split("_")[2])
    
    # フィルタ用の「ユニークキー」列を作成 (Group + Subject)
    df_new["_Unique_Filter_Key"] = df_new["Group"] + "_" + df_new["subject"]

    # カラム整理
    meta_cols = ["Group", "subject", "camera"]
    data_cols = [c for c in df_new.columns if c not in meta_cols + ["Key_ID", "_Unique_Filter_Key"]]
    df_new = df_new[meta_cols + ["_Unique_Filter_Key"] + data_cols]

    # 2. マスターCSVファイルのパス
    master_csv_path = os.path.join(OUT_SUMMARY, "ROMBERG_AllVariance_summary.csv")

    # 3. 既存ファイルがあるか確認
    if os.path.exists(master_csv_path):
        print(f"\n--- Updating existing CSV: {master_csv_path} ---")
        try:
            df_master = pd.read_csv(master_csv_path)
            
            if "Group" not in df_master.columns:
                 print("  ⚠ Existing CSV has no 'Group' column. Creating new file to avoid collision.")
                 df_final = df_new
            else:
                df_master["_Unique_Filter_Key"] = df_master["Group"].astype(str) + "_" + df_master["subject"].astype(str)
                
                processed_unique_keys = df_new["_Unique_Filter_Key"].unique()
                
                before_len = len(df_master)
                df_master = df_master[~df_master["_Unique_Filter_Key"].isin(processed_unique_keys)]
                after_len = len(df_master)
                
                if before_len != after_len:
                    print(f"  Existing data for {len(processed_unique_keys)} subjects (Group+ID) removed/updated.")
                
                df_final = pd.concat([df_master, df_new], ignore_index=True)

        except Exception as e:
            print(f"  ⚠ Error reading existing CSV. Creating new one. ({e})")
            df_final = df_new
    else:
        print(f"\n--- Creating new CSV: {master_csv_path} ---")
        df_final = df_new

    # 4. ソートして保存
    df_final = df_final.sort_values(by=["Group", "subject", "camera"])
    
    if "_Unique_Filter_Key" in df_final.columns:
        df_final = df_final.drop(columns=["_Unique_Filter_Key"])

    df_final.to_csv(master_csv_path, index=False)
    print(f"✅ CSV Saved successfully.")

else:
    print("\nNo data processed.")