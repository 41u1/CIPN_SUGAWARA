import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# =========================================================
# 1. 設定 (SETTINGS)
# =========================================================
BASE_ROOT = r"C:\Users\yuich\python_project\project_analysis_main_research"

# 分散の集計CSVがあるフォルダ
VARIANCE_SUMMARY_ROOT = os.path.join(BASE_ROOT, r"data/3_summary_feature/ROMBERG_Variance/summary")

# 結果出力先
OUTPUT_DIR = os.path.join(BASE_ROOT, r"daily_results/20251205/Crawford_Test_Variance_SeparateCam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ▼▼▼ 解析対象の指定 ▼▼▼
TARGET_ID    = "P002"   # 解析したい患者ID
TARGET_GROUP = "CIPN"   # その患者が属するグループ

# ▼▼▼ 比較対象（健常群）の指定 ▼▼▼
CONTROL_GROUP = "STUDENT" 

# =========================================================
# 2. 関数定義
# =========================================================
def load_variance_csv():
    """ 分散の集計CSV (ROMBERG_AllVariance_summary.csv) を読み込む """
    csv_path = os.path.join(VARIANCE_SUMMARY_ROOT, "ROMBERG_AllVariance_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

def perform_crawford_test(case_val, control_vals):
    """ Crawford & Howell (1998) t-test """
    n = len(control_vals)
    mean_c = np.mean(control_vals)
    std_c = np.std(control_vals, ddof=1)
    
    if std_c == 0: return np.nan, np.nan, mean_c, std_c

    t_val = (case_val - mean_c) / (std_c * np.sqrt((n + 1) / n))
    df = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df))
    
    return t_val, p_val, mean_c, std_c

# =========================================================
# 3. データ読み込み
# =========================================================
print(f"--- Loading Variance Data ---")

df_all = load_variance_csv()
if df_all is None:
    raise FileNotFoundError("Variance summary CSV is missing.")

# =========================================================
# 4. カメラごとのループ処理 (C1, C2)
# =========================================================
CAMERAS = ["C1", "C2"]

for cam in CAMERAS:
    print(f"\n{'='*20} Processing Camera: {cam} {'='*20}")

    # -----------------------------------------------------
    # データの抽出 (このカメラのデータのみにする)
    # -----------------------------------------------------
    # コントロール群 (該当カメラ)
    df_control_cam = df_all[(df_all["Group"] == CONTROL_GROUP) & (df_all["camera"] == cam)].copy()
    
    # ターゲット患者 (該当カメラ)
    target_row = df_all[(df_all["Group"] == TARGET_GROUP) & 
                        (df_all["subject"] == TARGET_ID) & 
                        (df_all["camera"] == cam)]

    # データチェック
    if len(df_control_cam) < 3:
        print(f"⚠ Skipping {cam}: Not enough control data (N={len(df_control_cam)})")
        continue
    
    if len(target_row) == 0:
        print(f"⚠ Skipping {cam}: Target subject {TARGET_ID} has no data for {cam}")
        continue

    # 患者データをSeriesとして取得
    patient_data = target_row.iloc[0]
    print(f"  Target: {TARGET_ID} ({cam}), Control: {CONTROL_GROUP} (N={len(df_control_cam)})")

    # -----------------------------------------------------
    # 統計解析 (Crawford Test)
    # -----------------------------------------------------
    metrics_list = [
        "Local_EO", "Local_EC", "Local_RR",
        "Global_EO", "Global_EC", "Global_RR",
        "Total_EO", "Total_EC", "Total_RR"
    ]
    summary_results = []

    for metric in metrics_list:
        if metric not in patient_data: continue

        val_p = patient_data[metric]
        vals_h = df_control_cam[metric].dropna()

        # コントロール群にターゲット本人が混入している場合（同一グループ比較時）は除外
        if TARGET_GROUP == CONTROL_GROUP:
            vals_h = vals_h[df_control_cam["subject"] != TARGET_ID]

        t_stat, p_val, mean_h, std_h = perform_crawford_test(val_p, vals_h)

        summary_results.append({
            "Camera": cam,
            "Metric": metric,
            "Patient_ID": TARGET_ID,
            "Patient_Value": val_p,
            "Control_Group": CONTROL_GROUP,
            "Control_N": len(vals_h),
            "Control_Mean": mean_h,
            "Control_SD": std_h,
            "Crawford_t": t_stat,
            "p_value": p_val,
            "Significant": "YES" if p_val < 0.05 else "no"
        })

    # CSV保存 (カメラごとに別ファイル)
    df_summary = pd.DataFrame(summary_results)
    csv_out_path = os.path.join(OUTPUT_DIR, f"Stats_{TARGET_ID}_vs_{CONTROL_GROUP}_{cam}.csv")
    df_summary.to_csv(csv_out_path, index=False)
    print(f"  Saved CSV: {os.path.basename(csv_out_path)}")

    # -----------------------------------------------------
    # 図の作成 (Visualization)
    # -----------------------------------------------------
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    sns.set_theme(style="ticks")

    plot_categories = {
        "Local_Variance":  ["Local_EO", "Local_EC", "Local_RR"],
        "Global_Variance": ["Global_EO", "Global_EC", "Global_RR"],
        "Total_Variance":  ["Total_EO", "Total_EC", "Total_RR"]
    }

    for cat_name, metrics in plot_categories.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(f"{cat_name} Analysis - Camera {cam}", fontsize=16)
        
        legend_handles, legend_labels = [], []

        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # データ再取得
            val_p = patient_data[metric]
            vals_h = df_control_cam[metric].dropna()
            if TARGET_GROUP == CONTROL_GROUP:
                vals_h = vals_h[df_control_cam["subject"] != TARGET_ID]

            # 検定
            t_stat, p_val, _, _ = perform_crawford_test(val_p, vals_h)

            # --- Plotting ---
            sns.boxplot(y=vals_h, width=0.35, ax=ax, 
                        boxprops=dict(facecolor="#ADD8E6", edgecolor="black"), showfliers=False)
            
            sns.stripplot(y=vals_h, ax=ax, color="royalblue", size=8, alpha=0.7, jitter=0.1, 
                          label=f"{CONTROL_GROUP} (N={len(vals_h)})")
            
            ax.scatter(0, val_p, color="red", marker="o", s=180, zorder=10, 
                       edgecolors="black", linewidth=1.5, label=f"Patient ({TARGET_ID})")

            # --- Design ---
            label_clean = metric.split("_")[1] 
            ax.set_title(label_clean, fontsize=14, fontweight="bold", pad=12)
            
            if "RR" in metric: ylabel = "Romberg Ratio"
            else: ylabel = "Variance (mm²)"
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel("")
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')

            # --- Stats Text ---
            df_dof = len(vals_h) - 1
            if not np.isnan(p_val):
                p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
                sig_mark = "*" if p_val < 0.05 else ""
                stats_text = f"t({df_dof})={t_stat:.2f}\np{p_str}{sig_mark}"
            else:
                stats_text = "N/A"

            ax.text(0.95, 0.96, stats_text, transform=ax.transAxes, fontsize=11, ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9))

            # 凡例収集
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                order = [f"Patient ({TARGET_ID})", f"{CONTROL_GROUP} (N={len(vals_h)})"]
                legend_handles = [by_label[l] for l in order if l in by_label]
                legend_labels = [l for l in order if l in by_label]
            
            if ax.get_legend(): ax.get_legend().remove()

        # 共通凡例
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                   bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True, edgecolor="black", fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)
        
        # ファイル名に _C1, _C2 を付与
        img_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID}_vs_{CONTROL_GROUP}_{cat_name}_{cam}.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved Plot: {os.path.basename(img_path)}")

print("\n=== Separate Camera Analysis Complete ===")