import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# =========================================================
# 1. 設定 (SETTINGS)
# =========================================================
# ベースパス (先ほど出力したフォルダ構成に合わせます)
BASE_ROOT = r"C:\Users\yuich\python_project\project_analysis_main_research"
SUMMARY_ROOT = os.path.join(BASE_ROOT, r"data/3_summary_feature/ROMBERG_ratio")

# 出力先
OUTPUT_DIR = os.path.join(BASE_ROOT, r"daily_results/20251205/Crawford_Test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ▼▼▼ 解析対象の指定 ▼▼▼
TARGET_ID    = "P002"   # 解析したい患者ID
TARGET_GROUP = "CIPN"   # その患者が属するグループ (CSVの場所)

# ▼▼▼ 比較対象（健常群）の指定 ▼▼▼
CONTROL_GROUP = "STUDENT" # 比較対照にするグループ名 (例: STUDENT, NOCIPN)

# =========================================================
# 2. 関数定義
# =========================================================

def load_summary_csv(group_name):
    """ 指定されたグループの集計CSVを読み込む """
    # 前回のコードの保存パス構造: {GROUP}/ROMBERG/filtered_summary/{GROUP}_ROMBERG_summary.csv
    csv_path = os.path.join(
        SUMMARY_ROOT, group_name, "ROMBERG", "filtered_summary", f"{group_name}_ROMBERG_summary.csv"
    )
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def perform_crawford_test(case_val, control_vals):
    """ Crawford & Howell (1998) t-test for single case vs control group """
    n = len(control_vals)
    mean_c = np.mean(control_vals)
    std_c = np.std(control_vals, ddof=1)
    
    # 標準偏差が0（全員同じ値）の場合は計算不可
    if std_c == 0:
        return np.nan, np.nan, mean_c, std_c

    t_val = (case_val - mean_c) / (std_c * np.sqrt((n + 1) / n))
    df = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df)) # 両側検定
    
    return t_val, p_val, mean_c, std_c

# =========================================================
# 3. データ読み込み & 抽出
# =========================================================
print(f"--- Loading Data ---")

# 1. コントロール群 (健常群) の読み込み
df_control_all = load_summary_csv(CONTROL_GROUP)
if df_control_all is None:
    raise FileNotFoundError(f"Control group data not found for: {CONTROL_GROUP}")
else:
    print(f"Control Group ({CONTROL_GROUP}): N={len(df_control_all)}")

# 2. ターゲット患者の読み込み
df_target_all = load_summary_csv(TARGET_GROUP)
if df_target_all is None:
    raise FileNotFoundError(f"Target group data not found for: {TARGET_GROUP}")

# ターゲット行を抽出
target_row = df_target_all[df_target_all["subject"] == TARGET_ID]

if len(target_row) == 0:
    raise ValueError(f"Subject '{TARGET_ID}' not found in {TARGET_GROUP} summary CSV.")

# Series化 (1行だけ取得)
patient_data = target_row.iloc[0]
print(f"Target Subject found: {TARGET_ID} in {TARGET_GROUP}")

# =========================================================
# 4. 統計解析 & CSV出力
# =========================================================
# 解析する指標リスト (前回のCSVのカラム名と一致させる)
metrics_list = ["EO_path", "EO_area", "EC_path", "EC_area", "RR_path", "RR_area"]

summary_results = []

print(f"\n--- Performing Crawford's t-test ---")

for metric in metrics_list:
    # 患者の値 (スカラー)
    val_p = patient_data[metric]
    
    # コントロール群の値 (配列) - 欠損値は除去
    vals_h = df_control_all[metric].dropna()
    
    # もしコントロール群にターゲット本人が混ざっている場合は除外する (念のため)
    if TARGET_GROUP == CONTROL_GROUP:
        vals_h = df_control_all[df_control_all["subject"] != TARGET_ID][metric].dropna()

    # 検定実行
    t_stat, p_val, mean_h, std_h = perform_crawford_test(val_p, vals_h)
    
    # 結果格納
    summary_results.append({
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

# 結果をDataFrameにして保存
df_summary = pd.DataFrame(summary_results)
csv_out_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID}_vs_{CONTROL_GROUP}_stats.csv")
df_summary.to_csv(csv_out_path, index=False)

print(f"Statistical summary saved: {csv_out_path}")
print("-" * 60)
print(df_summary[["Metric", "Patient_Value", "Crawford_t", "p_value", "Significant"]].to_string(index=False))
print("-" * 60)

# =========================================================
# 5. 図の作成 (Visualization)
# =========================================================
plt.rcParams['font.family'] = 'Arial' # 日本語が必要なら 'Meiryo' に変更
plt.rcParams['font.size'] = 12
sns.set_theme(style="ticks")

# 作図設定
plot_groups = {
    "Path_Length": [("EO_path", "EO Path Length (mm)"), ("EC_path", "EC Path Length (mm)")],
    "Sway_Area":   [("EO_area", "EO Sway Area (mm²)"),   ("EC_area", "EC Sway Area (mm²)")],
    "Romberg_Ratio": [("RR_path", "Romberg Ratio (Path)"), ("RR_area", "Romberg Ratio (Area)")]
}

for group_name, metrics_in_group in plot_groups.items():
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    legend_handles, legend_labels = [], []

    for i, (metric, label) in enumerate(metrics_in_group):
        ax = axes[i]
        
        # データ取得
        val_p = patient_data[metric]
        vals_h = df_control_all[metric].dropna()
        if TARGET_GROUP == CONTROL_GROUP:
             vals_h = df_control_all[df_control_all["subject"] != TARGET_ID][metric].dropna()

        # 検定 (図への記載用)
        t_stat, p_val, _, _ = perform_crawford_test(val_p, vals_h)

        # --- Plotting ---
        # 1. 箱ひげ図 (Control)
        sns.boxplot(y=vals_h, width=0.35, ax=ax, 
                    boxprops=dict(facecolor="#ADD8E6", edgecolor="black"), showfliers=False)
        
        # 2. 個別データ点 (Control)
        sns.stripplot(y=vals_h, ax=ax, color="royalblue", size=8, alpha=0.7, jitter=0.1, 
                      label=f"{CONTROL_GROUP} (N={len(vals_h)})")
        
        # 3. 患者データ点 (Target)
        ax.scatter(0, val_p, color="red", marker="o", s=180, zorder=10, 
                   edgecolors="black", linewidth=1.5, label=f"Patient ({TARGET_ID})")

        # --- Design ---
        ax.set_title(label, fontsize=14, fontweight="bold", pad=12)
        
        # Y軸ラベル
        if "Ratio" in label: ylabel = "Ratio"
        elif "Area" in label: ylabel = "Area (mm²)"
        else: ylabel = "Length (mm)"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("")
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')

        # --- 統計情報のテキスト表示 ---
        df_dof = len(vals_h) - 1
        if not np.isnan(p_val):
            p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
            sig_mark = "*" if p_val < 0.05 else ""
            stats_text = f"Crawford's t({df_dof}) = {t_stat:.2f}\np {p_str} {sig_mark}"
        else:
            stats_text = "N/A"

        ax.text(0.95, 0.96, stats_text, transform=ax.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9))

        # 凡例の収集 (最初のプロットのみ)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # 重複回避のため辞書経由で整理
            by_label = dict(zip(labels, handles))
            # 表示順序指定
            order = [f"Patient ({TARGET_ID})", f"{CONTROL_GROUP} (N={len(vals_h)})"]
            legend_handles = [by_label[l] for l in order if l in by_label]
            legend_labels = [l for l in order if l in by_label]
        
        # 軸内の自動凡例は消す
        if ax.get_legend(): ax.get_legend().remove()

    # 共通凡例の描画
    fig.legend(legend_handles, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True, edgecolor="black", fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) # 凡例スペース確保
    
    img_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID}_vs_{CONTROL_GROUP}_{group_name}.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {os.path.basename(img_path)}")

print("\n=== Analysis Complete ===")