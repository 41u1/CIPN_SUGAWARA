import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.patches as patches

# =========================================================
# 1. 設定 (SETTINGS)
# =========================================================
BASE_ROOT = r"C:\Users\yuich\python_project\project_analysis_main_research"

# 分散の集計CSVがあるフォルダ
VARIANCE_SUMMARY_ROOT = os.path.join(BASE_ROOT, r"daily_results\20260216\分散\summary")
# ※CSVファイル名がパスに含まれていたためフォルダパスに修正しました

# 結果出力先
OUTPUT_DIR = os.path.join(BASE_ROOT, r"daily_results\20260216\分散\Crawford_Variance_RedBlue_LargeFont_Referenced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ▼▼▼ 解析対象の指定 ▼▼▼
TARGET_GROUPS = ["CIPN", "NOCIPN"] 
CONTROL_GROUP = "STUDENT"
GROUP_FOR_COMPARISON = "CIPN" # 群間比較を行いたいターゲットグループ名

# ▼▼▼ 被験者の指定 ▼▼▼
TARGET_IDS = "ALL"

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
    """ Crawford & Howell (1998) t-test (個別の症例 vs コントロール群) """
    n = len(control_vals)
    mean_c = np.mean(control_vals)
    std_c = np.std(control_vals, ddof=1)
    
    if std_c == 0: return np.nan, np.nan, mean_c, std_c

    t_val = (case_val - mean_c) / (std_c * np.sqrt((n + 1) / n))
    df = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df))
    
    return t_val, p_val, mean_c, std_c

def format_p_value(p):
    """ p値のフォーマット関数 """
    if pd.isna(p): return "-"
    if p < 0.001: return "< 0.001*"
    elif p < 0.05: return f"{p:.3f}*"
    else: return f"{p:.3f}"

# =========================================================
# 3. データ読み込み & 前処理
# =========================================================
print(f"--- Loading Variance Data ---")

df_all = load_variance_csv()
if df_all is None:
    # 動作確認用ダミーデータ生成 (ファイルがない場合)
    print("Warning: CSV not found. Creating dummy data for testing.")
    data = {
        'subject': ['Sub1', 'Sub2', 'P001', 'P001', 'Ctrl1', 'Ctrl2', 'Ctrl3'],
        'Group': ['CIPN', 'NOCIPN', 'CIPN', 'NOCIPN'] + ['STUDENT']*3,
        'camera': ['C1']*7,
        'Local_EO': np.random.rand(7)*10, 'Local_EC': np.random.rand(7)*15, 'Local_RR': np.random.rand(7)*2,
        'Global_EO': np.random.rand(7)*20, 'Global_EC': np.random.rand(7)*25, 'Global_RR': np.random.rand(7)*2,
        'Total_EO': np.random.rand(7)*30, 'Total_EC': np.random.rand(7)*35, 'Total_RR': np.random.rand(7)*2
    }
    df_all = pd.DataFrame(data)
    # C2カメラのデータも追加
    df_c2 = df_all.copy()
    df_c2['camera'] = 'C2'
    df_all = pd.concat([df_all, df_c2], ignore_index=True)

# =========================================================
# 4. カメラごとのループ処理 (C1, C2)
# =========================================================
CAMERAS = ["C1", "C2"]

# ▼▼▼ フォントサイズ設定 ▼▼▼
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 20

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": ".8", 
    "grid.color": ".9",
    "font.family": "Arial",
    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30
})

for cam in CAMERAS:
    print(f"\n{'='*20} Processing Camera: {cam} {'='*20}")

    # -----------------------------------------------------
    # データの抽出 (カメラ & グループ)
    # -----------------------------------------------------
    # 1. コントロール群
    df_control_cam = df_all[(df_all["Group"] == CONTROL_GROUP) & (df_all["camera"] == cam)].copy()
    
    # 2. ターゲット群 (複数グループ)
    df_target_cam = df_all[(df_all["Group"].isin(TARGET_GROUPS)) & (df_all["camera"] == cam)].copy()

    # 3. 群間比較用のCIPN群データ抽出
    df_cipn_group = df_all[(df_all["Group"] == GROUP_FOR_COMPARISON) & (df_all["camera"] == cam)].copy()

    if len(df_control_cam) < 3:
        print(f"⚠ Skipping {cam}: Not enough control data (N={len(df_control_cam)})")
        continue

    # ▼▼▼【重要】IDのユニーク化 (ID + Group名) ▼▼▼
    # これにより "P001" が CIPN と NOCIPN 両方にいても、別の行として扱われます
    df_target_cam["subject_display"] = df_target_cam["subject"] + " (" + df_target_cam["Group"] + ")"

    # 処理順序を Group順 -> Subject順 にソート (表やグラフでの並びを整理するため)
    df_target_cam = df_target_cam.sort_values(by=["Group", "subject"])

    # 対象IDリストの確定 (subject_display を使用)
    target_list = df_target_cam["subject_display"].unique().tolist()

    if not target_list:
        print(f"⚠ No targets found for {cam}. Skipping.")
        continue

    print(f"  Targets to process ({len(target_list)} subjects)")

    # -----------------------------------------------------
    # 統計解析 (Crawford Test - Individual)
    # -----------------------------------------------------
    metrics_list = [
        "Local_EO", "Local_EC", "Local_RR",
        "Global_EO", "Global_EC", "Global_RR",
        "Total_EO", "Total_EC", "Total_RR"
    ]
    summary_results = []

    for pid_disp in target_list:
        # subject_display (例: P001 (CIPN)) をキーにしてデータを抽出
        patient_data = df_target_cam[df_target_cam["subject_display"] == pid_disp].iloc[0]
        original_id = patient_data["subject"]
        p_group = patient_data["Group"]

        for metric in metrics_list:
            if metric not in patient_data: continue

            val_p = patient_data[metric]
            vals_h = df_control_cam[metric].dropna()

            # 同じグループの場合は本人除外 (念のため。通常Control以外なら影響なし)
            if p_group == CONTROL_GROUP:
                vals_h = vals_h[df_control_cam["subject"] != original_id]

            t_stat, p_val, mean_h, std_h = perform_crawford_test(val_p, vals_h)

            summary_results.append({
                "Camera": cam,
                "Metric": metric,
                "Patient_ID": pid_disp, # ここにユニーク名が入る
                "Group": p_group,
                "Patient_Value": val_p,
                "Control_Mean": mean_h,
                "Control_SD": std_h,
                "p_value": p_val,
                "Significant": "YES" if p_val < 0.05 else "no"
            })

    # CSV保存
    df_stats = pd.DataFrame(summary_results)
    groups_str = "_".join(TARGET_GROUPS)
    csv_out_path = os.path.join(OUTPUT_DIR, f"Stats_{groups_str}_vs_{CONTROL_GROUP}_{cam}.csv")
    df_stats.to_csv(csv_out_path, index=False)
    print(f"  Saved Stats CSV: {os.path.basename(csv_out_path)}")

    # -----------------------------------------------------
    # 色の設定 (Red vs Blue)
    # -----------------------------------------------------
    cipn_subjects = [s for s in target_list if "(CIPN)" in s]
    nocipn_subjects = [s for s in target_list if "(NOCIPN)" in s]

    # カラーマップの生成 (要素数に応じてグラデーションを作成)
    cmap_red = plt.cm.Reds
    colors_cipn = cmap_red(np.linspace(0.6, 1.0, max(1, len(cipn_subjects))))

    cmap_blue = plt.cm.Blues
    colors_nocipn = cmap_blue(np.linspace(0.6, 1.0, max(1, len(nocipn_subjects))))

    color_dict = {}
    for i, sub in enumerate(cipn_subjects):
        color_dict[sub] = colors_cipn[i]
    for i, sub in enumerate(nocipn_subjects):
        color_dict[sub] = colors_nocipn[i]

    # -----------------------------------------------------
    # 図の作成 (参考コードのスタイルを適用)
    # -----------------------------------------------------
    plot_categories = {
        "Local_Variance":  ["Local_EO", "Local_EC", "Local_RR"],
        "Global_Variance": ["Global_EO", "Global_EC", "Global_RR"],
        "Total_Variance":  ["Total_EO", "Total_EC", "Total_RR"]
    }

    # ラベル変換用辞書
    label_dict = {
        "Local_EO": "Local EO Variance", "Local_EC": "Local EC Variance", "Local_RR": "Local Romberg Ratio",
        "Global_EO": "Global EO Variance", "Global_EC": "Global EC Variance", "Global_RR": "Global Romberg Ratio",
        "Total_EO": "Total EO Variance", "Total_EC": "Total EC Variance", "Total_RR": "Total Romberg Ratio"
    }

    for cat_name, metrics in plot_categories.items():
        # 文字が大きいので図のサイズも大きくする
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # -------------------------------------------------------
            # 箱ひげ図 1: Control群 (背景・太い・緑系)
            # -------------------------------------------------------
            vals_h = df_control_cam[metric].dropna()
            sns.boxplot(y=vals_h, width=0.6, ax=ax, 
                        boxprops=dict(facecolor="#BDFFA3", edgecolor="#888888", alpha=0.6, linewidth=2),
                        showfliers=False, zorder=1)
            
            # -------------------------------------------------------
            # 箱ひげ図 2: CIPN群 (重ねる・細い・薄い赤)
            # -------------------------------------------------------
            vals_cipn = df_target_cam[df_target_cam["Group"] == "CIPN"][metric].dropna()
            
            if len(vals_cipn) > 0:
                sns.boxplot(y=vals_cipn, width=0.3, ax=ax,
                            boxprops=dict(facecolor="#FFCCCC", edgecolor="red", alpha=0.7, linewidth=3),
                            showfliers=False, zorder=2)

            # -------------------------------------------------------
            # 散布図プロット (マーカー巨大化)
            # -------------------------------------------------------
            np.random.seed(42)
            for pid in target_list:
                # pid はユニークID (例: P001 (CIPN)) なので正しく識別される
                patient_data = df_target_cam[df_target_cam["subject_display"] == pid].iloc[0]
                val_p = patient_data[metric]
                p_color = color_dict.get(pid, "black")

                stat_row = df_stats[(df_stats["Patient_ID"] == pid) & (df_stats["Metric"] == metric)]
                is_sig = False
                if not stat_row.empty:
                    is_sig = stat_row["Significant"].values[0] == "YES"
                
                # ▼▼▼ マーカーサイズを大きく設定 (800, 1000) ▼▼▼
                if is_sig:
                    marker = 'D'
                    size = 1000  # 巨大化
                    edge_c = 'black'
                    lw = 2.5
                    z_ord = 20
                else:
                    marker = 'o'
                    size = 800  # 巨大化
                    edge_c = 'white'
                    lw = 2.0
                    z_ord = 10
                
                x_pos = np.random.uniform(-0.08, 0.08)
                ax.scatter(x_pos, val_p, color=p_color, marker=marker, s=size, zorder=z_ord,
                           edgecolors=edge_c, linewidth=lw, alpha=0.9)

            # 軸設定
            label_clean = label_dict.get(metric, metric)
            ax.set_title(label_clean, fontsize=35, fontweight="bold", pad=20)
            
            ylabel = "Romberg Ratio" if "RR" in metric else "Variance (mm²)"
            ax.set_ylabel(ylabel, fontsize=30)
            
            # X軸ラベルに群情報を記載
            ax.set_xlabel(f"Box: Gray=Control, Red=CIPN", color="#333333", fontsize=24)
            ax.set_xticks([]) # X軸の目盛りは削除
            
            sns.despine(ax=ax, left=False, bottom=True, top=True, right=True)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 凡例設定 (エラー回避のためリストが空でないか確認して色を取得)
        c_cipn_leg = colors_cipn[-1] if len(colors_cipn) > 0 else 'gray'
        c_nocipn_leg = colors_nocipn[-1] if len(colors_nocipn) > 0 else 'gray'

        custom_lines = [
            Line2D([0], [0], color="#BDFFA3", lw=10, label='Control Box'),
            Line2D([0], [0], color='#FFCCCC', lw=10, label='CIPN Box'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c_cipn_leg, markeredgecolor='w', markersize=20, label='CIPN'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c_nocipn_leg, markeredgecolor='w', markersize=20, label='Non-CIPN'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='white', markersize=20, label='Non-Sig'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=20, label='Sig (p<0.05)')
        ]
        
        # 凡例を枠外に配置
        fig.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                   title="Legend", frameon=True, fancybox=True, borderpad=1, fontsize=24, title_fontsize=26)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        img_path = os.path.join(OUTPUT_DIR, f"RedBlue_Large_{cat_name}_{cam}.png")
        img_path_svg = os.path.join(OUTPUT_DIR, f"RedBlue_Large_{cat_name}_{cam}.svg")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.savefig(img_path_svg, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved Plot: {os.path.basename(img_path)}")


    # -----------------------------------------------------
    # B. 表 (Summary Table) の作成 - 修正版 (フォントサイズ大)
    # -----------------------------------------------------
    # 列ヘッダー
    col_labels = ["Subject"]
    for m in metrics:
        m_short = m.split("_")[1] # EO, EC, RR
        col_labels.extend([f"{m_short}", "p"])
    
    table_data = []

    # === 1. コントロール群 (統計) ===
    row_ctrl = [f"Control (N={len(df_control_cam)})"]
    for m in metrics:
        vals = df_control_cam[m].dropna()
        mu, sigma = np.mean(vals), np.std(vals, ddof=1)
        row_ctrl.extend([f"{mu:.2f}±{sigma:.2f}", "-"])
    table_data.append(row_ctrl)

    # === 2. CIPN群 (統計) & 群間比較 ===
    row_cipn_stats = [f"{GROUP_FOR_COMPARISON} Mean (N={len(df_cipn_group)})"]
    for m in metrics:
        vals_ctrl = df_control_cam[m].dropna()
        vals_cipn = df_cipn_group[m].dropna()
        
        if len(vals_cipn) > 1 and len(vals_ctrl) > 1:
            mu_c = np.mean(vals_cipn)
            std_c = np.std(vals_cipn, ddof=1)
            t_stat_g, p_val_g = stats.ttest_ind(vals_cipn, vals_ctrl, equal_var=False)
            row_cipn_stats.extend([f"{mu_c:.2f}±{std_c:.2f}", format_p_value(p_val_g)])
        else:
            row_cipn_stats.extend(["-", "-"])
    table_data.append(row_cipn_stats)

    # === 3. 個別症例 (Crawford) ===
    # target_list は既に「P001 (CIPN)」と「P001 (NOCIPN)」が区別されたリスト
    for pid in target_list:
        row = [pid]
        for m in metrics:
            stat_row = df_stats[(df_stats["Patient_ID"] == pid) & (df_stats["Metric"] == m)]
            if not stat_row.empty:
                val = stat_row["Patient_Value"].values[0]
                pval = stat_row["p_value"].values[0]
                row.extend([f"{val:.2f}", format_p_value(pval)])
            else:
                row.extend(["-", "-"])
        table_data.append(row)

    # --- 表の描画設定 ---
    h_cell = 0.8
    w_img = 24 # 幅を広く
    h_img = (len(table_data) + 2) * h_cell

    fig_tbl, ax_tbl = plt.subplots(figsize=(w_img, h_img))
    ax_tbl.axis('off')

    table = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.3] + [0.12, 0.08] * 3
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(24) # フォントサイズ大
    table.scale(1, 2.5) # スケール大

    # --- 色とスタイルの設定 ---
    colors_header = {
        "EO": {"face": "#dae8fc", "edge": "#6c8ebf"},
        "EC": {"face": "#ffe6cc", "edge": "#d79b00"},
        "RR": {"face": "#d5e8d4", "edge": "#82b366"}
    }

    for (row, col), cell in table.get_celld().items():
        # ヘッダー行 (row=0)
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=26)
            cell.set_linewidth(1.5)
            if col == 0:
                cell.set_facecolor("#f0f0f0")
            elif 1 <= col <= 2:
                cell.set_facecolor(colors_header["EO"]["face"])
            elif 3 <= col <= 4:
                cell.set_facecolor(colors_header["EC"]["face"])
            elif 5 <= col <= 6:
                cell.set_facecolor(colors_header["RR"]["face"])
        
        # データ行
        else:
            cell.set_text_props(fontsize=24)
            
            # CIPN Group行の背景色変更
            if "CIPN Mean" in table_data[row-1][0] or "CIPN Group Mean" in table_data[row-1][0]:
                cell.set_facecolor("#ffcccc")
                if col == 0: cell.set_text_props(weight='bold', color='darkred', fontsize=24)
            
            elif row == 1: # Control行
                cell.set_facecolor("#f9f9f9")
                if col == 0: cell.set_text_props(weight='bold')

            else: # 個別データの行
                if col == 0: # Subject名
                    txt = cell.get_text().get_text()
                    if "(CIPN)" in txt:
                        cell.set_text_props(color="darkred")
                    elif "(NOCIPN)" in txt:
                        cell.set_text_props(color="navy")

    plt.title(f"Statistical Summary: {cat_name} ({cam})", y=1.0, pad=40, fontsize=30, fontweight='bold')
    
    tbl_path = os.path.join(OUTPUT_DIR, f"Table_{cat_name}_{cam}.png")
    tbl_path_svg = os.path.join(OUTPUT_DIR, f"Table_{cat_name}_{cam}.svg")
    plt.savefig(tbl_path, dpi=300, bbox_inches='tight')
    plt.savefig(tbl_path_svg, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Table: {os.path.basename(tbl_path)}")

print("\n=== Analysis & Table Generation Complete ===")