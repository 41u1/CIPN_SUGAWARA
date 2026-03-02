import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

# ==========================================
# ★ 設定エリア ★
# ==========================================
# 指定された入力CSVファイルのパス
INPUT_CSV_PATH = r"C:\Users\Kei15\CIPN\CIPN_SUGAWARA\daily_results/20260223/TUG/tug_final_ratio_metric/final_svg/tug_metrics_averaged.csv"

# 出力先フォルダ
INPUT_DIR = os.path.dirname(INPUT_CSV_PATH)
OUTPUT_BASE_DIR = r"CC:\Users\Kei15\CIPN\CIPN_SUGAWARA\daily_results/20260227/TUG"


# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif']
sns.set(font=['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif'])

# 解析対象の指標リスト
TARGET_METRICS = [
    'TUG Score (s)', 
    'Total Steps', 
    'Turn Radius (m)', 
    'Turn Duration (s)', 
    'Stride Length Ratio', 
    'Rise Duration (s)',
    'Sit Duration (s)'
]

# ==========================================
# 1. 可視化・解析関数群
# ==========================================
def plot_radar_chart_zscore(df_cond, target_cols, condition_name, output_dir):
    """ レーダーチャート """
    control_data = df_cond[df_cond['Group'] == 'Control']
    if len(control_data) == 0: return

    mean_ctrl = control_data[target_cols].mean()
    std_ctrl = control_data[target_cols].std()
    
    df_z = df_cond.copy()
    for col in target_cols:
        if std_ctrl[col] == 0: df_z[col] = 0
        else: df_z[col] = (df_cond[col] - mean_ctrl[col]) / std_ctrl[col]

    df_z_grouped = df_z.groupby('Group')[target_cols].mean()
    
    categories = target_cols; N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]; angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='black', size=12)
    plt.ylim(-3, 3)
    plt.yticks([-2, -1, 0, 1, 2], ["-2σ", "-1σ", "Avg", "+1σ", "+2σ"], color="grey", size=10)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    colors = {'Control': 'gray', 'Non-CIPN': 'blue', 'CIPN': 'red'}
    for grp, color in colors.items():
        if grp in df_z_grouped.index:
            val = df_z_grouped.loc[grp].values.flatten().tolist(); val += val[:1]
            width = 3 if grp == 'CIPN' else 1
            style = 'solid' if grp != 'Control' else '--'
            ax.plot(angles, val, linewidth=width, linestyle=style, label=grp, color=color)
            if grp != 'Control': ax.fill(angles, val, color, alpha=0.1)

    plt.title(f"Radar Chart (Z-score) - {condition_name}", size=18, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(os.path.join(output_dir, f"Radar_{condition_name}.png"), format='png', bbox_inches='tight')  
    plt.close()

def plot_heatmap(df_cond, target_cols, condition_name, output_dir):
    """ ヒートマップ """
    df_hm = df_cond[['Subject', 'Group'] + target_cols].copy().set_index('Subject')
    for col in target_cols:
        m, s = df_hm[col].mean(), df_hm[col].std()
        df_hm[col] = (df_hm[col] - m) / s if s != 0 else 0

    lut = {'Control': 'skyblue', 'Non-CIPN': 'blue', 'CIPN': 'red'}
    row_colors = df_hm['Group'].map(lut)
    g = sns.clustermap(df_hm[target_cols], row_colors=row_colors, cmap="vlag", center=0, col_cluster=False, figsize=(10, 8))
    g.fig.suptitle(f"Heatmap - {condition_name}", y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, f"Heatmap_{condition_name}.png"), format='png', bbox_inches='tight')    
    plt.close()

def plot_effect_sizes(df_cond, target_cols, condition_name, output_dir):
    """ 効果量プロット """
    ctrl = df_cond[df_cond['Group'] == 'Control']
    cipn = df_cond[df_cond['Group'] == 'CIPN']
    if len(ctrl) < 2 or len(cipn) < 2: return

    effect_sizes = []
    for col in target_cols:
        m1, m2 = ctrl[col].mean(), cipn[col].mean()
        s1, s2 = ctrl[col].std(), cipn[col].std()
        n1, n2 = len(ctrl), len(cipn)
        pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
        d = (m2 - m1) / pooled_sd if pooled_sd != 0 else 0
        effect_sizes.append({'Metric': col, 'Effect Size (d)': d})
        
    df_es = pd.DataFrame(effect_sizes).sort_values(by='Effect Size (d)', key=abs, ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_es, x='Effect Size (d)', y='Metric', palette='coolwarm')
    plt.axvline(0.8, color='gray', linestyle='--'); plt.axvline(-0.8, color='gray', linestyle='--')
    plt.title(f"Effect Size (Cohen's d) - {condition_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"EffectSize_{condition_name}.png"), format='png')   
    plt.close()

def plot_pca_scatter(df_cond, target_cols, condition_name, output_dir):
    """ ★ PCA散布図 (ラベル・タイトルを大型化，Controlを緑化) ★ """
    if len(df_cond) < 3: return
    X = df_cond[target_cols].fillna(df_cond[target_cols].mean())
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2); X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['Group'] = df_cond['Group'].values
    df_pca['Subject'] = df_cond['Subject'].values

    pc1_ctrl = df_pca[df_pca['Group'] == 'Control']['PC1']
    pc1_cipn = df_pca[df_pca['Group'] == 'CIPN']['PC1']
    stats_text = ""
    if len(pc1_ctrl) > 1 and len(pc1_cipn) > 1:
        t, p = stats.ttest_ind(pc1_ctrl, pc1_cipn, equal_var=False)
        stats_text = f"PC1 T-test: p={p:.3f}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")

    # サイズ設定
    plt.figure(figsize=(14, 10))
    from scipy.spatial import ConvexHull
    pts = df_pca[df_pca['Group'] == 'Control'][['PC1', 'PC2']].values
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        # ★【修正】Convex Hullの塗りつぶし色を緑に変更
        plt.fill(pts[hull.vertices,0], pts[hull.vertices,1], 'limegreen', alpha=0.1)
        plt.plot(pts[hull.vertices,0], pts[hull.vertices,1], 'k--', alpha=0.3)

    # ★【修正】散布図のパレットで 'Control' を 'limegreen' に変更
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Group', style='Group', 
                    palette={'Control': 'limegreen', 'Non-CIPN': 'blue', 'CIPN': 'red'}, s=600)
    
    # 症例番号のフォントサイズ
    for i in range(len(df_pca)):
        if df_pca.Group[i] in ['CIPN', 'Non-CIPN']:
            plt.text(df_pca.PC1[i]+0.2, df_pca.PC2[i], df_pca.Subject[i], 
                     color='darkred', weight='bold', fontsize=25)

    # --- タイトルと軸のフォントサイズ設定 ---
    plt.title(f"PCA Map ({condition_name})\n{stats_text}", fontsize=35, fontweight='bold', pad=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=28)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=28)
    
    # 目盛りの数字
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    
    plt.axhline(0, color='gray', linestyle=':'); plt.axvline(0, color='gray', linestyle=':')
    
    # 凡例の大型化（paletteの設定と自動で連動します）
    plt.legend(fontsize=20, title="Group", title_fontsize=24, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"PCA_{condition_name}.png"), format='png')   
    plt.close()

def plot_composite_score_single(df_cond, target_cols, condition_name, output_dir):
    """ ★ RATIO専用の1枚パネル・大型 Composite Score ★ """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # --- 統計計算 (Control基準のZスコア平均) ---
    ctrl_df = df_cond[df_cond['Group'] == 'Control']
    if len(ctrl_df) > 0:
        mean_vec = ctrl_df[target_cols].mean()
        std_vec = ctrl_df[target_cols].std()
        
        z_scores = pd.DataFrame()
        for col in target_cols:
            if std_vec[col] != 0:
                z_scores[col] = ((df_cond[col] - mean_vec[col]) / std_vec[col]).abs()
            else:
                z_scores[col] = 0
        df_cond = df_cond.copy()
        df_cond['Composite_Score'] = z_scores.mean(axis=1)
    else:
        df_cond['Composite_Score'] = 0

    # --- Control Box (緑) ---
    vals_ctrl = df_cond[df_cond['Group'] == 'Control']['Composite_Score']
    sns.boxplot(y=vals_ctrl, width=0.5, ax=ax, 
                boxprops=dict(facecolor="#BDFFA3", edgecolor="#888888", alpha=0.6, linewidth=2),
                showfliers=False, zorder=1)
    
    # --- CIPN Box (赤) ---
    vals_cipn = df_cond[df_cond['Group'] == 'CIPN']['Composite_Score']
    if len(vals_cipn) > 0:
        sns.boxplot(y=vals_cipn, width=0.25, ax=ax,
                    boxprops=dict(facecolor="#FFCCCC", edgecolor="red", alpha=0.7, linewidth=3),
                    showfliers=False, zorder=2)

    # --- 散布図プロット ---
    np.random.seed(42)
    control_array = vals_ctrl.values
    for _, row in df_cond.iterrows():
        val_p = row['Composite_Score']
        grp = row['Group']
        pid = row['Subject']
        
        # Crawford 有意差判定
        is_sig = False
        if grp != 'Control' and len(control_array) > 1:
            c_mean, c_std = np.mean(control_array), np.std(control_array, ddof=1)
            if c_std > 0:
                t_val = (val_p - c_mean) / (c_std * np.sqrt((len(control_array) + 1) / len(control_array)))
                p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=len(control_array)-1))
                is_sig = p_val < 0.05
        
        p_color = "red" if grp == "CIPN" else ("blue" if grp == "Non-CIPN" else "gray")
        marker, size, edge, lw = ('D', 1000, 'black', 2.5) if is_sig else ('o', 800, 'white', 2.0)
        alpha_val = 0.9 if grp != 'Control' else 0.4
        
        ax.scatter(np.random.uniform(-0.06, 0.06), val_p, color=p_color, marker=marker, s=size, 
                   edgecolors=edge, linewidth=lw, alpha=alpha_val, zorder=10)

    # --- 軸ラベル設定 ---
    ax.tick_params(axis='y', labelsize=30)
    ax.set_title(f"Composite Score ({condition_name})", fontsize=40, fontweight="bold", pad=20)
    ax.set_ylabel("Abnormality Index (Z-mean Ratio)", fontsize=30)
    ax.set_xlabel("Box: Green=Control, Red=CIPN", fontsize=24)
    ax.set_xticks([])
    sns.despine(ax=ax, left=False, bottom=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # --- 凡例設定 ---
    custom_lines = [
        Line2D([0], [0], color="#BDFFA3", lw=10, label='Control Box'),
        Line2D([0], [0], color='#FFCCCC', lw=10, label='CIPN Box'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20, label='CIPN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=20, label='Non-CIPN'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=20, label='Sig (p<0.05)')
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize=18, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CompositeScore_{condition_name}.png"), dpi=300)
    plt.close()

# ==========================================
# 2. メイン実行フロー
# ==========================================
def main():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return
    
    print(f"Loading data from: {INPUT_CSV_PATH}")
    df_raw = pd.read_csv(INPUT_CSV_PATH)

    # ★ 群名称を統一置換 ★
    df_raw['Group'] = df_raw['Group'].replace({'Student': 'Control', 'NOCIPN': 'Non-CIPN'})
    df_mean = df_raw.copy()

    # Ratioデータの生成
    if 'RATIO' in df_mean['Condition'].unique():
        print("Ratio data found in CSV. Using it directly.")
        df_all = df_mean
    else:
        print("Calculating Ratio (MAX/NORMAL)...")
        df_pivot = df_mean.pivot(index=['Group', 'Subject'], columns='Condition', values=TARGET_METRICS)
        df_ratio = pd.DataFrame(index=df_pivot.index)
        valid_ratio = False
        for col in TARGET_METRICS:
            if 'MAX' in df_pivot[col].columns and 'NORMAL' in df_pivot[col].columns:
                df_ratio[col] = df_pivot[col]['MAX'] / df_pivot[col]['NORMAL']
                valid_ratio = True
        
        if valid_ratio:
            df_ratio = df_ratio.reset_index()
            df_ratio['Condition'] = 'RATIO'
            df_all = pd.concat([df_mean, df_ratio], ignore_index=True)
        else:
            print("Warning: Failed to calculate Ratio. MAX or NORMAL data might be missing.")
            df_all = df_mean

    # ★ 解析ループを RATIO のみに限定 ★
    for cond in ['RATIO']:
        print(f"\n--- Focused Analysis: {cond} ONLY ---")
        df_cond = df_all[df_all['Condition'] == cond]
        if len(df_cond) == 0: 
            print("No data found for RATIO.")
            continue
        
        plot_radar_chart_zscore(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_heatmap(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_effect_sizes(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_pca_scatter(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        
        # 1パネル用の Composite Score 関数を呼び出す
        plot_composite_score_single(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)

    print(f"\nCompleted! Check output folder: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()