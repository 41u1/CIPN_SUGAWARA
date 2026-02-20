import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================
# ① CSV のパス
# ============================
csv_path = r"C:\Users\yuich\python_project\project_analysis_main_research\daily_results\20251201\ROMBERG_Variance\summary\ROMBERG_AllVariance_summary.csv"

# 保存先ディレクトリ
save_dir = r"C:\Users\yuich\python_project\project_analysis_main_research\daily_results\20251201\ROMBERG_Variance"
os.makedirs(save_dir, exist_ok=True)

# ============================
# ② CSV 読み込み + グループ指定
# ============================
TARGET_GROUP = "STUDENT_CORR"   # ← ここを変えるだけで他のグループもOK

df_all = pd.read_csv(csv_path)

# ============================
# ③ スパゲッティ図 描画関数
# ============================
def plot_slope_chart(df_subset, eo_col, ec_col, ylabel, title, RR_value, save_name):
    # データ準備
    EO = df_subset[eo_col].to_numpy()
    EC = df_subset[ec_col].to_numpy()
    subjects = df_subset["subject"].tolist()

    plt.figure(figsize=(10, 6)) # 少しサイズ調整
    x = [0, 1]  # EO=0, EC=1
    labels = ["EO", "EC"]

    # 個人ごとの線
    for i, sid in enumerate(subjects):
        if np.isnan(EO[i]) or np.isnan(EC[i]): continue # 欠損値対策
        plt.plot(
            x, [EO[i], EC[i]],
            marker="o", alpha=0.6, label=sid
        )

    # 平均の太線
    EO_mean = np.nanmean(EO)
    EC_mean = np.nanmean(EC)

    plt.plot(
        x, [EO_mean, EC_mean],
        marker="o", color="black", linewidth=4, label="Mean"
    )

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.4)

    # 凡例（数が多すぎる場合は調整が必要だが一旦そのまま）
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Romberg ratio 記載
    plt.text(
        0.5, -0.15,
        f"Mean Romberg Ratio: {RR_value:.2f}",
        ha="center",
        fontsize=14,
        transform=plt.gca().transAxes
    )

    plt.tight_layout()

    # 保存
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.close()

# ============================
# ④ メイン処理：カメラごとにループ
# ============================

# 解析したい指標リスト
targets = [
    ("Local_EO",  "Local_EC",  "Local_RR",  "Local Change"),
    ("Global_EO", "Global_EC", "Global_RR", "Global Change"),
    ("Total_EO",  "Total_EC",  "Total_RR",  "Total Change"),
]

CAMERAS = ["C1", "C2"]

for cam in CAMERAS:
    print(f"\n--- Processing Camera: {cam} ---")
    
    # グループ かつ カメラ でフィルタリング
    # ※ CSVに 'camera' カラムがある前提（前のコードで作成済み）
    df_cam = df_all[(df_all["Local_Group"] == TARGET_GROUP) & (df_all["camera"] == cam)].reset_index(drop=True)
    
    if len(df_cam) == 0:
        print(f"No data for {cam}")
        continue

    for eo_col, ec_col, rr_col, title_base in targets:
        
        # タイトルとファイル名にカメラ名を含める
        plot_title = f"{title_base} (EO → EC) [{TARGET_GROUP} - {cam}]"
        save_name  = f"{TARGET_GROUP}_{cam}_{title_base.split()[0]}_EO_EC_change.png"
        
        # 平均RR算出
        RR_mean = np.nanmean(df_cam[rr_col])

        plot_slope_chart(
            df_cam,
            eo_col, ec_col,
            ylabel="Variance Value",
            title=plot_title,
            RR_value=RR_mean,
            save_name=save_name
        )

print("\n=== All Figures Saved Successfully ===")