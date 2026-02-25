import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================
# 1. CSV パス
# ============================
csv_path = r"C:\Users\yuich\python_project\project_analysis_main_research\daily_results\20251201\KISEKI\STUDENT_CORR\ROMBERG\filtered_summary\STUDENT_CORR_ROMBERG_summary.csv"

# 出力フォルダ
save_dir = r"C:\Users\yuich\python_project\project_analysis_main_research\daily_results\20251201\KISEKI"
os.makedirs(save_dir, exist_ok=True)

# ============================
# 2. CSV 読み込み
# ============================
df = pd.read_csv(csv_path)

subjects = df["subject"].tolist()

# 値
EO_path = df["EO_path"].to_numpy()
EC_path = df["EC_path"].to_numpy()
EO_area = df["EO_area"].to_numpy()
EC_area = df["EC_area"].to_numpy()

# ============================
# 3. 平均値
# ============================
EO_path_mean = np.nanmean(EO_path)
EC_path_mean = np.nanmean(EC_path)
EO_area_mean = np.nanmean(EO_area)
EC_area_mean = np.nanmean(EC_area)

RR_path = EC_path_mean / EO_path_mean if EO_path_mean > 0 else np.nan
RR_area = EC_area_mean / EO_area_mean if EO_area_mean > 0 else np.nan

# ============================
# 4. 描画処理（関数化）
# ============================
def plot_slope_chart(EO, EC, ylabel, title, RR_value, save_name):
    plt.figure(figsize=(12, 7))
    x = [0, 1]  # EO=0, EC=1
    labels = ["EO", "EC"]

    # 個人データ
    for i, sid in enumerate(subjects):
        plt.plot(x, [EO[i], EC[i]],
                 marker="o",
                 label=sid)

    # 平均の太線
    plt.plot(x, [np.nanmean(EO), np.nanmean(EC)],
             marker="o",
             color="black",
             linewidth=4,
             label="Mean")

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.4)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Romberg ratio を下に記載
    plt.text(
        0.5, -0.12,
        f"Mean Romberg ratio: {RR_value:.2f}",
        ha="center",
        fontsize=14,
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()

    # 保存
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.close()


# ============================
# 5. Path の EO → EC 変化
# ============================
plot_slope_chart(
    EO_path, EC_path,
    ylabel="Path length [mm]",
    title="Path Length Change (EO → EC)",
    RR_value=RR_path,
    save_name="ROMBERG_path_EO_EC_change.png"
)

# ============================
# 6. Area の EO → EC 変化
# ============================
plot_slope_chart(
    EO_area, EC_area,
    ylabel="Area [mm²]",
    title="Convex Hull Area Change (EO → EC)",
    RR_value=RR_area,
    save_name="ROMBERG_area_EO_EC_change.png"
)

print("\n=== All Figures Saved ===")
