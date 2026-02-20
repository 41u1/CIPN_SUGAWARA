import os
import glob
import re
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# ==========================================
# ★ 設定エリア ★
# ==========================================
INPUT_ROOT_DIR = r'C:\Users\yuich\python_project\project_analysis_main_research\data\1_processed\3D_Result' 
OUTPUT_BASE_DIR = r"C:\Users\yuich\python_project\project_analysis_main_research\daily_results\20260126\TUG\Phase_Metrics"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

EXCLUDE_SUBJECTS = [] 
TARGET_DISTANCE_X = 3.0 

# ==========================================
# 1. 共通関数 (フィルタ・属性抽出)
# ==========================================
def extract_file_info(filename):
    fname_upper = filename.upper()
    if "NOCIPN" in fname_upper: group = "NOCIPN"
    elif "CIPN" in fname_upper: group = "CIPN"
    elif "STUDENT" in fname_upper: group = "Student"
    else: group = "Unknown"
    
    match = re.search(r'(P\d+|Subject\d+|S\d+)', fname_upper)
    subject_id = match.group(1) if match else "Unknown"
    
    if "MAX" in fname_upper: condition = "MAX"
    elif "NORMAL" in fname_upper: condition = "NORMAL"
    else: condition = "Unknown"
    
    return group, subject_id, condition

def lowpass_filter(data, fps, cutoff=6.0, order=4):
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) <= 15: return data
    return filtfilt(b, a, data)

# ==========================================
# 2. セグメンテーション (時刻の特定)
# ==========================================
def get_segmentation_times(df):
    """
    各フェーズの開始・終了時刻(sec)を特定して返す。
    dfには既にフィルタ済みの座標や時間が含まれている前提。
    """
    try:
        # 時間軸
        time = df['Time_Sec'].values
        fps = 1.0 / np.median(np.diff(time))

        # 座標 (Hip Center)
        hc_x = df['HC_X'].values
        hc_y = df['HC_Y'].values
        
        # 距離・速度
        dist_from_start = np.sqrt((hc_x - hc_x[0])**2) # 簡易的にX軸距離で判定
        vy = np.gradient(hc_y, time) # 垂直速度

        # --- イベント検知ロジック ---
        
        # 1. Turn (旋回)
        max_dist = np.max(dist_from_start)
        th_dist = TARGET_DISTANCE_X if max_dist >= TARGET_DISTANCE_X else max_dist * 0.98
        in_turn = dist_from_start > th_dist
        turn_idxs = np.where(in_turn)[0]
        turn_center = np.argmax(dist_from_start)

        if len(turn_idxs) > 0:
            turn_s_idx, turn_e_idx = turn_idxs[0], turn_idxs[-1]
            # 異常値ガード
            if turn_e_idx <= turn_s_idx:
                turn_s_idx = max(0, turn_center - int(1.0*fps))
                turn_e_idx = min(len(time)-1, turn_center + int(1.0*fps))
        else:
            turn_s_idx = max(0, turn_center - int(1.0*fps))
            turn_e_idx = min(len(time)-1, turn_center + int(1.0*fps))

        # 2. Rise (起立)
        # Turn開始より前で、最も高い位置(直立)の90%未満かつ上向き速度がある箇所
        stand_h = np.percentile(hc_y[:turn_s_idx], 90)
        rising = np.where((vy[:turn_s_idx] > 0.05) & (hc_y[:turn_s_idx] < stand_h * 0.8))[0]
        sist_s_idx = rising[0] if len(rising) > 0 else 0
        
        # 開始点からさらに速度0まで遡る
        for k in range(sist_s_idx, 0, -1):
            if vy[k] <= 0.01: sist_s_idx = k; break
        
        # 起立終了 (Turn開始前で、高さが安定し速度が落ちた点)
        sist_e_cands = np.where((time > time[sist_s_idx]) & (time < time[turn_s_idx]) & (hc_y > stand_h * 0.9) & (vy < 0.05))[0]
        sist_e_idx = sist_e_cands[0] if len(sist_e_cands) > 0 else sist_s_idx + int(1.5*fps)

        # 3. Sit (着座)
        after_turn = np.where((time > time[turn_e_idx]) & (dist_from_start < 1.0))[0]
        stsi_s_idx = after_turn[0] if len(after_turn) > 0 else turn_e_idx
        
        stsi_e_idx = len(time) - 1
        # 終了点探索 (速度がほぼ0で安定する点)
        if stsi_s_idx < len(time) - 10:
            local_vy = vy[stsi_s_idx:]
            search_start = stsi_s_idx + np.argmin(local_vy) if np.min(local_vy) < -0.1 else stsi_s_idx
            for k in range(search_start, len(time)):
                if np.abs(vy[k]) < 0.05:
                    if k+5 < len(time) and np.all(np.abs(vy[k:k+5]) < 0.05): stsi_e_idx = k; break
                    else: stsi_e_idx = k; break

        # --- 時刻の辞書を作成 (ご指定のキー名に合わせる) ---
        times = {
            'Time_Total_Start': time[sist_s_idx],
            'Time_Total_End':   time[stsi_e_idx],
            
            'Time_Rise_Start':  time[sist_s_idx],
            'Time_Rise_End':    time[sist_e_idx],
            
            # WalkOutは Rise終了〜Turn開始
            'Time_WalkOut_Start': time[sist_e_idx],
            'Time_WalkOut_End':   time[turn_s_idx], # 便宜上つなげる
            
            'Time_Turn_Start':  time[turn_s_idx],
            'Time_Turn_End':    time[turn_e_idx],
            
            # WalkBackは Turn終了〜Sit開始
            'Time_WalkBack_Start': time[turn_e_idx], # 便宜上つなげる
            'Time_WalkBack_End':   time[stsi_s_idx],
            
            'Time_Sit_Start':   time[stsi_s_idx],
            'Time_Sit_End':     time[stsi_e_idx]
        }
        return times

    except Exception as e:
        print(f"Segmentation Error: {e}")
        return None

# ==========================================
# 3. メトリクス計算関数 (ご要望の関数)
# ==========================================
def calculate_phase_metrics(df, start_time, end_time, phase_prefix):
    """
    指定された時間区間(start_time <= t <= end_time)のデータを切り出し、
    平均速度、移動距離などを計算して辞書で返す。
    """
    # 時間でスライス
    df_phase = df[(df['Time_Sec'] >= start_time) & (df['Time_Sec'] <= end_time)].copy()
    
    if len(df_phase) < 2:
        # データが少なすぎる場合はNaN埋め
        return {
            f"{phase_prefix}_Duration": 0,
            f"{phase_prefix}_Dist": 0,
            f"{phase_prefix}_MeanVel": 0,
            f"{phase_prefix}_MaxVel": 0
        }
    
    # 時間・座標取得
    dt = df_phase['Time_Sec'].diff().fillna(0).values
    x = df_phase['HC_X'].values
    z = df_phase['HC_Z'].values
    
    # 1. Duration (sec)
    duration = end_time - start_time
    
    # 2. Distance (m) : 累積移動距離
    dist_steps = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
    total_dist = np.sum(dist_steps)
    
    # 3. Velocity (m/s) : 水平面速度
    # 全体の平均速度 = 総距離 / 時間
    mean_vel = total_dist / duration if duration > 0 else 0
    
    # 瞬時速度の最大値 (ノイズ除去済み前提)
    inst_vel = np.sqrt(np.gradient(x, df_phase['Time_Sec'])**2 + np.gradient(z, df_phase['Time_Sec'])**2)
    max_vel = np.max(inst_vel)

    return {
        f"{phase_prefix}_Duration": duration,
        f"{phase_prefix}_Dist": total_dist,
        f"{phase_prefix}_MeanVel": mean_vel,
        f"{phase_prefix}_MaxVel": max_vel
    }

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**", "*TUG*.csv"), recursive=True)
    results = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        group, subj, cond = extract_file_info(fname)
        if subj in EXCLUDE_SUBJECTS: continue
        
        try:
            # 1. データ読み込みと前処理
            df = pd.read_csv(fpath).dropna()
            
            # 時間作成
            if 'TIME' in df.columns: t = df['TIME'].values / 1000.0
            else: t = np.arange(len(df)) / 30.0
            t = t - t[0]
            
            # 座標計算とフィルタリング
            fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 30.0
            
            hc_x_raw = (df['left_hip_X'] + df['right_hip_X']) / 2
            hc_z_raw = (df['left_hip_Z'] + df['right_hip_Z']) / 2
            hc_y_raw = (df['left_hip_Y'] + df['right_hip_Y']) / 2
            
            # Y軸反転チェック
            sc_y = (df['left_shoulder_Y'] + df['right_shoulder_Y']) / 2
            y_sign = -1 if np.mean(sc_y) < np.mean(hc_y_raw) else 1
            
            # フィルタ適用済みデータをdfに追加 (計算用にまとめる)
            df_proc = pd.DataFrame({
                'Time_Sec': t,
                'HC_X': lowpass_filter(hc_x_raw, fps),
                'HC_Y': lowpass_filter(hc_y_raw * y_sign, fps),
                'HC_Z': lowpass_filter(hc_z_raw, fps)
            })
            # Yのベースライン補正
            df_proc['HC_Y'] -= df_proc['HC_Y'].min()

            # 2. セグメンテーション (時刻取得)
            time_res = get_segmentation_times(df_proc)
            if not time_res: continue
            
            # 3. 結果行の初期化 (属性 + 時刻情報)
            row = {'Group': group, 'Subject': subj, 'Condition': cond, 'FileName': fname}
            row.update(time_res) # ここで Time_Rise_Start などが入る

            # 4. 各フェーズの詳細メトリクス計算 (ご指定のブロック)
            res_metrics = {}
            
            # --- 各フェーズ解析 ---
            # 1. Rise Phase
            res_metrics.update(calculate_phase_metrics(df_proc, row['Time_Rise_Start'], row['Time_Rise_End'], "Rise"))
            
            # 2. Walk Out (Rise End -> Turn Start)
            # ※ WalkOutの開始はRiseの終了、終了はTurnの開始と定義
            res_metrics.update(calculate_phase_metrics(df_proc, row['Time_Rise_End'], row['Time_Turn_Start'], "WalkOut"))
            
            # 3. Turn
            res_metrics.update(calculate_phase_metrics(df_proc, row['Time_Turn_Start'], row['Time_Turn_End'], "Turn"))
            
            # 4. Walk Back (Turn End -> Sit Start)
            res_metrics.update(calculate_phase_metrics(df_proc, row['Time_Turn_End'], row['Time_Sit_Start'], "WalkBack"))
            
            # 5. Sit
            res_metrics.update(calculate_phase_metrics(df_proc, row['Time_Sit_Start'], row['Time_Sit_End'], "Sit"))
            
            # 結果を結合
            row.update(res_metrics)
            results.append(row)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # --- CSV保存 ---
    if not results:
        print("データがありません")
        return

    df_result = pd.DataFrame(results)
    
    # カラム並び替え (見やすくする)
    base_cols = ['Group', 'Subject', 'Condition', 'FileName']
    # 時間カラム
    time_cols = [c for c in df_result.columns if 'Time_' in c]
    # メトリクスカラム (Rise_Duration, Rise_Vel...)
    metric_cols = [c for c in df_result.columns if c not in base_cols and c not in time_cols]
    
    # Phase順に並べたい場合の工夫 (Rise -> WalkOut -> Turn...)
    phase_order = ['Rise', 'WalkOut', 'Turn', 'WalkBack', 'Sit']
    sorted_metric_cols = []
    for p in phase_order:
        sorted_metric_cols.extend([c for c in metric_cols if p in c])
        
    final_cols = base_cols + time_cols + sorted_metric_cols
    # 存在しないカラムを除外して選択
    final_cols = [c for c in final_cols if c in df_result.columns]
    
    df_result = df_result[final_cols]

    save_path = os.path.join(OUTPUT_BASE_DIR, 'TUG_Phase_Detailed_Metrics.csv')
    df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"保存完了: {save_path}")

if __name__ == "__main__":
    main()