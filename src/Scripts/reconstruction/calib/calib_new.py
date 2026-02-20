import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import os
from pathlib import Path

# ==========================================================
# === 設定エリア ===========================================
# ==========================================================
# 1. 入力: 動画があるフォルダ
TARGET_DIR = r"C:\Users\yuich\python_project\project_analysis_main_research\data\1_processed\calib_trimed\CIPN\P003\Setting1"

# 2. 出力: 結果(.npz)を保存するフォルダ
OUTPUT_DIR = TARGET_DIR

# 3. ChArUcoボードの設定
SQUARES_VERTICALLY = 4     # 行数
SQUARES_HORIZONTALLY = 6   # 列数
SQUARE_LENGTH = 0.087      # マスのサイズ (m)
MARKER_LENGTH = 0.065      # マーカーのサイズ (m)
DICTIONARY_VAR = aruco.DICT_4X4_50 # 使用した辞書

# 4. 解析間隔 (変更点: 45 -> 15 へ細かくしてデータ数を稼ぐ)
FRAME_STEP = 45

# ==========================================================
# === 関数定義: ボード生成 =================================
# ==========================================================
def get_charuco_board():
    dictionary = aruco.getPredefinedDictionary(DICTIONARY_VAR)
    board = aruco.CharucoBoard(
        (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    try: board.setLegacyPattern(True)
    except: pass
    return board, dictionary

# ==========================================================
# === 関数定義: 単眼キャリブレーション =====================
# ==========================================================
def run_mono_calibration(video_path, camera_name, save_path):
    print(f"\n🚀 [{camera_name}] 単眼解析を開始: {Path(video_path).name}")
    
    board, dictionary = get_charuco_board()
    cap = cv2.VideoCapture(video_path)
    
    objpoints = [] # 3D座標
    imgpoints = [] # 2D座標
    
    img_size = None
    frame_count = 0
    valid_frames = 0
    
    all_board_corners = board.getChessboardCorners()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % FRAME_STEP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = gray.shape[::-1]
            
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)
            if len(corners) > 0:
                _, c_corners, c_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                
                # 点が6個以上あるフレームだけ採用
                if c_corners is not None and len(c_corners) > 5:
                    current_obj_pts = []
                    for c_id in c_ids:
                        current_obj_pts.append(all_board_corners[c_id[0]])
                    
                    objpoints.append(np.array(current_obj_pts, dtype=np.float32))
                    imgpoints.append(c_corners)
                    
                    valid_frames += 1
                    print(f"  [{camera_name}] Frame {frame_count}: OK (Total {valid_frames})", end="\r")
        frame_count += 1
    
    cap.release()
    print(f"\n  ✅ データ収集完了: {valid_frames} 枚")

    if valid_frames < 10:
        print(f"  ❌ エラー: [{camera_name}] 有効なフレームが足りません。")
        return None, None

    print(f"  🧮 計算中 (標準モード)...")
    try:
        ret, mtx, dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        print(f"  ✨ {camera_name} 完了! 誤差: {ret:.4f} px")
        
        np.savez(save_path, mtx=mtx, dist=dist, ret=ret)
        print(f"  💾 保存: {save_path}")
        return mtx, dist
    except Exception as e:
        print(f"  ❌ 計算エラー: {e}")
        return None, None

# ==========================================================
# === 関数定義: ステレオキャリブレーション (改良版) ========
# ==========================================================
def run_stereo_calibration_process(video_c1, video_c2, mtx1, dist1, mtx2, dist2, save_path):
    print(f"\n🚀 [Stereo] ステレオ解析を開始...")
    
    board, dictionary = get_charuco_board()
    cap1 = cv2.VideoCapture(video_c1)
    cap2 = cv2.VideoCapture(video_c2)
    
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    
    frame_count = 0
    valid_pairs = 0
    img_size = None
    all_board_corners = board.getChessboardCorners()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2: break
        
        if frame_count % FRAME_STEP == 0:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            if img_size is None: img_size = gray1.shape[::-1]

            corners1, ids1, _ = aruco.detectMarkers(gray1, dictionary)
            corners2, ids2, _ = aruco.detectMarkers(gray2, dictionary)

            if len(corners1) > 0 and len(corners2) > 0:
                _, c_corners1, c_ids1 = aruco.interpolateCornersCharuco(corners1, ids1, gray1, board)
                _, c_corners2, c_ids2 = aruco.interpolateCornersCharuco(corners2, ids2, gray2, board)

                if c_corners1 is not None and c_corners2 is not None:
                    if len(c_corners1) > 5 and len(c_corners2) > 5:
                        common_ids = np.intersect1d(c_ids1, c_ids2)
                        
                        # 共通して写っている点が6個以上なら採用
                        if len(common_ids) > 6:
                            obj_pts_tmp = []
                            img_pts_l_tmp = []
                            img_pts_r_tmp = []

                            for cid in common_ids:
                                idx1 = np.where(c_ids1 == cid)[0][0]
                                idx2 = np.where(c_ids2 == cid)[0][0]
                                img_pts_l_tmp.append(c_corners1[idx1])
                                img_pts_r_tmp.append(c_corners2[idx2])
                                obj_pts_tmp.append(all_board_corners[cid])

                            imgpoints_l.append(np.array(img_pts_l_tmp, dtype=np.float32))
                            imgpoints_r.append(np.array(img_pts_r_tmp, dtype=np.float32))
                            objpoints.append(np.array(obj_pts_tmp, dtype=np.float32))
                            
                            valid_pairs += 1
                            print(f"  [Stereo] Frame {frame_count}: Pair Found (Total {valid_pairs})", end="\r")
        frame_count += 1
    
    cap1.release()
    cap2.release()
    print(f"\n  ✅ データ収集完了: {valid_pairs} 組")

    if valid_pairs < 10:
        print("  ❌ エラー: ペアが足りません。")
        return

    print("  🧮 相対位置(R, T)を計算中 (USE_INTRINSIC_GUESS)...")
    
    # 【重要変更点】微調整を許可するフラグを使用
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    
    # 【重要変更点】戻り値の new_mtx, new_dist を受け取る
    ret, new_mtx1, new_dist1, new_mtx2, new_dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx1, dist1, mtx2, dist2,
        img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )
    
    print(f"\n📊 最終結果レポート")
    print(f"  RMS Error: {ret:.4f} pixel")
    print(f"  並進ベクトル T (距離 [m]):\n{T}")
    
    # 【重要変更点】最適化された新しいパラメータを保存する
    np.savez(save_path, 
             mtx1=new_mtx1, dist1=new_dist1, 
             mtx2=new_mtx2, dist2=new_dist2, 
             R=R, T=T, ret=ret)
    print(f"\n💾 全工程完了! ファイル保存: {save_path}")

# ==========================================================
# === メイン実行処理 =======================================
# ==========================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    search_path = Path(TARGET_DIR)
    
    # MP4とMOVの両方に対応
    c1_files = list(search_path.glob("*_C1_*.mp4")) + list(search_path.glob("*_C1_*.MOV"))
    c2_files = list(search_path.glob("*_C2_*.mp4")) + list(search_path.glob("*_C2_*.MOV"))

    if not c1_files or not c2_files:
        print(f"❌ 指定フォルダに動画が見つかりません: {TARGET_DIR}")
    else:
        video_c1 = str(c1_files[0])
        video_c2 = str(c2_files[0])
        
        out_path = Path(OUTPUT_DIR)
        path_c1_res = str(out_path / "calibration_result_C1.npz")
        path_c2_res = str(out_path / "calibration_result_C2.npz")
        path_stereo_res = str(out_path / "camera_params_stereo.npz")

        # Step 1: 左カメラ
        mtx1, dist1 = run_mono_calibration(video_c1, "Camera 1 (Left)", path_c1_res)
        
        # Step 2: 右カメラ
        if mtx1 is not None:
            mtx2, dist2 = run_mono_calibration(video_c2, "Camera 2 (Right)", path_c2_res)

            # Step 3: ステレオ
            if mtx2 is not None:
                run_stereo_calibration_process(video_c1, video_c2, mtx1, dist1, mtx2, dist2, path_stereo_res)