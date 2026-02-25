# CIPN_SUGAWARA
CIPN患者の身体機能評価タスクを評価するためのコードです。
研究室のパソコン上でViTPoseを使って推定したキーポイントの座標について解析を行います。

# 姿勢推定解析パイプライン

## データ取得
研究室サーバーからCSVデータをコピーします．
- **URL**: `http://10.240.77.18:5000/sharing/UJkwnmlqt` （研究室パソコン用）

---

## ロンベルグ試験

### 重心座標の算出
- `src\2_time_series_feature\GoB_比率考慮版.py`
  - カメラC1，C2の両方について重心の座標を算出します．
  - **使い方**: 被験者番号と被験者グループを指定して実行します．
- `src\2_time_series_feature\GoB_L_only.py`
  - ロンベルグ試験のC1用スクリプトです．右側が映っておらずガクガクしてしまうため，左側だけで算出して置き換えます．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/456d52bd-2d0e-471d-91d2-a87c8e7f6a9c" />
</div>

### 軌跡長・凸包面積
- `src\3_summary_feature\ROMBERG\ROMBERG_ratio_filtered.py`
  - 重心座標から2次元平面への復元を行います．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/983ca2d1-7278-4918-b0dc-bb68214dc9d6" />
</div>

- `src\4_analysis\ROMBERG\クロフォードテスト_軌跡.py`
  - CSVデータを読み込んでEO（開眼），EC（閉眼），ROMBERG Ratioの3つを箱ひげ図として出力します．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/d084036a-ee51-42b4-858e-548fe62d6ed3" />
</div>

- `src\4_analysis\ROMBERG\重心軌跡比較.py`
  - スライド用の重心軌跡の可視化図の比較用スクリプトです．表示する患者の試行を選択できます．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/3e5e5002-abf4-46b5-825c-0cb814458d35" />
</div>

### 分散
- `src\3_summary_feature\ROMBERG\narrow_window_variance.py`
  - 3つの分散（TOTAL，GLOBAL，LOCAL）を出力します．基本的にはTOTAL（一般的な分散）を用います．
- `src\4_analysis\ROMBERG\クロフォードテスト_分散.py`
  - CSVデータを読み込んでEO，EC，ROMBERG Ratioの3つを箱ひげ図として出力します．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/8dcc89ea-6a34-4574-9710-d1c9fe51812f" />
</div>

### 重症度比較
- `src\4_analysis\重症度plot.py`
  - 重症度指標とロンベルグ率との比較を行うコードです．入力は手動で行います．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/4fa8ce31-08e1-4bd8-8e67-d65bb091c25c" />
</div>

---

## 動的タスク（※現在めぼしい結果は出ていない）

### 3次元復元
- `src\Scripts\reconstruction\calib\0_trim_calib.py`
  - 拍手音によって同期とトリミングを行います．出力は，Setting1（動的タスク）とSetting2（静的タスク）の2つです．ちゃんと同期できているかを波形で確認します．
- `src\Scripts\reconstruction\calib\1_calib.py`
  - 単眼とステレオのキャリブレーションを行います．再投影誤差が十分に低いかを確認します．うまくいかない場合は `src\Scripts\reconstruction\calib\calib_new.py` も試します．
- `src\Scripts\reconstruction\calib\2_triangulation.py`
  - 三角測量により3次元座標を出力します．
- `src\Scripts\reconstruction\calib\3_3d_visual.py`
  - 出力された座標を3次元空間にプロットします．3次元化がうまくいったかの確認とスライド用です．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/33672e38-2ed7-45ce-8b8a-644c481f91c3" />
</div>

### 3D解析（3次元化したデータを活用）
- `src\2_time_series_feature\3D\ANGLE.py`
  - 3次元の関節角度を求めます．
- `src\2_time_series_feature\3D\GoB.py`
  - 3次元の重心の座標を求めます．

---

## TUG試験

### フェーズ分割・レーダーチャート・多角的分析
- `src\4_analysis\TUG\指標算出.py`
  - フェーズ分割と起立時間，旋回時間，旋回半径，着席時間，総秒数，総歩数，歩幅の比を算出します．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/79487c26-7f43-4a4d-9d4a-7a60c843d625" />
</div>

- `src\4_analysis\TUG\レーダー＋凡例.py`
  - 7つの指標のレーダーチャートを作成します（`tug_metrics_averaged.csv` を読み込みます）．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/6acb3360-fba8-4bd1-988b-6d9d8f421f56" />
</div>

- `src\4_analysis\TUG\多角的分析.py`
  - 7つの指標についてPCA分析などの多角的な分析を行います．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/71920d01-efb2-4ffb-99bd-d0b0d8135c96" />
</div>

- `src\4_analysis\TUG\kiseki.py`
  - 腰座標の水平面上の軌跡をフェーズ分割で色分けします．分割がうまくいっているかの確認とスライド用です．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/7a8ba6c5-28fa-4b27-b3f7-93ac0c49c3a8" />
</div>

---

## 4m歩行

- `src\3_summary_feature\4MWALK\step_detection.py`
  - 接地の検出の可視化用です．処理が結構重いです．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/c120d0d6-6dd1-4d57-a6b1-59b1c7df5eca" />
</div>

- `src\3_summary_feature\4MWALK\箱ひげ図.py`
  - 一般的な歩行パラメータ（歩幅，歩隔，歩行速度，ケイデンス）を箱ひげ図にまとめます．
## 📁 フォルダ構成と役割

<details open>
<summary>1. data/</summary>
<pre>
raw/        実験で撮った元の動画データ（編集しない原本）
processed/  MediaPipe や ViTPose の出力（キーポイントCSV・動画）
features/   特徴量（歩行軌跡・関節角度・重心など）の保存
</pre>
</details>

---

<details>
<summary>2. notebooks/</summary>
<pre>
exploration.ipynb       データ探索（EDA）・PCA分析などの試行用ノート
feature_analysis.ipynb  特徴量（関節角度・歩行周期など）の分析
model_evaluation.ipynb  分類・クラスタリングなどモデル評価用
</pre>
</details>

---

<details>
<summary>3. src/</summary>
<pre>
preprocessing/        姿勢推定や動画前処理（MediaPipe, ViTPoseなど）
feature_engineering/  特徴量計算（角度・重心・歩行軌跡など）
reconstruction/       3Dポーズ再構築・可視化
analysis/             PCA・クラスタリング・グラフ描画など分析処理
__init__.py           モジュール認識用（空ファイル）
</pre>
</details>

---

<details>
<summary>4. results_YYYYMMDD/</summary>
<pre>
figures/  グラフ（png, pdfなど）
tables/   分析結果のCSVや統計表
reports/  まとめレポート（Markdown, PDFなど）
</pre>
</details>


---

<details>
<summary>5. configs/</summary>
<pre>
paths.yaml          データ・結果フォルダのパス設定（例：./results_{date}）
model_params.yaml   モデルの設定値（入力サイズ・バッチサイズなど）
preprocessing.yaml  前処理パラメータ（フィルタ設定など）
</pre>
</details>

---

<details>
<summary>6. その他</summary>
<pre>
requirements.txt        使用ライブラリ一覧（再現性確保）
README.md               プロジェクト概要と手順
.vscode/settings.json   VS Code の設定（整形・仮想環境指定など）
.vscode/launch.json     デバッグ実行設定
</pre>
</details><br>



# Tracking movie data
cipn患者解析用mediapie
### 使い方
1. requirements.txtに記述されたライブラリを作成した仮想環境にインストール
```
pip install -r requirements.txt
```
2. dataディレクトリまたはhand_dataディレクトリにトラッキングしたい動画を配置

3. src内でプログラムを実行<br>
姿勢のトラッキング
```
python main_tracking.py
```
手のトラッキング
```
python main_hand_tracking.py
```
4. outputディレクトリに結果の動画およびcsvが出力される


#### 参考リンク
- body tracking<br>
[Python 用姿勢ランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python?hl=ja)<br>
[姿勢ランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ja)<br>

- hand tracking<br>
[Python 用手のランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?hl=ja#image)<br>
[手のランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index?hl=ja#models)<br>

- API リファレンス<br>
[Module:mp](https://ai.google.dev/edge/api/mediapipe/python/mp#classes)<br>

# 環境構築
[venvで手軽にPythonの仮想環境を構築しよう](https://qiita.com/shun_sakamoto/items/7944d0ac4d30edf91fde)

