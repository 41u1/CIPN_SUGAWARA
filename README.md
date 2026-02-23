# CIPN_SUGAWARA
CIPN患者の身体機能評価タスクを評価するためのコードです。
研究室のパソコン上でViTPoseを使って推定したキーポイントの座標について解析を行います。

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

💡 <code>YYYYMMDD</code> は実行日で自動作成される（例：<code>results_20251020</code>）
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

