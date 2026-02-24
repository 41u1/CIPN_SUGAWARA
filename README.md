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

姿勢推定（研究室パソコン）


ロンベルグ試験

重心座標の算出
src\2_time_series_feature\GoB_比率考慮版.py
c1,C2両方について重心の座標を算出する
使い方　被験者番号と被験者グループを指定
src\2_time_series_feature\GoB_L_only.py
ロンベルグ試験のC1についてだけ，右側が映っていないのでガクガクしてしまうから左側だけで算出して置き換える

軌跡長・凸包面積
src\3_summary_feature\ROMBERG\ROMBERG_ratio_filtered.py
重心座標から2次元平面に復元を行う
src\4_analysis\ROMBERG\クロフォードテスト_軌跡.py
CSVデータを読み込んでEO，EC，ROMBERG Ratioの3つを箱ひげ図として出力
<img width="10602" height="2785" alt="image" src="https://github.com/user-attachments/assets/d084036a-ee51-42b4-858e-548fe62d6ed3" />

src\4_analysis\ROMBERG\重心軌跡比較.py
スライド用の重心軌跡の可視化図の比較．どれの患者のどの試行を表示するかは選べる．



分散
src\3_summary_feature\ROMBERG\narrow_window_variance.py
3つの分散が出る（TOTAL，GLOBAL,LOCAL）　基本的にはTOTAL（一般的な分散）を用いる

src\4_analysis\ROMBERG\クロフォードテスト_分散.py
CSVデータを読み込んでEO，EC，ROMBERG Ratioの3つを箱ひげ図として出力
<img width="9963" height="2949" alt="image" src="https://github.com/user-attachments/assets/8dcc89ea-6a34-4574-9710-d1c9fe51812f" />

重症度比較
src\4_analysis\重症度plot.py　
重症度指標と，ロンベルグ率との比較をするコード．入力は手動．

以下はめぼしい結果が出てない．
動的タスク
3次元復元
src\Scripts\reconstruction\calib\0_trim_calib.py
拍手音によって同期とトリミング．出力は，Setting1(動的タスク），Setting2(静的タスク）の二つ．ちゃんと同期できているかを波形で確認．
src\Scripts\reconstruction\calib\1_calib.py
単眼とステレオのキャリブレーションを行う．再投影誤差が十分に低いかを確認．うまくいかない場合(src\Scripts\reconstruction\calib\calib_new.py)も試す
src\Scripts\reconstruction\calib\2_triangulation.py
三角測量により3次元座標を出力する．
src\Scripts\reconstruction\calib\3_3d_visual.py
出力された座標を3次元空間にプロットする．3次元化がうまくいったかの確認とスライド用．
<img width="628" height="417" alt="image" src="https://github.com/user-attachments/assets/33672e38-2ed7-45ce-8b8a-644c481f91c3" />


TUG試験
フェーズ分割？
レーダ―チャート
src\4_analysis\TUG\指標算出.py
フェーズ分割と起立時間,旋回時間、旋回半径,着席時間，総秒数,総歩数,歩幅の比の算出
<img width="2304" height="1920" alt="image" src="https://github.com/user-attachments/assets/79487c26-7f43-4a4d-9d4a-7a60c843d625" />

src\4_analysis\TUG\レーダー＋凡例.py
7つの指標のレーダーチャートを作成（tug_metrics_averaged.csvを読み込む)
<img width="5122" height="3559" alt="image" src="https://github.com/user-attachments/assets/6acb3360-fba8-4bd1-988b-6d9d8f421f56" />

PCA？

4m歩行


