# workout_sync_form

筋トレフォームを理想に近づけるための動画比較・評価Webアプリ **KinNi Kun** のリポジトリです。

## 概要

ユーザーが撮影した筋トレ動画と理想フォームの動画をアップロードし、骨格推定技術を用いてフォームを可視化・比較・評価します。対応種目はベンチプレス・デッドリフト・スクワットです。

## 技術スタック

### 言語

| 技術 | 用途 |
|------|------|
| Python | アプリ全体の実装言語 |

### フレームワーク・UI

| ライブラリ | バージョン指定 | 用途 |
|-----------|--------------|------|
| [Streamlit](https://streamlit.io/) | 未指定 | WebアプリのUIフレームワーク（マルチページ構成） |
| [streamlit-image-select](https://github.com/jrieke/streamlit-image-select) | 未指定 | 種目選択用の画像セレクターUI |

### コンピュータビジョン・姿勢推定

| ライブラリ | バージョン指定 | 用途 |
|-----------|--------------|------|
| [MediaPipe](https://mediapipe.dev/) | ==0.10.32 | 動画から33点の骨格ランドマーク（姿勢推定）を取得 |
| [OpenCV (cv2)](https://opencv.org/) | >=4.8 | 動画読み込み・フレーム処理・映像入出力 |

### データ処理・機械学習

| ライブラリ | バージョン指定 | 用途 |
|-----------|--------------|------|
| [NumPy](https://numpy.org/) | <2 | 骨格座標の数値演算 |
| [Pandas](https://pandas.pydata.org/) | 未指定 | フォーム指標データの集計・管理 |
| [SciPy](https://scipy.org/) | 未指定 | 信号処理・統計解析 |
| [scikit-learn](https://scikit-learn.org/) | 未指定 | 機械学習（クラスタリング・評価等） |
| [dtaidistance](https://dtaidistance.readthedocs.io/) | 未指定 | Dynamic Time Warping（DTW）によるフォームの時系列比較 |

### 可視化

| ライブラリ | バージョン指定 | 用途 |
|-----------|--------------|------|
| [Plotly](https://plotly.com/python/) | >=5.18 | インタラクティブなフォーム評価グラフの表示 |
| [Matplotlib](https://matplotlib.org/) | 未指定 | 静的グラフの描画 |

## アプリの構成

```
workout_sync_form/
├── app.py                  # トップページ（種目選択）
├── pages/
│   ├── upload_videos.py    # 動画アップロードページ
│   ├── video_trimming.py   # 動画トリミングページ（自動）
│   ├── video_trimming_manual.py  # 動画トリミングページ（手動）
│   └── form_results.py     # フォーム評価結果ページ
├── form_metric/
│   ├── src/                # 姿勢推定・指標計算のコアロジック
│   └── scripts/            # 評価スクリプト（スクワット・デッドリフト・ベンチプレス）
├── menu_images/            # 種目選択用の画像
├── video/                  # 動画ファイル置き場
└── requirements.txt        # 依存ライブラリ一覧
```

## セットアップ

```bash
pip install -r requirements.txt
streamlit run app.py
```