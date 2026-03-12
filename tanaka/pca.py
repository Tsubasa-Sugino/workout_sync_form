import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA


# ==========================================
# 1. データ整形
# ==========================================
def format_data_for_pca(all_frames_data):
    """
    33関節の3D座標(x, y, z)を，1フレームあたり99次元の配列に平坦化するよ．
    出力: (フレーム数, 99) のNumPy配列
    """
    data_matrix = []

    for frame in all_frames_data:
        if frame is None:
            continue

        frame_features = []
        for lm in frame["3d"]:
            frame_features.extend([lm["x"], lm["y"], lm["z"]])

        data_matrix.append(frame_features)

    matrix_np = np.array(data_matrix)
    print(f"整列完了！ データの形: {matrix_np.shape}")
    return matrix_np


# ==========================================
# 2. 主成分分析（PCA）
# ==========================================
def perform_pca(data_matrix, n_components=3):
    """
    多次元データを圧縮して，一番大きく変化している「主成分（PC1）」を抽出するよ．
    出力: 第1主成分の波形(1D配列)，寄与率(float)，第1主成分の係数(1D配列)
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_matrix)

    # 第1主成分（PC1）を取り出す
    pc1 = pca_result[:, 0]

    # 寄与率
    explained_variance = pca.explained_variance_ratio_[0]

    # 第1主成分の係数（各入力変数に対する重みベクトル）
    # shapeは (n_features,) になるよ
    components = pca.components_[0]

    print(f"PCA完了！ 第1主成分の寄与率: {explained_variance * 100:.1f}%")

    return pc1, explained_variance, components


# ==========================================
# 3. データ可視化とタイミング検出
# ==========================================
def visualize_pca_wave(pc1, explained_variance, title="Squat Movement Analysis"):
    """
    抽出した波形を描画し，動作のチェックポイント（谷底）を赤のバツ印でプロットするよ．
    出力: 検出したピークのフレーム番号のリスト
    """
    # しゃがみこんだタイミング（谷底＝マイナス方向のピーク）を検出
    peaks, _ = find_peaks(-pc1, distance=30, prominence=0.1, height=0.1)

    plt.figure(figsize=(12, 6))
    plt.plot(
        pc1, label=f"PC1 (Main Motion) - {explained_variance * 100:.1f}%", color="blue"
    )

    # 検出したチェックポイントに印をつける
    plt.plot(peaks, pc1[peaks], "rx", markersize=10, label="Checkpoints (Bottom)")

    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Principal Component 1 Value")
    plt.grid(True)
    plt.legend()
    plt.savefig("pca_waveform.png")
    plt.close()

    return peaks


import copy


def normalize_pose_landmarks(all_frames_data):
    """
    全フレームの3D座標に対して，位置・回転・スケールの正規化を行うよ．
    """
    if not all_frames_data:
        return []

    normalized_data = copy.deepcopy(all_frames_data)

    # ランドマークIDの定義
    L_HIP, R_HIP = 23, 24
    L_SHOULDER, R_SHOULDER = 11, 12
    # 首の代用（肩の中点）

    for frame in normalized_data:
        if frame is None or "3d" not in frame:
            continue

        # 3D座標をnumpy配列に変換 (Shape: [33, 3])
        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in frame["3d"]])

        # --- 1. 位置の補正 (Translation) ---
        # 左右の腰の中点を原点 (0,0,0) に移動
        hip_center = (coords[L_HIP] + coords[R_HIP]) / 2.0
        coords -= hip_center

        # --- 2. スケールの補正 (Scaling) ---
        # 腰の中点から肩の中点までの距離を「基準の1.0」とする
        shoulder_center = (coords[L_SHOULDER] + coords[R_SHOULDER]) / 2.0
        spine_vec = shoulder_center - [0, 0, 0]  # 腰中点(0,0,0)から肩中点へのベクトル
        scale = np.linalg.norm(spine_vec)

        if scale > 0:
            coords /= scale

        # --- 3. 回転の補正 (Rotation) ---
        # ① Y軸周りの回転（左右の向きを正面に固定）
        # 右腰へのベクトルを使い，XZ平面での角度を計算
        hip_vec = coords[R_HIP] - coords[L_HIP]
        angle_y = np.arctan2(hip_vec[2], hip_vec[0])

        c, s = np.cos(-angle_y), np.sin(-angle_y)
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        coords = coords @ R_y.T

        # ② Z軸周りの回転（体の傾きを垂直に固定）
        # 肩の中点の位置を使って，上半身が真上(Y軸)を向くように回転
        new_shoulder_center = (coords[L_SHOULDER] + coords[R_SHOULDER]) / 2.0
        angle_z = np.arctan2(new_shoulder_center[0], -new_shoulder_center[1])

        c, s = np.cos(-angle_z), np.sin(-angle_z)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        coords = coords @ R_z.T

        # --- 4. 結果の書き戻し ---
        for i, lm in enumerate(frame["3d"]):
            lm["x"], lm["y"], lm["z"] = coords[i]

    return normalized_data


def load_json_data(file_path):
    """JSONファイルからデータを読み込む関数"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_data(data, file_path):
    """データをJSONファイルに保存する関数"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # JSONファイルからデータを読み込む
    all_frames_data = load_json_data("result/output_video_result1.json")
    all_frames_data = normalize_pose_landmarks(all_frames_data)

    # PCAで分析してグラフを保存
    pc1, peaks, _ = perform_pca(format_data_for_pca(all_frames_data))
    visualize_pca_wave(pc1, peaks)
    data = {"pc1": pc1.tolist(), "peaks": peaks.tolist()}
    save_json_data(data, "result/pca_results1.json")
