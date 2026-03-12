import json

import numpy as np
from dtaidistance import dtw


class DTWActionSegmenter:
    def __init__(self, template, threshold=1.5, step_size=2):
        """
        Args:
            template (np.array): お手本の1動作波形
            threshold (float): 検出のしきい値（小さいほど厳密）
            step_size (int): 窓を動かすフレーム数
        """
        self.template = self._normalize(template)
        self.template_len = len(template)
        self.threshold = threshold
        self.step_size = step_size

    def _normalize(self, data):
        """窓ごとに正規化することで振幅の差（動きの大きさ）を無視する"""
        std = np.std(data)
        return (data - np.mean(data)) / std if std > 0 else data - np.mean(data)

    def search(self, long_signal):
        """
        スライディングウィンドウと非線形アライメント（DTW）を組み合わせて
        動作の開始・終了地点を探し出す
        """
        results = []
        # スキャン範囲の全DTW距離を記録
        all_distances = []

        # 1. スライディングウィンドウで探索
        # ウィンドウサイズはお手本の80%〜120%くらいで可変にしても良いけど，
        # DTW自体が伸縮を吸収するので，まずはお手本と同じ長さで固定するのが基本．
        for i in range(0, len(long_signal) - self.template_len, self.step_size):
            window = long_signal[i : i + self.template_len]
            norm_window = self._normalize(window)

            # ここで「非線形アライメント」が行われ，時間を伸ばし縮みさせた最小距離が出る
            dist = dtw.distance(self.template, norm_window)
            all_distances.append((i, dist))

        # 2. 距離が「谷」になっていて，かつしきい値以下の場所を特定（非極大値抑制の簡易版）
        for j in range(1, len(all_distances) - 1):
            idx, dist = all_distances[j]
            if dist < self.threshold:
                # 前後より小さい（＝その付近で最も似ている）地点を特定
                if dist < all_distances[j - 1][1] and dist < all_distances[j + 1][1]:
                    results.append(
                        {"start": idx, "end": idx + self.template_len, "distance": dist}
                    )

        return results, all_distances


def load_json_data(file_path):
    """JSONファイルからデータを読み込む関数"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["pc1"]


if __name__ == "__main__":
    # JSONファイルからデータを読み込む
    all_frames_data = load_json_data("result/pca_results1.json")

    # お手本の波形（例: 最初のしゃがみ動作）を抽出
    template = np.array(all_frames_data[5:95])  # フレーム5〜95をお手本とする

    # 長い波形（例: 全体の動き）を抽出
    long_signal = np.array(
        load_json_data("result/pca_results2.json")
    )  # PCAの第1主成分を使用

    # DTWで動作の開始・終了地点を検出
    segmenter = DTWActionSegmenter(template, threshold=1.5, step_size=2)
    results, all_distances = segmenter.search(long_signal)

    print("検出された動作セグメント:")
    for res in results:
        print(
            f"開始フレーム: {res['start']}, 終了フレーム: {res['end']}, 距離: {res['distance']:.2f}"
        )
