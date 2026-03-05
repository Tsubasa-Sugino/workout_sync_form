import numpy as np
from typing import List, Tuple


def ema(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if len(x) == 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def detect_reps_from_hipy(
    hip_y_smooth: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    min_rep_sec: float = 0.6,
    prominence: float = 0.01,
) -> List[Tuple[int, int, int]]:
    """
    ざっくり：
      hip_y（画像座標y）は「下がるほど値が増える」ので
      1レップの最下点は hip_y の局所最大になりやすい
    戻り値: (start, bottom, end)
    """
    n = len(hip_y_smooth)
    reps = []
    min_rep_frames = max(1, int(min_rep_sec * fps))

    for i in range(1, n - 1):
        if not valid_mask[i]:
            continue

        is_local_max = hip_y_smooth[i] > hip_y_smooth[i - 1] and hip_y_smooth[i] > hip_y_smooth[i + 1]
        if not is_local_max:
            continue

        if (hip_y_smooth[i] - min(hip_y_smooth[i - 1], hip_y_smooth[i + 1])) < prominence:
            continue

        bottom = i

        # 左側の局所最小を探す（トップ）
        start = bottom
        for j in range(bottom - 1, 1, -1):
            if not valid_mask[j]:
                continue
            if hip_y_smooth[j] < hip_y_smooth[j - 1] and hip_y_smooth[j] < hip_y_smooth[j + 1]:
                start = j
                break

        # 右側の局所最小を探す（トップ）
        end = bottom
        for j in range(bottom + 1, n - 2):
            if not valid_mask[j]:
                continue
            if hip_y_smooth[j] < hip_y_smooth[j - 1] and hip_y_smooth[j] < hip_y_smooth[j + 1]:
                end = j
                break

        if start < bottom < end and (end - start) >= min_rep_frames:
            reps.append((start, bottom, end))

    # 重複の簡易除去（bottom順に、重なったら無視）
    reps = sorted(reps, key=lambda x: x[1])
    filtered = []
    last_end = -1
    for s, b, e in reps:
        if s <= last_end:
            continue
        filtered.append((s, b, e))
        last_end = e
    return filtered