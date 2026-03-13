import math
from typing import List, Optional, Tuple


L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _xy(lm) -> Tuple[float, float]:
    return float(lm.x), float(lm.y)


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na < 1e-8 or nc < 1e-8:
        return None
    cosv = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nc)))
    return float(math.degrees(math.acos(cosv)))


def _frame_knee_angle(pose: Optional[list]) -> Optional[float]:
    if pose is None:
        return None

    lh, rh = _xy(pose[L_HIP]), _xy(pose[R_HIP])
    lk, rk = _xy(pose[L_KNEE]), _xy(pose[R_KNEE])
    la, ra = _xy(pose[L_ANKLE]), _xy(pose[R_ANKLE])

    left = _angle_deg(lh, lk, la)
    right = _angle_deg(rh, rk, ra)

    vals = []
    if left is not None:
        vals.append(left)
    if right is not None:
        vals.append(right)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _ema_optional(x: List[Optional[float]], alpha: float) -> List[Optional[float]]:
    if not x:
        return []
    alpha = max(0.0, min(1.0, float(alpha)))
    y: List[Optional[float]] = [None] * len(x)
    prev: Optional[float] = None
    for i, v in enumerate(x):
        if v is None:
            y[i] = prev
            continue
        if prev is None:
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        y[i] = prev
    return y


def detect_low_knee_segments(
    poses: List[Optional[list]],
    knee_threshold_deg: float,
    min_low_knee_frames: int,
    pre_frames: int,
    post_frames: int,
    merge_gap_frames: int,
    ema_alpha: float,
) -> List[Tuple[int, int]]:
    knee_angles_raw = [_frame_knee_angle(pose) for pose in poses]
    knee_angles = _ema_optional(knee_angles_raw, alpha=ema_alpha)

    n = len(knee_angles)
    if n == 0:
        return []

    min_len = max(1, int(min_low_knee_frames))
    pre = max(0, int(pre_frames))
    post = max(0, int(post_frames))
    merge_gap = max(0, int(merge_gap_frames))

    low = [(v is not None and v < knee_threshold_deg) for v in knee_angles]
    runs: List[Tuple[int, int]] = []

    i = 0
    while i < n:
        if not low[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and low[i + 1]:
            i += 1
        end = i
        if (end - start + 1) >= min_len:
            runs.append((max(0, start - pre), min(n - 1, end + post)))
        i += 1

    if not runs:
        return []

    merged: List[Tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

