import math
from typing import List, Optional, Tuple


L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _xy(lm) -> Tuple[float, float]:
    return float(lm.x), float(lm.y)


def _angle_deg(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Optional[float]:
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

    knee_l = _angle_deg(lh, lk, la)
    knee_r = _angle_deg(rh, rk, ra)

    vals: List[float] = []
    if knee_l is not None:
        vals.append(knee_l)
    if knee_r is not None:
        vals.append(knee_r)
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


def _merge_segments(
    segments: List[Tuple[int, int]],
    merge_gap_frames: int,
) -> List[Tuple[int, int]]:
    if not segments:
        return []

    gap = max(0, int(merge_gap_frames))
    segs = sorted((int(s), int(e)) for s, e in segments if e > s)
    if not segs:
        return []

    merged: List[Tuple[int, int]] = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def detect_deadlift_segments(
    poses: List[Optional[list]],
    high_min_deg: float,
    high_max_deg: float,
    min_drop_deg: float,
    min_recovery_deg: float,
    min_rep_frames: int,
    max_rep_frames: int,
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

    high_min = float(high_min_deg)
    high_max = float(high_max_deg)
    drop_need = max(0.0, float(min_drop_deg))
    recover_need = max(0.0, float(min_recovery_deg))
    rep_min = max(1, int(min_rep_frames))
    rep_max = max(rep_min, int(max_rep_frames))
    pre = max(0, int(pre_frames))
    post = max(0, int(post_frames))

    state = "WAIT_HIGH"
    start_idx = -1
    start_angle = 0.0
    min_idx = -1
    min_angle = 0.0
    segments: List[Tuple[int, int]] = []

    for i, angle in enumerate(knee_angles):
        if angle is None:
            continue
        v = float(angle)

        if state == "WAIT_HIGH":
            if high_min <= v <= high_max:
                state = "WAIT_DROP"
                start_idx = i
                start_angle = v
                min_idx = i
                min_angle = v
            continue

        if state == "WAIT_DROP":
            if v < min_angle:
                min_angle = v
                min_idx = i

            if (start_angle - min_angle) >= drop_need:
                state = "WAIT_RETURN_HIGH"
                continue

            if i - start_idx > rep_max:
                state = "WAIT_HIGH"
                continue

            if high_min <= v <= high_max:
                start_idx = i
                start_angle = v
                min_idx = i
                min_angle = v
            continue

        if state == "WAIT_RETURN_HIGH":
            if v < min_angle:
                min_angle = v
                min_idx = i

            if i - start_idx > rep_max:
                state = "WAIT_HIGH"
                continue

            recovered = v - min_angle
            if high_min <= v <= high_max and i > min_idx and recovered >= recover_need:
                if (i - start_idx + 1) >= rep_min:
                    seg_start = max(0, start_idx - pre)
                    seg_end = min(n - 1, i + post)
                    segments.append((seg_start, seg_end))
                state = "WAIT_HIGH"

    return _merge_segments(segments, merge_gap_frames=merge_gap_frames)
