import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .rep_detection import ema, detect_reps_from_hipy

# BlazePose indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _xy(lm) -> Tuple[float, float]:
    return (float(lm.x), float(lm.y))


def _mid(a, b) -> Tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _angle_deg(a, b, c) -> float:
    # angle ABC
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na < 1e-8 or nc < 1e-8:
        return float("nan")
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return float(math.degrees(math.acos(cosv)))


@dataclass
class RepMetrics:
    rep_idx: int
    start: int
    bottom: int
    end: int
    knee_angle_deg: float     # 最下点の膝角（平均）
    shoulder_fb_norm: float   # 最下点の肩前後（肩x-股関節x を体幹長で正規化）
    down_time_s: float
    up_time_s: float


@dataclass
class VideoSummary:
    video_path: str
    fps: float
    reps: List[RepMetrics]
    rep_count: int
    mean_knee_angle: float
    mean_shoulder_fb: float
    mean_down_time: float
    mean_up_time: float


def _frame_features(pose) -> Dict[str, float]:
    ls, rs = _xy(pose[L_SHOULDER]), _xy(pose[R_SHOULDER])
    lh, rh = _xy(pose[L_HIP]), _xy(pose[R_HIP])
    lk, rk = _xy(pose[L_KNEE]), _xy(pose[R_KNEE])
    la, ra = _xy(pose[L_ANKLE]), _xy(pose[R_ANKLE])

    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)
    knee_mid = _mid(lk, rk)

    torso_len = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]) + 1e-8

    # 1) 膝角（デッドリフトと同じ：hip-knee-ankle）
    knee_l = _angle_deg(lh, lk, la)
    knee_r = _angle_deg(rh, rk, ra)
    knee_angle = float(np.nanmean([knee_l, knee_r]))

    # 2) 肩の前後（“肩の位置”を相対化して安定させる）
    #    肩が股関節より前に出るほど値が増える（サイド撮影想定）
    shoulder_fb_norm = (shoulder_mid[0] - hip_mid[0]) / torso_len

    # レップ検出用：hip_y
    hip_y = hip_mid[1]

    return {
        "hip_y": hip_y,
        "knee_angle_deg": knee_angle,
        "shoulder_fb_norm": shoulder_fb_norm,
    }


def analyze_squat(
    poses: List[Optional[list]],
    fps: float,
    ema_alpha: float = 0.2,
    min_rep_sec: float = 0.6,
    prominence: Optional[float] = None,
) -> Tuple[List[RepMetrics], Dict[str, float]]:
    feats: List[Optional[Dict[str, float]]] = []
    valid = np.zeros(len(poses), dtype=bool)

    for i, pose in enumerate(poses):
        if pose is None:
            feats.append(None)
        else:
            f = _frame_features(pose)
            feats.append(f)
            valid[i] = True

    # hip_y signal
    hip_y = np.zeros(len(poses), dtype=float)
    last = None
    for i in range(len(poses)):
        if feats[i] is not None:
            hip_y[i] = feats[i]["hip_y"]
            last = hip_y[i]
        else:
            hip_y[i] = last if last is not None else 0.0

    hip_y_s = ema(hip_y, alpha=ema_alpha)

    # 固定 prominence が強すぎるとレップが0件になるので、未指定時は信号レンジから推定する
    if prominence is None:
        if np.any(valid):
            v = hip_y_s[valid]
            hip_range = float(np.max(v) - np.min(v))
            prominence_eff = min(0.005, max(0.0008, hip_range * 0.01))
        else:
            prominence_eff = 0.001
    else:
        prominence_eff = float(prominence)

    reps_idx = detect_reps_from_hipy(
        hip_y_smooth=hip_y_s,
        valid_mask=valid,
        fps=fps,
        min_rep_sec=float(min_rep_sec),
        prominence=prominence_eff,
    )

    reps: List[RepMetrics] = []
    for r_i, (s, b, e) in enumerate(reps_idx):
        if feats[b] is None or feats[s] is None or feats[e] is None:
            continue

        reps.append(
            RepMetrics(
                rep_idx=r_i,
                start=s,
                bottom=b,
                end=e,
                knee_angle_deg=float(feats[b]["knee_angle_deg"]),
                shoulder_fb_norm=float(feats[b]["shoulder_fb_norm"]),
                down_time_s=float((b - s) / fps),
                up_time_s=float((e - b) / fps),
            )
        )

    def _mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    agg = {
        "rep_count": len(reps),
        "mean_knee_angle": _mean([r.knee_angle_deg for r in reps]),
        "mean_shoulder_fb": _mean([r.shoulder_fb_norm for r in reps]),
        "mean_down_time": _mean([r.down_time_s for r in reps]),
        "mean_up_time": _mean([r.up_time_s for r in reps]),
    }
    return reps, agg


def summarize(video_path: str, fps: float, reps: List[RepMetrics], agg: Dict[str, float]) -> VideoSummary:
    return VideoSummary(
        video_path=video_path,
        fps=fps,
        reps=reps,
        rep_count=int(agg["rep_count"]),
        mean_knee_angle=float(agg["mean_knee_angle"]),
        mean_shoulder_fb=float(agg["mean_shoulder_fb"]),
        mean_down_time=float(agg["mean_down_time"]),
        mean_up_time=float(agg["mean_up_time"]),
    )
