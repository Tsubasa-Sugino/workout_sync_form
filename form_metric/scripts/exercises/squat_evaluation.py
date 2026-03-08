import argparse
import math
import os
import sys
from typing import List, Optional, Tuple


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from .squat_detection import detect_low_knee_segments


L_SHOULDER, R_SHOULDER = 11, 12
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


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def _frame_features(pose: Optional[list]) -> Optional[dict]:
    if pose is None:
        return None

    ls, rs = _xy(pose[L_SHOULDER]), _xy(pose[R_SHOULDER])
    lh, rh = _xy(pose[L_HIP]), _xy(pose[R_HIP])
    lk, rk = _xy(pose[L_KNEE]), _xy(pose[R_KNEE])
    la, ra = _xy(pose[L_ANKLE]), _xy(pose[R_ANKLE])

    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)
    torso_len = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]) + 1e-8

    knee_l = _angle_deg(lh, lk, la)
    knee_r = _angle_deg(rh, rk, ra)
    knee_vals = []
    if knee_l is not None:
        knee_vals.append(knee_l)
    if knee_r is not None:
        knee_vals.append(knee_r)
    if not knee_vals:
        return None

    knee_angle = float(sum(knee_vals) / len(knee_vals))
    shoulder_fb_norm = (shoulder_mid[0] - hip_mid[0]) / torso_len
    hip_y = hip_mid[1]  # image y increases downward

    return {
        "hip_y": hip_y,
        "knee_angle_deg": knee_angle,
        "shoulder_fb_norm": shoulder_fb_norm,
    }


def _segment_visibility_mean(poses: List[Optional[list]], start: int, end: int) -> float:
    vals: List[float] = []
    s = max(0, int(start))
    e = min(len(poses) - 1, int(end))
    for i in range(s, e + 1):
        pose = poses[i]
        if pose is None:
            continue
        for lm in pose:
            vals.append(float(lm.visibility))
    if len(vals) == 0:
        return float("-inf")
    return float(sum(vals) / len(vals))


def _segment_metrics(
    poses: List[Optional[list]],
    fps: float,
    seg_start: int,
    seg_end: int,
) -> dict:
    s = max(0, int(seg_start))
    e = min(len(poses) - 1, int(seg_end))

    items: List[Tuple[int, dict]] = []
    for i in range(s, e + 1):
        f = _frame_features(poses[i])
        if f is None:
            continue
        items.append((i, f))

    if len(items) == 0:
        return {
            "knee_angle_deg": float("nan"),
            "shoulder_fb_norm": float("nan"),
            "down_time_s": float("nan"),
            "up_time_s": float("nan"),
            "valid_frames": 0,
        }

    bottom_idx, bottom_feat = max(items, key=lambda x: x[1]["hip_y"])

    return {
        "knee_angle_deg": float(bottom_feat["knee_angle_deg"]),
        "shoulder_fb_norm": float(bottom_feat["shoulder_fb_norm"]),
        "down_time_s": float((bottom_idx - s) / fps),
        "up_time_s": float((e - bottom_idx) / fps),
        "valid_frames": len(items),
    }


def evaluate(template_video: str, target_video: str, args: argparse.Namespace) -> None:
    from src.pose_runner import PoseLandmarkerRunner

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    runner = PoseLandmarkerRunner(model_path=args.model, num_poses=1)

    tpl_fps, tpl_poses = runner.iter_video_landmarks(
        template_video,
        visibility_th=args.visibility_th,
        min_visible_keypoints=args.min_visible_keypoints,
    )
    tgt_fps, tgt_poses = runner.iter_video_landmarks(
        target_video,
        visibility_th=args.visibility_th,
        min_visible_keypoints=args.min_visible_keypoints,
    )

    tpl_segments = detect_low_knee_segments(
        poses=tpl_poses,
        knee_threshold_deg=args.knee_threshold_deg,
        min_low_knee_frames=args.min_low_knee_frames,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        merge_gap_frames=args.merge_gap_frames,
        ema_alpha=args.ema_alpha,
    )
    if len(tpl_segments) == 0:
        raise ValueError("Template video: squat segment was not detected by knee-angle rule.")

    tpl_best = max(
        tpl_segments,
        key=lambda seg: _segment_visibility_mean(tpl_poses, seg[0], seg[1]),
    )
    tpl_best_vis = _segment_visibility_mean(tpl_poses, tpl_best[0], tpl_best[1])
    tpl = _segment_metrics(
        poses=tpl_poses,
        fps=tpl_fps,
        seg_start=tpl_best[0],
        seg_end=tpl_best[1],
    )

    tgt_segments = detect_low_knee_segments(
        poses=tgt_poses,
        knee_threshold_deg=args.knee_threshold_deg,
        min_low_knee_frames=args.min_low_knee_frames,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        merge_gap_frames=args.merge_gap_frames,
        ema_alpha=args.ema_alpha,
    )

    print("==== Squat compare (3 metrics) ====")
    print(f"Template segments: {len(tpl_segments)}")
    print(f"Template selected segment: [{tpl_best[0]}:{tpl_best[1]}], mean visibility={tpl_best_vis:.4f}")
    print(f"Template valid frames: {tpl['valid_frames']}")
    print(f"Target segments: {len(tgt_segments)}")
    print()

    if len(tgt_segments) == 0:
        print("No target squat segments were detected by knee-angle rule.")
        return

    for i, (s, e) in enumerate(tgt_segments, start=1):
        tgt_vis = _segment_visibility_mean(tgt_poses, s, e)
        tgt = _segment_metrics(
            poses=tgt_poses,
            fps=tgt_fps,
            seg_start=s,
            seg_end=e,
        )

        print(f"---- Target Segment {i} [{s}:{e}] ----")
        print(f"Visibility mean: {tgt_vis:.4f}")
        print(f"Target valid frames: {tgt['valid_frames']}")
        print("Metrics (@ segment bottom, diff = target - template)")
        print(f"Knee angle (deg):   tpl={tpl['knee_angle_deg']:.2f}  tgt={tgt['knee_angle_deg']:.2f}  diff={tgt['knee_angle_deg'] - tpl['knee_angle_deg']:+.2f}")
        print(f"Shoulder F/B(norm): tpl={tpl['shoulder_fb_norm']:+.4f} tgt={tgt['shoulder_fb_norm']:+.4f} diff={tgt['shoulder_fb_norm'] - tpl['shoulder_fb_norm']:+.4f}")
        print(f"Down time (s):      tpl={tpl['down_time_s']:.2f}  tgt={tgt['down_time_s']:.2f}  diff={tgt['down_time_s'] - tpl['down_time_s']:+.2f}")
        print(f"Up time (s):        tpl={tpl['up_time_s']:.2f}    tgt={tgt['up_time_s']:.2f}    diff={tgt['up_time_s'] - tpl['up_time_s']:+.2f}")
        print()

    print("Notes:")
    print("- Knee angle: 小さいほど深くしゃがめている傾向（ただし流儀・個体差あり）")
    print("- Shoulder F/B(norm): + 方向 = 肩が股関節より前に出やすい（前傾・潰れの指標になりやすい）")
    print("- Tempo: down/up は検出セグメントの開始/終了から最下点までの秒数")
