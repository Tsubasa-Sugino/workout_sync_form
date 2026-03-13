import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16


def _xy(lm) -> Tuple[float, float]:
    return float(lm.x), float(lm.y)


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def _frame_features(pose: Optional[list]) -> Optional[dict]:
    if pose is None:
        return None

    lw = _xy(pose[L_WRIST])
    rw = _xy(pose[R_WRIST])
    ls = _xy(pose[L_SHOULDER])
    rs = _xy(pose[R_SHOULDER])

    wrist_mid = _mid(lw, rw)
    shoulder_mid = _mid(ls, rs)
    shoulder_width = math.hypot(ls[0] - rs[0], ls[1] - rs[1]) + 1e-8
    wrist_dist = math.hypot(lw[0] - rw[0], lw[1] - rw[1])

    # カメラの上下位置ずれの影響を抑えるため、肩中点に対する手首の相対yを使う
    bar_y_rel = wrist_mid[1] - shoulder_mid[1]
    grip_width_norm = wrist_dist / shoulder_width

    return {
        "bar_y_rel": float(bar_y_rel),
        "grip_width_norm": float(grip_width_norm),
    }


def _segment_metrics(poses: List[Optional[list]], fps: float) -> dict:
    if len(poses) == 0:
        return {
            "wrist_vertical_range": float("nan"),
            "grip_width_norm": float("nan"),
            "down_time_s": float("nan"),
            "up_time_s": float("nan"),
            "valid_frames": 0,
        }

    items: List[Tuple[int, dict]] = []
    for i, pose in enumerate(poses):
        feat = _frame_features(pose)
        if feat is None:
            continue
        items.append((i, feat))

    if len(items) == 0:
        return {
            "wrist_vertical_range": float("nan"),
            "grip_width_norm": float("nan"),
            "down_time_s": float("nan"),
            "up_time_s": float("nan"),
            "valid_frames": 0,
        }

    first_idx = items[0][0]
    last_idx = items[-1][0]

    bar_seq = [it[1]["bar_y_rel"] for it in items]
    grip_seq = [it[1]["grip_width_norm"] for it in items]

    vertical_range = float(max(bar_seq) - min(bar_seq))
    grip_width_mean = float(np.mean(grip_seq))

    bottom_idx = max(items, key=lambda x: x[1]["bar_y_rel"])[0]
    down_time_s = float(max(0, bottom_idx - first_idx) / fps)
    up_time_s = float(max(0, last_idx - bottom_idx) / fps)

    return {
        "wrist_vertical_range": vertical_range,
        "grip_width_norm": grip_width_mean,
        "down_time_s": down_time_s,
        "up_time_s": up_time_s,
        "valid_frames": len(items),
    }


def _metric_similarity(template_value: float, target_value: float, tolerance: float) -> float:
    if math.isnan(template_value) or math.isnan(target_value):
        return 0.0
    if tolerance <= 1e-8:
        return 0.0
    score = 100.0 * (1.0 - abs(target_value - template_value) / tolerance)
    return max(0.0, min(100.0, float(score)))


def _benchpress_match_score_percent(template_metrics: dict, target_metrics: dict) -> Dict[str, float]:
    range_percent = _metric_similarity(
        template_metrics["wrist_vertical_range"],
        target_metrics["wrist_vertical_range"],
        tolerance=0.10,
    )
    grip_percent = _metric_similarity(
        template_metrics["grip_width_norm"],
        target_metrics["grip_width_norm"],
        tolerance=0.20,
    )
    tempo_down_percent = _metric_similarity(
        template_metrics["down_time_s"],
        target_metrics["down_time_s"],
        tolerance=0.7,
    )
    tempo_up_percent = _metric_similarity(
        template_metrics["up_time_s"],
        target_metrics["up_time_s"],
        tolerance=0.7,
    )
    tempo_percent = float(np.mean([tempo_down_percent, tempo_up_percent]))
    overall_percent = float(np.mean([range_percent, grip_percent, tempo_percent]))

    return {
        "range_percent": range_percent,
        "grip_percent": grip_percent,
        "tempo_percent": tempo_percent,
        "tempo_down_percent": tempo_down_percent,
        "tempo_up_percent": tempo_up_percent,
        "overall_percent": overall_percent,
    }


def _ensure_unique_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    i = 2
    while True:
        cand = f"{base_dir}_{i:02d}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=True)
            return cand
        i += 1


def _safe_relpath(path: str) -> str:
    try:
        return os.path.relpath(path, start=os.getcwd())
    except ValueError:
        return path


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

    tpl_metrics = _segment_metrics(tpl_poses, tpl_fps)
    tgt_metrics = _segment_metrics(tgt_poses, tgt_fps)
    score = _benchpress_match_score_percent(tpl_metrics, tgt_metrics)

    template_stem = Path(template_video).stem
    target_stem = Path(target_video).stem
    base_out_dir = os.path.abspath(args.benchpress_compare_out_dir)
    os.makedirs(base_out_dir, exist_ok=True)
    run_dir = _ensure_unique_dir(os.path.join(base_out_dir, f"{template_stem}_vs_{target_stem}"))

    print("==== Benchpress compare ====")
    print(f"Template valid frames: {tpl_metrics['valid_frames']}")
    print(f"Target valid frames: {tgt_metrics['valid_frames']}")
    print(f"一致度: {score['overall_percent']:.1f}%")
    print(
        "項目別一致度(%): "
        f"上下可動域={score['range_percent']:.1f}, "
        f"手幅={score['grip_percent']:.1f}, "
        f"上下テンポ={score['tempo_percent']:.1f}"
    )
    print(
        f"  (テンポ内訳: down={score['tempo_down_percent']:.1f}, up={score['tempo_up_percent']:.1f})"
    )

    report = {
        "exercise": "benchpress",
        "template_video": _safe_relpath(template_video),
        "target_video": _safe_relpath(target_video),
        "match_percent": score["overall_percent"],
        "metric_match_percent": {
            "range_percent": score["range_percent"],
            "grip_percent": score["grip_percent"],
            "tempo_percent": score["tempo_percent"],
            "tempo_down_percent": score["tempo_down_percent"],
            "tempo_up_percent": score["tempo_up_percent"],
        },
        "template_metrics": tpl_metrics,
        "target_metrics": tgt_metrics,
    }

    report_path = os.path.join(run_dir, "result.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report: {report_path}")
