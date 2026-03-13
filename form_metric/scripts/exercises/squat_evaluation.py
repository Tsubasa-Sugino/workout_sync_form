import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dtaidistance import dtw
from sklearn.decomposition import PCA


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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
    knee_vals: List[float] = []
    if knee_l is not None:
        knee_vals.append(knee_l)
    if knee_r is not None:
        knee_vals.append(knee_r)
    if not knee_vals:
        return None

    return {
        "hip_y": hip_mid[1],
        "knee_angle_deg": float(sum(knee_vals) / len(knee_vals)),
        "shoulder_fb_norm": (shoulder_mid[0] - hip_mid[0]) / torso_len,
    }


def _segment_metrics(
    poses: List[Optional[list]],
    fps: float,
    seg_start: int,
    seg_end: int,
) -> dict:
    if len(poses) == 0:
        return {
            "knee_angle_deg": float("nan"),
            "shoulder_fb_norm": float("nan"),
            "down_time_s": float("nan"),
            "up_time_s": float("nan"),
            "valid_frames": 0,
        }

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

    # 膝評価は「90度に最も近い」フレームを基準にする
    _, knee_ref_feat = min(
        items,
        key=lambda x: abs(float(x[1]["knee_angle_deg"]) - 90.0),
    )
    bottom_idx, bottom_feat = max(items, key=lambda x: x[1]["hip_y"])
    return {
        "knee_angle_deg": float(knee_ref_feat["knee_angle_deg"]),
        "shoulder_fb_norm": float(bottom_feat["shoulder_fb_norm"]),
        "down_time_s": float((bottom_idx - s) / fps),
        "up_time_s": float((e - bottom_idx) / fps),
        "valid_frames": len(items),
    }


def _metric_similarity(template_value: float, target_value: float, tolerance: float) -> float:
    if math.isnan(template_value) or math.isnan(target_value):
        return 0.0
    if tolerance <= 1e-8:
        return 0.0
    score = 100.0 * (1.0 - abs(target_value - template_value) / tolerance)
    return max(0.0, min(100.0, float(score)))


def _match_score_percent(template_metrics: dict, target_metrics: dict) -> Dict[str, float]:
    knee_percent = _metric_similarity(
        template_metrics["knee_angle_deg"],
        target_metrics["knee_angle_deg"],
        tolerance=20.0,
    )
    shoulder_percent = _metric_similarity(
        template_metrics["shoulder_fb_norm"],
        target_metrics["shoulder_fb_norm"],
        tolerance=0.25,
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
    overall_percent = float(np.mean([knee_percent, shoulder_percent, tempo_percent]))

    return {
        "knee_percent": knee_percent,
        "shoulder_percent": shoulder_percent,
        "tempo_percent": tempo_percent,
        "tempo_down_percent": tempo_down_percent,
        "tempo_up_percent": tempo_up_percent,
        "overall_percent": overall_percent,
    }


def _pose_to_xyz(pose: Optional[list]) -> Optional[np.ndarray]:
    if pose is None:
        return None
    coords = np.zeros((33, 3), dtype=np.float64)
    for i, lm in enumerate(pose):
        coords[i, 0] = float(lm.x)
        coords[i, 1] = float(lm.y)
        coords[i, 2] = float(lm.z)
    return coords


def _normalize_pose_coords(coords: np.ndarray) -> np.ndarray:
    out = coords.astype(np.float64).copy()

    hip_center = (out[L_HIP] + out[R_HIP]) * 0.5
    out -= hip_center

    shoulder_center = (out[L_SHOULDER] + out[R_SHOULDER]) * 0.5
    scale = float(np.linalg.norm(shoulder_center))
    if scale > 1e-8:
        out /= scale

    hip_vec = out[R_HIP] - out[L_HIP]
    angle_y = math.atan2(float(hip_vec[2]), float(hip_vec[0]))
    cy, sy = math.cos(-angle_y), math.sin(-angle_y)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    out = out @ rot_y.T

    shoulder_center_after = (out[L_SHOULDER] + out[R_SHOULDER]) * 0.5
    angle_z = math.atan2(float(shoulder_center_after[0]), float(-shoulder_center_after[1]))
    cz, sz = math.cos(-angle_z), math.sin(-angle_z)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out = out @ rot_z.T

    return out


def _poses_to_matrix(poses: List[Optional[list]]) -> np.ndarray:
    if len(poses) == 0:
        return np.empty((0, 99), dtype=np.float64)

    rows: List[np.ndarray] = []
    last_valid: Optional[np.ndarray] = None
    for pose in poses:
        coords = _pose_to_xyz(pose)
        if coords is None:
            if last_valid is None:
                coords = np.zeros((33, 3), dtype=np.float64)
            else:
                coords = last_valid.copy()
        else:
            last_valid = coords.copy()

        normalized = _normalize_pose_coords(coords)
        rows.append(normalized.reshape(-1))

    return np.stack(rows, axis=0)


def _extract_template_window(pc1: np.ndarray) -> Tuple[int, int]:
    n = len(pc1)
    if n <= 4:
        return 0, max(0, n - 1)

    med = float(np.median(pc1))
    deviation = np.abs(pc1 - med)
    bottom = int(np.argmax(deviation))
    if deviation[bottom] < 1e-8:
        return 0, n - 1

    sign = 1.0 if pc1[bottom] >= med else -1.0
    start = 0
    end = n - 1

    for i in range(bottom - 1, -1, -1):
        if (pc1[i] - med) * sign <= 0.0:
            start = i
            break

    for i in range(bottom + 1, n):
        if (pc1[i] - med) * sign <= 0.0:
            end = i
            break

    if end - start + 1 < max(8, int(0.3 * n)):
        return 0, n - 1

    pad = max(2, int(0.05 * n))
    return max(0, start - pad), min(n - 1, end + pad)


def _normalize_wave(w: np.ndarray) -> np.ndarray:
    mean = float(np.mean(w))
    std = float(np.std(w))
    if std < 1e-8:
        return w - mean
    return (w - mean) / std


def _auto_dtw_threshold(all_distances: List[Tuple[int, float]]) -> float:
    vals = np.array([d for _, d in all_distances], dtype=np.float64)
    if len(vals) == 0:
        return float("inf")
    best = float(np.min(vals))
    median = float(np.median(vals))
    if median <= best:
        return best
    return best + 0.45 * (median - best)


def _overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    s1, e1 = int(a["start"]), int(a["end"])
    s2, e2 = int(b["start"]), int(b["end"])
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    if inter <= 0:
        return 0.0
    la = e1 - s1 + 1
    lb = e2 - s2 + 1
    return float(inter / max(1, min(la, lb)))


def _search_dtw_matches(
    template_wave: np.ndarray,
    target_wave: np.ndarray,
    step_size: int,
    threshold: float,
) -> Tuple[List[Dict[str, float]], List[Tuple[int, float]], float]:
    w = _normalize_wave(template_wave)
    m = len(w)
    n = len(target_wave)
    if m <= 0 or n < m:
        return [], [], threshold

    step = max(1, int(step_size))
    all_distances: List[Tuple[int, float]] = []
    for i in range(0, n - m + 1, step):
        window = _normalize_wave(target_wave[i : i + m])
        dist = float(dtw.distance(w, window))
        all_distances.append((i, dist))

    if len(all_distances) == 0:
        return [], [], threshold

    effective_threshold = float(threshold)
    if effective_threshold <= 0.0:
        effective_threshold = _auto_dtw_threshold(all_distances)

    candidates: List[Dict[str, float]] = []
    for idx, (start, dist) in enumerate(all_distances):
        prev_d = all_distances[idx - 1][1] if idx > 0 else float("inf")
        next_d = all_distances[idx + 1][1] if idx + 1 < len(all_distances) else float("inf")
        if dist <= prev_d and dist <= next_d and dist <= effective_threshold:
            candidates.append(
                {
                    "start": int(start),
                    "end": int(start + m - 1),
                    "distance": float(dist),
                }
            )

    if len(candidates) == 0:
        best_start, best_dist = min(all_distances, key=lambda x: x[1])
        candidates = [
            {
                "start": int(best_start),
                "end": int(best_start + m - 1),
                "distance": float(best_dist),
            }
        ]

    chosen: List[Dict[str, float]] = []
    for cand in sorted(candidates, key=lambda x: x["distance"]):
        if any(_overlap_ratio(cand, prev) > 0.45 for prev in chosen):
            continue
        chosen.append(cand)

    chosen = sorted(chosen, key=lambda x: x["start"])
    return chosen, all_distances, effective_threshold


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


def _save_clip(video_path: str, start_frame: int, end_frame: int, out_path: str) -> Optional[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    s = max(0, int(start_frame))
    e = min(frame_count - 1, int(end_frame))
    if e <= s:
        cap.release()
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    to_write = e - s + 1
    written = 0
    while written < to_write:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        return None
    return out_path


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

    tpl_matrix = _poses_to_matrix(tpl_poses)
    tgt_matrix = _poses_to_matrix(tgt_poses)
    if len(tpl_matrix) < 8:
        raise ValueError("Template video is too short for PCA rep matching.")
    if len(tgt_matrix) < len(tpl_matrix):
        raise ValueError("Target video must be longer than template video for PCA rep matching.")

    pca = PCA(n_components=1)
    tpl_pc1 = pca.fit_transform(tpl_matrix)[:, 0]
    tgt_pc1 = pca.transform(tgt_matrix)[:, 0]

    tpl_start, tpl_end = _extract_template_window(tpl_pc1)
    template_wave = tpl_pc1[tpl_start : tpl_end + 1]
    if len(template_wave) < 8:
        raise ValueError("Template PCA waveform is too short after trimming.")

    matches, all_distances, used_threshold = _search_dtw_matches(
        template_wave=template_wave,
        target_wave=tgt_pc1,
        step_size=args.squat_pca_step_size,
        threshold=args.squat_pca_dtw_threshold,
    )

    tpl_metrics = _segment_metrics(
        poses=tpl_poses,
        fps=tpl_fps,
        seg_start=tpl_start,
        seg_end=tpl_end,
    )

    template_stem = Path(template_video).stem
    target_stem = Path(target_video).stem
    base_out_dir = os.path.abspath(args.squat_compare_out_dir)
    os.makedirs(base_out_dir, exist_ok=True)
    run_dir = _ensure_unique_dir(os.path.join(base_out_dir, f"{template_stem}_vs_{target_stem}"))

    template_clip_path = _save_clip(
        video_path=template_video,
        start_frame=tpl_start,
        end_frame=tpl_end,
        out_path=os.path.join(run_dir, f"{template_stem}_template.mp4"),
    )

    print("==== Squat compare (PCA waveform + DTW) ====")
    print(f"Template PCA window: [{tpl_start}:{tpl_end}] (frames={tpl_end - tpl_start + 1})")
    print(f"Target matches: {len(matches)}")
    print(f"DTW threshold used: {used_threshold:.4f}")
    print(f"Output dir: {run_dir}")
    print()

    if len(matches) == 0:
        print("No target squat segments were detected by PCA waveform matching.")
        return

    best_dist = min((m["distance"] for m in matches), default=0.0)
    match_records: List[dict] = []

    for i, match in enumerate(matches, start=1):
        s = int(match["start"])
        e = int(match["end"])
        dist = float(match["distance"])
        tgt = _segment_metrics(
            poses=tgt_poses,
            fps=tgt_fps,
            seg_start=s,
            seg_end=e,
        )
        score = _match_score_percent(tpl_metrics, tgt)

        clip_path = _save_clip(
            video_path=target_video,
            start_frame=s,
            end_frame=e,
            out_path=os.path.join(run_dir, f"{target_stem}_rep_{i:02d}.mp4"),
        )

        print(f"---- Target Rep {i} [{s}:{e}] ----")
        print(f"DTW distance: {dist:.4f}")
        print(f"一致度: {score['overall_percent']:.1f}%")
        print(
            "項目別一致度(%): "
            f"膝の曲げ具合={score['knee_percent']:.1f}, "
            f"肩の前後={score['shoulder_percent']:.1f}, "
            f"上下テンポ={score['tempo_percent']:.1f}"
        )
        print(
            f"  (テンポ内訳: down={score['tempo_down_percent']:.1f}, up={score['tempo_up_percent']:.1f})"
        )
        print()

        match_records.append(
            {
                "rep_index": i,
                "target_segment": {"start": s, "end": e},
                "dtw_distance": dist,
                "dtw_relative_to_best": float(dist / best_dist) if best_dist > 1e-8 else 1.0,
                "match_percent": score["overall_percent"],
                "metric_match_percent": {
                    "knee_percent": score["knee_percent"],
                    "shoulder_percent": score["shoulder_percent"],
                    "tempo_percent": score["tempo_percent"],
                    "tempo_down_percent": score["tempo_down_percent"],
                    "tempo_up_percent": score["tempo_up_percent"],
                },
                "template_metrics": tpl_metrics,
                "target_metrics": tgt,
                "target_clip_path": _safe_relpath(clip_path) if clip_path else None,
            }
        )

    report = {
        "exercise": "squat",
        "template_video": _safe_relpath(template_video),
        "target_video": _safe_relpath(target_video),
        "template_segment": {
            "start": tpl_start,
            "end": tpl_end,
            "template_clip_path": _safe_relpath(template_clip_path) if template_clip_path else None,
        },
        "dtw_config": {
            "step_size": int(args.squat_pca_step_size),
            "threshold_input": float(args.squat_pca_dtw_threshold),
            "threshold_used": float(used_threshold),
            "num_windows": len(all_distances),
        },
        "matches": match_records,
    }

    report_path = os.path.join(run_dir, "result.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report: {report_path}")
