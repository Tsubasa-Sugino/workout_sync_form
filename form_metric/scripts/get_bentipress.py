import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pose_runner import PoseLandmarkerRunner
from src.rep_detection import detect_reps_from_hipy, ema


L_WRIST, R_WRIST = 15, 16


def _xy(lm) -> Tuple[float, float]:
    return float(lm.x), float(lm.y)


def _frame_mean_wrist_y(pose: Optional[list]) -> Optional[float]:
    if pose is None:
        return None

    vals = []
    lw = _xy(pose[L_WRIST])[1]
    rw = _xy(pose[R_WRIST])[1]
    vals.append(lw)
    vals.append(rw)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _build_wrist_series(wrist_y_raw: List[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wrist_y_raw)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    valid = np.array([v is not None for v in wrist_y_raw], dtype=bool)
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=bool)

    x = np.empty(n, dtype=float)
    first_valid = next(float(v) for v in wrist_y_raw if v is not None)
    last = first_valid
    for i, v in enumerate(wrist_y_raw):
        if v is None:
            x[i] = last
        else:
            last = float(v)
            x[i] = last
    return x, valid


def _estimate_prominence(wrist_y_smooth: np.ndarray, valid_mask: np.ndarray) -> float:
    v = wrist_y_smooth[valid_mask]
    wrist_range = float(np.max(v) - np.min(v))
    return min(0.03, max(0.003, wrist_range * 0.2))


def _segment_wrist_range(
    wrist_y_smooth: np.ndarray,
    start: int,
    end: int,
) -> float:
    s = max(0, int(start))
    e = min(len(wrist_y_smooth) - 1, int(end))
    if e <= s:
        return 0.0
    seg = wrist_y_smooth[s:e + 1]
    return float(np.max(seg) - np.min(seg))


def _separate_overlaps(segments: List[Tuple[int, int]], n_frames: int) -> List[Tuple[int, int]]:
    if not segments:
        return []

    segs = sorted((max(0, s), min(n_frames - 1, e)) for s, e in segments if e > s)
    if not segs:
        return []

    out = [list(segs[0])]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        if s <= pe:
            mid = (s + pe) // 2
            out[-1][1] = mid
            s = mid + 1
        if e > s:
            out.append([s, e])
    return [(s, e) for s, e in out if e > s]


def _save_action_clips(
    video_path: str,
    segments: List[Tuple[int, int]],
    output_dir: str,
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video cannot be opened: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    stem = Path(video_path).stem
    saved_paths: List[str] = []

    for i, (seg_start, seg_end) in enumerate(segments, start=1):
        start = max(0, int(seg_start))
        end = min(frame_count - 1, int(seg_end))
        if end <= start:
            continue

        out_path = os.path.join(output_dir, f"{stem}_benchpress_{i:02d}.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output clip: {out_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            writer.release()
            raise FileNotFoundError(f"Video cannot be reopened: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        target_frames = end - start + 1
        written = 0

        while written < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1

        cap.release()
        writer.release()

        if written > 0:
            saved_paths.append(out_path)
        else:
            if os.path.exists(out_path):
                os.remove(out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect bench-press reps by wrist vertical motion and save one clip per rep."
    )
    parser.add_argument("--model", default="models/pose_landmarker_heavy.task")
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="切り抜き対象動画。複数指定可。",
    )
    parser.add_argument("--visibility_th", type=float, default=0.2)
    parser.add_argument(
        "--min_visible_keypoints",
        type=int,
        default=4,
        help="主要8関節中、可視判定を満たす必要本数",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="保存先ディレクトリ（未指定時は <project_root>/video）",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.25,
        help="手首y座標平滑化のEMA係数",
    )
    parser.add_argument(
        "--min_rep_sec",
        type=float,
        default=0.4,
        help="1レップ最小秒数",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.0,
        help="最下点ピークのprominence（<=0で自動推定）",
    )
    parser.add_argument(
        "--min_wrist_vertical_range",
        type=float,
        default=0.015,
        help="1レップ内で必要な手首y変動量（正規化座標）",
    )
    parser.add_argument("--pre_frames", type=int, default=12, help="検出区間の前に足すフレーム数")
    parser.add_argument("--post_frames", type=int, default=12, help="検出区間の後に足すフレーム数")
    parser.add_argument("--min_save_frames", type=int, default=6, help="保存する最小クリップ長")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    project_root = os.path.dirname(ROOT_DIR)
    out_dir = args.out_dir if args.out_dir else os.path.join(project_root, "video")
    os.makedirs(out_dir, exist_ok=True)

    runner = PoseLandmarkerRunner(model_path=args.model, num_poses=1)

    total_saved = 0
    for video_path in args.videos:
        fps, poses = runner.iter_video_landmarks(
            video_path,
            visibility_th=args.visibility_th,
            min_visible_keypoints=args.min_visible_keypoints,
        )

        wrist_raw = [_frame_mean_wrist_y(pose) for pose in poses]
        wrist_y, valid_mask = _build_wrist_series(wrist_raw)
        if len(wrist_y) == 0:
            print(f"[{video_path}] no valid wrist landmarks.")
            continue

        wrist_y_s = ema(wrist_y, alpha=float(args.ema_alpha))

        if float(args.prominence) > 0:
            prominence_eff = float(args.prominence)
        else:
            prominence_eff = _estimate_prominence(wrist_y_s, valid_mask)

        reps_idx = detect_reps_from_hipy(
            hip_y_smooth=wrist_y_s,
            valid_mask=valid_mask,
            fps=float(fps),
            min_rep_sec=float(args.min_rep_sec),
            prominence=prominence_eff,
        )

        pre = max(0, int(args.pre_frames))
        post = max(0, int(args.post_frames))
        min_range = float(args.min_wrist_vertical_range)

        segments: List[Tuple[int, int]] = []
        for s, _, e in reps_idx:
            wrist_range = _segment_wrist_range(wrist_y_s, s, e)
            if wrist_range < min_range:
                continue
            segments.append((max(0, s - pre), min(len(wrist_y_s) - 1, e + post)))

        segments = _separate_overlaps(segments, len(wrist_y_s))
        min_save = max(1, int(args.min_save_frames))
        segments = [(s, e) for s, e in segments if (e - s + 1) >= min_save]

        print(
            f"[{video_path}] detected bench reps by wrist motion: {len(segments)} "
            f"(prominence={prominence_eff:.5f})"
        )
        print(
            f"[{video_path}] wrist y stats: min={float(np.min(wrist_y_s)):.4f}, "
            f"max={float(np.max(wrist_y_s)):.4f}"
        )

        if len(segments) == 0:
            continue

        saved = _save_action_clips(
            video_path=video_path,
            segments=segments,
            output_dir=out_dir,
        )
        total_saved += len(saved)

        for path in saved:
            print(f"saved: {path}")

    print(f"done. total clips saved: {total_saved}")


if __name__ == "__main__":
    main()

