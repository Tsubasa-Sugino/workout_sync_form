import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pose_runner import PoseLandmarkerRunner


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


def _merge_segments(segments: List[Tuple[int, int]], merge_gap_frames: int) -> List[Tuple[int, int]]:
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


def _detect_deadlift_segments(
    knee_angles: List[Optional[float]],
    high_min_deg: float,
    high_max_deg: float,
    min_drop_deg: float,
    min_recovery_deg: float,
    min_rep_frames: int,
    max_rep_frames: int,
    pre_frames: int,
    post_frames: int,
    merge_gap_frames: int,
) -> List[Tuple[int, int]]:
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

            drop = start_angle - min_angle
            if drop >= drop_need:
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


def _save_action_clips(
    video_path: str,
    segments: List[Tuple[int, int]],
    output_dir: str,
    min_save_frames: int,
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

    min_frames = max(1, int(min_save_frames))
    stem = Path(video_path).stem
    saved_paths: List[str] = []

    for i, (seg_start, seg_end) in enumerate(segments, start=1):
        start = max(0, int(seg_start))
        end = min(frame_count - 1, int(seg_end))
        if end <= start or (end - start + 1) < min_frames:
            continue

        out_path = os.path.join(output_dir, f"{stem}_deadlift_{i:02d}.mp4")
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

        if written >= min_frames:
            saved_paths.append(out_path)
        else:
            if os.path.exists(out_path):
                os.remove(out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect deadlift reps by knee-angle pattern and save one clip per rep."
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
    parser.add_argument("--high_min_deg", type=float, default=170.0, help="レップ開始/終了に使う膝角下限")
    parser.add_argument("--high_max_deg", type=float, default=180.0, help="レップ開始/終了に使う膝角上限")
    parser.add_argument("--min_drop_deg", type=float, default=12.0, help="開始角度から必要な最小屈曲量")
    parser.add_argument("--min_recovery_deg", type=float, default=10.0, help="最小角度から必要な最小伸展量")
    parser.add_argument("--min_rep_frames", type=int, default=8, help="1レップ最小フレーム数")
    parser.add_argument("--max_rep_frames", type=int, default=240, help="1レップ最大フレーム数")
    parser.add_argument("--pre_frames", type=int, default=10, help="検出区間の前に足すフレーム数")
    parser.add_argument("--post_frames", type=int, default=10, help="検出区間の後に足すフレーム数")
    parser.add_argument("--merge_gap_frames", type=int, default=0, help="近い検出区間を結合する最大ギャップ")
    parser.add_argument("--ema_alpha", type=float, default=0.25, help="膝角平滑化のEMA係数")
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
        knee_angles_raw = [_frame_knee_angle(pose) for pose in poses]
        knee_angles = _ema_optional(knee_angles_raw, alpha=args.ema_alpha)

        max_rep_frames = int(args.max_rep_frames)
        if max_rep_frames <= 0:
            max_rep_frames = max(int(fps * 6.0), int(args.min_rep_frames))

        segments = _detect_deadlift_segments(
            knee_angles=knee_angles,
            high_min_deg=args.high_min_deg,
            high_max_deg=args.high_max_deg,
            min_drop_deg=args.min_drop_deg,
            min_recovery_deg=args.min_recovery_deg,
            min_rep_frames=args.min_rep_frames,
            max_rep_frames=max_rep_frames,
            pre_frames=args.pre_frames,
            post_frames=args.post_frames,
            merge_gap_frames=args.merge_gap_frames,
        )

        print(
            f"[{video_path}] detected reps(high:{args.high_min_deg:.1f}-{args.high_max_deg:.1f} deg): {len(segments)}"
        )
        if len(segments) == 0:
            continue

        saved = _save_action_clips(
            video_path=video_path,
            segments=segments,
            output_dir=out_dir,
            min_save_frames=args.min_save_frames,
        )
        total_saved += len(saved)

        for path in saved:
            print(f"saved: {path}")

    print(f"done. total clips saved: {total_saved}")


if __name__ == "__main__":
    main()
