import os
import argparse
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pose_runner import PoseLandmarkerRunner
from src.squat_metrics import analyze_squat, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pose_landmarker_heavy.task")
    parser.add_argument("--template", required=True, help="お手本スクワット動画")
    parser.add_argument("--target", required=True, help="比較したいスクワット動画")
    parser.add_argument("--visibility_th", type=float, default=0.2)
    parser.add_argument("--min_visible_keypoints", type=int, default=4, help="主要8関節中、可視判定を満たす必要本数")
    parser.add_argument("--min_rep_sec", type=float, default=0.6, help="1レップ最短秒数")
    parser.add_argument("--prominence", type=float, default=None, help="hip_yピークの突出量。未指定時は自動推定")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    runner = PoseLandmarkerRunner(model_path=args.model, num_poses=1)

    tpl_fps, tpl_poses = runner.iter_video_landmarks(
        args.template,
        visibility_th=args.visibility_th,
        min_visible_keypoints=args.min_visible_keypoints,
    )
    tgt_fps, tgt_poses = runner.iter_video_landmarks(
        args.target,
        visibility_th=args.visibility_th,
        min_visible_keypoints=args.min_visible_keypoints,
    )

    tpl_reps, tpl_agg = analyze_squat(
        tpl_poses,
        tpl_fps,
        min_rep_sec=args.min_rep_sec,
        prominence=args.prominence,
    )
    tgt_reps, tgt_agg = analyze_squat(
        tgt_poses,
        tgt_fps,
        min_rep_sec=args.min_rep_sec,
        prominence=args.prominence,
    )

    tpl = summarize(args.template, tpl_fps, tpl_reps, tpl_agg)
    tgt = summarize(args.target, tgt_fps, tgt_reps, tgt_agg)

    print("==== Squat compare (3 metrics) ====")
    print(f"Template: reps={tpl.rep_count} fps={tpl.fps:.2f}  {tpl.video_path}")
    print(f"Target  : reps={tgt.rep_count} fps={tgt.fps:.2f}  {tgt.video_path}")
    print()

    # diff = target - template
    print("---- Metrics (mean over reps @ bottom) ----")
    print(f"Knee angle (deg):   tpl={tpl.mean_knee_angle:.2f}  tgt={tgt.mean_knee_angle:.2f}  diff={tgt.mean_knee_angle - tpl.mean_knee_angle:+.2f}")
    print(f"Shoulder F/B(norm): tpl={tpl.mean_shoulder_fb:+.4f} tgt={tgt.mean_shoulder_fb:+.4f} diff={tgt.mean_shoulder_fb - tpl.mean_shoulder_fb:+.4f}")
    print(f"Down time (s):      tpl={tpl.mean_down_time:.2f}  tgt={tgt.mean_down_time:.2f}  diff={tgt.mean_down_time - tpl.mean_down_time:+.2f}")
    print(f"Up time (s):        tpl={tpl.mean_up_time:.2f}    tgt={tgt.mean_up_time:.2f}    diff={tgt.mean_up_time - tpl.mean_up_time:+.2f}")
    print()
    print("Notes:")
    print("- Knee angle: 小さいほど深くしゃがめている傾向（ただし流儀・個体差あり）")
    print("- Shoulder F/B(norm): + 方向 = 肩が股関節より前に出やすい（前傾・潰れの指標になりやすい）")
    print("- Tempo: down/up は1レップの下降/上昇にかかった秒数")


if __name__ == "__main__":
    main()
