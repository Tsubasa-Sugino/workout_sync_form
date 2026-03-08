import argparse


def add_common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="models/pose_landmarker_heavy.task")
    parser.add_argument("--visibility_th", type=float, default=0.2)
    parser.add_argument(
        "--min_visible_keypoints",
        type=int,
        default=4,
        help="主要8関節中、可視判定を満たす必要本数",
    )


def add_squat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--knee_threshold_deg",
        type=float,
        default=100.0,
        help="膝角がこの値未満の区間をスクワット区間として採用",
    )
    parser.add_argument(
        "--min_low_knee_frames",
        type=int,
        default=4,
        help="膝角が閾値未満で連続する最小フレーム数",
    )
    parser.add_argument(
        "--pre_frames",
        type=int,
        default=12,
        help="検出区間の前に足すフレーム数",
    )
    parser.add_argument(
        "--post_frames",
        type=int,
        default=12,
        help="検出区間の後に足すフレーム数",
    )
    parser.add_argument(
        "--merge_gap_frames",
        type=int,
        default=10,
        help="近い検出区間を結合する最大ギャップ",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.25,
        help="膝角平滑化のEMA係数",
    )


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate form for one or more exercises via a single entrypoint.",
    )
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        metavar="EXERCISE",
        help="評価種目。複数指定可能（--template/--target と同じ回数指定）。例: squat",
    )
    parser.add_argument(
        "--template",
        action="append",
        required=True,
        metavar="VIDEO_PATH",
        help="テンプレート動画。複数指定可能（--task/--target と同じ回数指定）。",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="VIDEO_PATH",
        help="ターゲット動画。複数指定可能（--task/--template と同じ回数指定）。",
    )
    add_common_runtime_arguments(parser)
    add_squat_arguments(parser)
    return parser


def build_compare_squat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="お手本スクワット動画")
    parser.add_argument("--target", required=True, help="比較したいスクワット動画")
    add_common_runtime_arguments(parser)
    add_squat_arguments(parser)
    return parser

