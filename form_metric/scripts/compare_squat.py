import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from args import build_compare_squat_parser
from exercises.squat_evaluation import evaluate as evaluate_squat


def main() -> None:
    parser = build_compare_squat_parser()
    args = parser.parse_args()

    evaluate_squat(
        template_video=args.template,
        target_video=args.target,
        args=args,
    )


if __name__ == "__main__":
    main()
