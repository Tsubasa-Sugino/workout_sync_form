import argparse
import importlib
import os
import sys
from typing import Callable, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from args import build_main_parser


def _build_tasks(
    exercises: List[str],
    template_videos: List[str],
    target_videos: List[str],
) -> List[Tuple[str, str, str]]:
    if not (len(exercises) == len(template_videos) == len(target_videos)):
        raise ValueError(
            "The number of --task, --template, and --target values must match."
        )

    tasks: List[Tuple[str, str, str]] = []
    for exercise, template_video, target_video in zip(
        exercises, template_videos, target_videos
    ):
        ex = exercise.strip().lower()
        tpl = template_video.strip()
        tgt = target_video.strip()
        if not ex or not tpl or not tgt:
            raise ValueError(
                "--task, --template, and --target must all be non-empty."
            )
        tasks.append((ex, tpl, tgt))
    return tasks


def _load_evaluator(exercise: str) -> Callable[[str, str, argparse.Namespace], None]:
    module_name = f"exercises.{exercise}_evaluation"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise NotImplementedError(
                f"Exercise '{exercise}' is not implemented yet. Missing module: {module_name}.py"
            ) from exc
        raise

    evaluate = getattr(module, "evaluate", None)
    if evaluate is None:
        raise AttributeError(f"{module_name}.evaluate is required")
    return evaluate


def main() -> None:
    parser = build_main_parser()
    args = parser.parse_args()

    tasks = _build_tasks(args.task, args.template, args.target)

    for i, (exercise, template_video, target_video) in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] exercise={exercise}")
        evaluator = _load_evaluator(exercise)
        evaluator(template_video=template_video, target_video=target_video, args=args)
        if i < len(tasks):
            print()


if __name__ == "__main__":
    main()
