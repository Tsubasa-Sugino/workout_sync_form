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


EXERCISE_ALIASES = {
    "squad": "squat",
    "bench": "benchpress",
    "bench_press": "benchpress",
    "bench-press": "benchpress",
    "bentipress": "benchpress",
}


def _build_tasks(
    exercises: List[str],
    template_videos: List[str],
    target_videos: List[str],
    args: argparse.Namespace,
) -> List[Tuple[str, str, str]]:
    norm_exercises = [EXERCISE_ALIASES.get(ex.strip().lower(), ex.strip().lower()) for ex in exercises]
    templates = [tpl.strip() for tpl in template_videos]
    targets = [tgt.strip() for tgt in target_videos]

    def allows_one_template_many_targets(exercise: str) -> bool:
        if exercise == "benchpress":
            return True
        if exercise == "squat" and getattr(args, "squat_eval_mode", "auto") == "manual":
            return True
        if exercise == "deadlift" and getattr(args, "deadlift_eval_mode", "auto") == "manual":
            return True
        return False

    # 対応モード時のみ、template 1本 + target 複数本を許可する
    if (
        len(norm_exercises) == 1
        and len(templates) == 1
        and len(targets) >= 1
        and allows_one_template_many_targets(norm_exercises[0])
    ):
        ex = norm_exercises[0]
        tpl = templates[0]
        if not ex or not tpl:
            raise ValueError("--task and --template must be non-empty.")
        tasks = []
        for tgt in targets:
            if not tgt:
                raise ValueError("--target must be non-empty.")
            tasks.append((ex, tpl, tgt))
        return tasks

    if not (len(norm_exercises) == len(templates) == len(targets)):
        raise ValueError(
            "The number of --task, --template, and --target values must match "
            "(except supported modes: 1 template with multiple targets is allowed)."
        )

    tasks: List[Tuple[str, str, str]] = []
    for ex, tpl, tgt in zip(norm_exercises, templates, targets):
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

    tasks = _build_tasks(args.task, args.template, args.target, args)

    for i, (exercise, template_video, target_video) in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] exercise={exercise}")
        evaluator = _load_evaluator(exercise)
        evaluator(template_video=template_video, target_video=target_video, args=args)
        if i < len(tasks):
            print()


if __name__ == "__main__":
    main()
