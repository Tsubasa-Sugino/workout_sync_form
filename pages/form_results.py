"""
Form Results Page
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import streamlit as st


EXERCISE_TO_TASK = {
    "ベンチプレス": "benchpress",
    "デッドリフト": "deadlift",
    "スクワット": "squat",
}


def get_inputs() -> tuple[
    bytes | None,
    bytes | None,
    str | None,
    str | None,
    str | None,
]:
    """Get page input values from session state."""

    user_video = st.session_state.get("user_video_bytes")
    ideal_video = st.session_state.get("ideal_video_bytes")
    exercise = st.session_state.get("exercise")
    user_video_name = st.session_state.get("user_video_name")
    ideal_video_name = st.session_state.get("ideal_video_name")
    return user_video, ideal_video, exercise, user_video_name, ideal_video_name


def guard_missing_videos(
    user_video: bytes | None,
    ideal_video: bytes | None,
) -> None:
    """Stop page rendering when required videos are missing."""

    if user_video is None or ideal_video is None:
        st.error("動画が未入力です。前の画面で2つの動画を入力してください。")
        if st.button("動画入力に戻る"):
            st.switch_page("pages/upload_videos.py")
        st.stop()


def render_videos(
    user_video: bytes,
    ideal_video: bytes,
    ideal_video_name: str | None,
) -> tuple[bytes, str]:
    """Render uploaded and ideal videos side by side, and return template input."""

    saved_clips: list[dict] = st.session_state.get(
        "saved_clips", [],
    )

    selected_template_video = ideal_video
    selected_template_name = ideal_video_name or "template.mp4"

    left, right = st.columns(2)
    with left:
        st.subheader("自分の動画")
        st.video(
            user_video,
            autoplay=True, loop=True, muted=True,
        )

    with right:
        st.subheader("理想のフォーム動画")
        if saved_clips:
            clip_names = [c["name"] for c in saved_clips if "name" in c]
            selected = st.selectbox(
                "評価に使うお手本動画",
                options=["元のお手本動画", *clip_names],
                key="results_template_select",
            )
            if selected != "元のお手本動画":
                clip = next(
                    (c for c in saved_clips if c.get("name") == selected),
                    None,
                )
                if clip and isinstance(clip.get("bytes"), (bytes, bytearray)):
                    selected_template_video = bytes(clip["bytes"])
                    selected_template_name = str(clip.get("name") or "template.mp4")
                else:
                    st.warning("選択した切り出し動画の読み込みに失敗したため、元動画を使います。")

        st.video(
            selected_template_video,
            autoplay=True, loop=True, muted=True,
        )

    return selected_template_video, selected_template_name


def _write_temp_video(
    temp_dir: Path,
    video_bytes: bytes,
    filename: str,
    prefix: str,
) -> Path:
    """Write bytes to a temp file and return the path."""

    sanitized_name = Path(filename).name if filename else f"{prefix}.mp4"
    suffix = Path(sanitized_name).suffix or ".mp4"
    stem = Path(sanitized_name).stem or prefix
    path = temp_dir / f"{prefix}_{stem}{suffix}"
    path.write_bytes(video_bytes)
    return path


def run_main_evaluation(
    exercise_task: str,
    template_video: bytes,
    template_video_name: str,
    target_video: bytes,
    target_video_name: str,
) -> tuple[bool, str]:
    """Run form_metric/scripts/main.py and return success + output text."""

    project_root = Path(__file__).resolve().parents[1]
    main_script = project_root / "form_metric" / "scripts" / "main.py"
    model_path = project_root / "form_metric" / "models" / "pose_landmarker_heavy.task"

    if not main_script.exists():
        return False, f"評価スクリプトが見つかりません: {main_script}"
    if not model_path.exists():
        return False, f"モデルファイルが見つかりません: {model_path}"

    try:
        with tempfile.TemporaryDirectory(prefix="form_eval_") as temp_dir:
            temp_path = Path(temp_dir)
            template_path = _write_temp_video(
                temp_dir=temp_path,
                video_bytes=template_video,
                filename=template_video_name,
                prefix="template",
            )
            target_path = _write_temp_video(
                temp_dir=temp_path,
                video_bytes=target_video,
                filename=target_video_name,
                prefix="target",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(main_script),
                    "--task", exercise_task,
                    "--template", str(template_path),
                    "--target", str(target_path),
                    "--model", str(model_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(project_root),
            )
    except Exception as exc:
        return False, f"main.py の実行中に例外が発生しました: {exc}"

    outputs: list[str] = []
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        outputs.append(stdout)
    if stderr:
        outputs.append(f"[stderr]\n{stderr}")
    output_text = "\n\n".join(outputs) if outputs else "出力はありませんでした。"

    if completed.returncode != 0:
        return (
            False,
            "main.py が異常終了しました "
            f"(exit code {completed.returncode})\n\n{output_text}",
        )

    return True, output_text


def _build_eval_signature(
    exercise_task: str,
    template_name: str,
    template_video: bytes,
    target_name: str,
    target_video: bytes,
) -> str:
    """Build a simple signature to invalidate stale evaluation output."""

    return "|".join(
        [
            exercise_task,
            template_name,
            str(len(template_video)),
            target_name,
            str(len(target_video)),
        ],
    )


def _extract_score_entries(output_text: str) -> list[dict[str, str | float]]:
    """Extract only overall/metric percentages from main.py stdout text."""

    entries: list[dict[str, str | float]] = []
    current_target = "全体"

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        rep_match = re.match(r"^----\s*Target Rep\s+(\d+)\s*\[.*\]\s*----$", line)
        if rep_match:
            current_target = f"Rep {rep_match.group(1)}"
            continue

        overall_match = re.match(r"^一致度:\s*([0-9]+(?:\.[0-9]+)?)%$", line)
        if overall_match:
            entries.append(
                {
                    "対象": current_target,
                    "一致度(%)": float(overall_match.group(1)),
                    "項目別一致度(%)": "-",
                },
            )
            continue

        metric_match = re.match(r"^項目別一致度\(%\):\s*(.+)$", line)
        if metric_match and entries:
            entries[-1]["項目別一致度(%)"] = metric_match.group(1)

    return entries


def _render_score_only(output_text: str) -> None:
    """Render only match percent and per-metric percent lines."""

    entries = _extract_score_entries(output_text)
    if not entries:
        st.warning("一致度の抽出に失敗しました。")
        return

    if len(entries) == 1:
        entry = entries[0]
        st.metric("一致度", f"{float(entry['一致度(%)']):.1f}%")
        st.write(f"項目別一致度(%): {entry['項目別一致度(%)']}")
        return

    st.table(entries)


def render_result(
    exercise: str | None,
    target_video: bytes,
    target_video_name: str | None,
    template_video: bytes,
    template_video_name: str,
) -> None:
    """Run main.py and render its output text."""

    st.markdown("---")
    st.subheader("フォーム評価結果")

    exercise_task = EXERCISE_TO_TASK.get(exercise or "")
    if exercise_task is None:
        st.error("種目が未選択です。種目選択画面に戻って選択してください。")
        return

    resolved_target_name = target_video_name or "target.mp4"
    current_signature = _build_eval_signature(
        exercise_task=exercise_task,
        template_name=template_video_name,
        template_video=template_video,
        target_name=resolved_target_name,
        target_video=target_video,
    )

    if st.session_state.get("form_eval_signature") != current_signature:
        st.session_state.pop("form_eval_signature", None)
        st.session_state.pop("form_eval_output", None)
        st.session_state.pop("form_eval_success", None)

    if st.button("評価を実行", key="run_form_eval"):
        with st.spinner("フォームを評価しています..."):
            success, output_text = run_main_evaluation(
                exercise_task=exercise_task,
                template_video=template_video,
                template_video_name=template_video_name,
                target_video=target_video,
                target_video_name=resolved_target_name,
            )
        st.session_state.form_eval_signature = current_signature
        st.session_state.form_eval_output = output_text
        st.session_state.form_eval_success = success

    output_text = st.session_state.get("form_eval_output")
    if output_text is None:
        st.info("`評価を実行` を押すと `main.py` の結果テキストを表示します。")
        return

    if st.session_state.get("form_eval_success"):
        st.success("評価が完了しました。")
        _render_score_only(output_text)
    else:
        st.error("評価に失敗しました。ログを確認してください。")
        st.code(output_text)


def render_navigation() -> None:
    """Render navigation buttons."""

    step1, step2 = st.columns(2)
    with step1:
        if st.button("種目選択に戻る"):
            st.switch_page("app.py")
    with step2:
        if st.button("動画入力に戻る"):
            st.switch_page("pages/upload_videos.py")


def main() -> None:
    """Render form results page."""

    st.title("フォーム評価結果")

    (
        user_video,
        ideal_video,
        exercise,
        user_video_name,
        ideal_video_name,
    ) = get_inputs()
    guard_missing_videos(user_video, ideal_video)

    st.write(f"選択種目: **{exercise}**")
    template_video, template_video_name = render_videos(
        user_video,
        ideal_video,
        ideal_video_name,
    )
    render_result(
        exercise=exercise,
        target_video=user_video,
        target_video_name=user_video_name,
        template_video=template_video,
        template_video_name=template_video_name,
    )
    render_navigation()


main()
