"""
Form Results Page
"""
import json
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
) -> tuple[bool, str, str | None]:
    """Run form_metric/scripts/main.py and return success + output text."""

    project_root = Path(__file__).resolve().parents[1]
    main_script = project_root / "form_metric" / "scripts" / "main.py"
    model_path = project_root / "form_metric" / "models" / "pose_landmarker_heavy.task"

    if not main_script.exists():
        return False, f"評価スクリプトが見つかりません: {main_script}", None
    if not model_path.exists():
        return False, f"モデルファイルが見つかりません: {model_path}", None

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
        return False, f"main.py の実行中に例外が発生しました: {exc}", None

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
            None,
        )

    report_path = _extract_report_path(output_text, project_root)
    return True, output_text, report_path


def _extract_report_path(output_text: str, project_root: Path) -> str | None:
    """Extract report path from main.py output and normalize it."""

    found = re.findall(r"Saved report:\s*(.+)", output_text)
    if not found:
        return None

    candidate = Path(found[-1].strip())
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        return None
    return str(candidate)


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


def _resolve_media_path(path_text: str | None, project_root: Path) -> Path | None:
    """Resolve JSON path fields into existing absolute paths."""

    if not path_text:
        return None
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        return None
    return candidate


def _make_playable_video_bytes(video_path: Path) -> bytes | None:
    """Return browser-playable video bytes (prefer H.264, fallback to source bytes)."""

    try:
        stat = video_path.stat()
        cache_key = f"{video_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        cache = st.session_state.setdefault("comparison_video_cache", {})
        if cache_key in cache:
            return cache[cache_key]
    except OSError:
        return None

    try:
        source_bytes = video_path.read_bytes()
    except OSError:
        return None

    # video_trimming.py と同じく ffmpeg で H.264 化を試みる
    h264_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    h264_temp.close()
    playable_bytes = source_bytes
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                h264_temp.name,
            ],
            check=True,
            capture_output=True,
        )
        converted = Path(h264_temp.name).read_bytes()
        if converted:
            playable_bytes = converted
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # ffmpeg が使えない場合は元動画をそのまま試す
        playable_bytes = source_bytes
    finally:
        try:
            Path(h264_temp.name).unlink(missing_ok=True)
        except OSError:
            pass

    # キャッシュ肥大化を避ける
    cache = st.session_state.setdefault("comparison_video_cache", {})
    if len(cache) > 12:
        cache.clear()
    cache[cache_key] = playable_bytes
    return playable_bytes


def _render_video_or_warning(video_path: Path, caption: str) -> None:
    """Render one video column with graceful fallback message."""

    st.markdown(
        (
            "<p style='font-size:1.1rem; font-weight:800; margin:0 0 0.4rem 0;'>"
            f"{caption}</p>"
        ),
        unsafe_allow_html=True,
    )
    video_bytes = _make_playable_video_bytes(video_path)
    if video_bytes is None:
        st.warning(f"動画の読み込みに失敗しました: {video_path.name}")
        return
    if not st.session_state.get("comparison_video_css_applied", False):
        st.markdown(
            """
            <style>
            div[data-testid="stVideo"] video {
                width: 100% !important;
                height: auto !important;
                object-fit: contain !important;
                aspect-ratio: auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.comparison_video_css_applied = True

    st.video(video_bytes, format="video/mp4", autoplay=True, loop=True, muted=True)


def _render_comparison_videos(report_path: str | None) -> None:
    """Render the compared template/target videos side by side."""

    st.markdown("---")
    st.subheader("比較に使われた動画")

    if report_path is None:
        st.info("比較動画情報が見つかりませんでした。")
        return

    project_root = Path(__file__).resolve().parents[1]
    report_file = Path(report_path)
    if not report_file.exists():
        st.info("比較動画情報ファイルが見つかりませんでした。")
        return

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.info("比較動画情報の読み込みに失敗しました。")
        return

    template_path = _resolve_media_path(
        str(report.get("template_segment", {}).get("template_clip_path", "")),
        project_root=project_root,
    )
    if template_path is None:
        template_path = _resolve_media_path(
            str(report.get("template_video", "")),
            project_root=project_root,
        )

    matches = report.get("matches")
    if isinstance(matches, list) and matches:
        pair_rows: list[tuple[str, Path]] = []
        for idx, match in enumerate(matches, start=1):
            target_path = _resolve_media_path(
                str(match.get("target_clip_path", "")),
                project_root=project_root,
            )
            if target_path is None:
                continue
            rep_idx = match.get("rep_index", idx)
            percent = match.get("match_percent")
            label = f"Rep {rep_idx}"
            if isinstance(percent, (int, float)):
                label = f"{label} ({float(percent):.1f}%)"
            pair_rows.append((label, target_path))

        if pair_rows and template_path is not None:
            options = [row[0] for row in pair_rows]
            selected = st.selectbox(
                "表示する比較ペア",
                options=options,
                key="comparison_pair_select",
            )
            selected_target = dict(pair_rows)[selected]
            left, right = st.columns(2)
            with left:
                _render_video_or_warning(selected_target, "自分の動画")
            with right:
                _render_video_or_warning(template_path, "お手本動画")
            return

    # ベンチプレスなど、rep切り出しを持たない場合のフォールバック
    target_path = _resolve_media_path(
        str(report.get("target_video", "")),
        project_root=project_root,
    )
    if template_path is not None and target_path is not None:
        left, right = st.columns(2)
        with left:
            _render_video_or_warning(target_path, "比較に使った自分の動画")
        with right:
            _render_video_or_warning(template_path, "比較に使ったお手本動画")
        return

    st.info("表示可能な比較動画が見つかりませんでした。")


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
        st.session_state.pop("form_eval_report_path", None)
        st.session_state.pop("comparison_video_cache", None)

    if st.button("評価を実行", key="run_form_eval"):
        with st.spinner("フォームを評価しています..."):
            success, output_text, report_path = run_main_evaluation(
                exercise_task=exercise_task,
                template_video=template_video,
                template_video_name=template_video_name,
                target_video=target_video,
                target_video_name=resolved_target_name,
            )
        st.session_state.form_eval_signature = current_signature
        st.session_state.form_eval_output = output_text
        st.session_state.form_eval_success = success
        st.session_state.form_eval_report_path = report_path

    output_text = st.session_state.get("form_eval_output")
    if output_text is None:
        st.info("`評価を実行` を押すと `main.py` の結果テキストを表示します。")
        return

    if st.session_state.get("form_eval_success"):
        st.success("評価が完了しました。")
        _render_score_only(output_text)
        _render_comparison_videos(st.session_state.get("form_eval_report_path"))
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
