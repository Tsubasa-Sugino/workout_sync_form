"""
Manual Video trimming Page
"""
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import cv2
import streamlit as st


st.set_page_config(
    page_title="KinNi Kun",
    page_icon="🏋️",
    layout="centered",
)

cv = cv2

EXERCISE_TO_TASK = {
    "ベンチプレス": "benchpress",
    "デッドリフト": "deadlift",
    "スクワット": "squat",
}


def init_state() -> None:
    """Initialize session state used by this page."""

    defaults = {
        "manual_trim_source_file_id": None,
        "manual_trim_source_path": None,
        "manual_trim_total_frames": 0,
        "manual_trim_fps": 0.0,
        "manual_trim_width": 0,
        "manual_trim_height": 0,
        "manual_trim_rotation_deg": 0,
        "manual_trim_ranges": [],
        "manual_trimmed_videos": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_video_bytes(video_bytes: bytes, filename: str) -> str:
    """Persist raw video bytes with filename extension and return path."""

    suffix = Path(filename).suffix or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(video_bytes)
    temp.flush()
    temp.close()
    return temp.name


def _extract_report_path(output_text: str, project_root: Path) -> str | None:
    """Extract single report path from main.py output."""

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


def detect_similar_ranges(
    exercise_task: str,
    template_video: bytes,
    template_video_name: str,
    target_video: bytes,
    target_video_name: str,
) -> tuple[bool, list[dict[str, int]], str]:
    """Detect similar target ranges by running auto mode once."""

    project_root = Path(__file__).resolve().parents[1]
    main_script = project_root / "form_metric" / "scripts" / "main.py"
    model_path = project_root / "form_metric" / "models" / "pose_landmarker_heavy.task"
    if not main_script.exists():
        return False, [], f"評価スクリプトが見つかりません: {main_script}"
    if not model_path.exists():
        return False, [], f"モデルファイルが見つかりません: {model_path}"

    try:
        with tempfile.TemporaryDirectory(prefix="manual_detect_") as temp_dir:
            temp_root = Path(temp_dir)
            template_suffix = Path(template_video_name).suffix or ".mp4"
            target_suffix = Path(target_video_name).suffix or ".mp4"
            template_path = temp_root / f"template{template_suffix}"
            target_path = temp_root / f"target{target_suffix}"
            template_path.write_bytes(template_video)
            target_path.write_bytes(target_video)

            command = [
                sys.executable,
                str(main_script),
                "--task", exercise_task,
                "--template", str(template_path),
                "--target", str(target_path),
                "--model", str(model_path),
            ]
            if exercise_task == "squat":
                command.extend(["--squat_eval_mode", "auto"])
            if exercise_task == "deadlift":
                command.extend(["--deadlift_eval_mode", "auto"])

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(project_root),
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            log_text = "\n\n".join(
                part for part in [stdout, f"[stderr]\n{stderr}" if stderr else ""]
                if part
            )
            if completed.returncode != 0:
                return (
                    False,
                    [],
                    "自動検出に失敗しました。\n\n"
                    f"(exit code {completed.returncode})\n\n{log_text}",
                )

            report_path = _extract_report_path(log_text, project_root)
            if report_path is None:
                return False, [], f"自動検出結果の report が見つかりません。\n\n{log_text}"

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            matches = report.get("matches", [])
            ranges: list[dict[str, int]] = []
            if isinstance(matches, list):
                for match in matches:
                    segment = match.get("target_segment", {})
                    if not isinstance(segment, dict):
                        continue
                    start = int(segment.get("start", -1))
                    end = int(segment.get("end", -1))
                    if start >= 0 and end >= start:
                        ranges.append({"start": start, "end": end})
            return True, ranges, log_text
    except Exception as exc:
        return False, [], f"自動検出中に例外が発生しました: {exc}"


def get_video_metadata(video_path: str) -> tuple[int, float, int, int]:
    """Read total frames, fps, width and height from the video."""

    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return total_frames, fps, width, height


def get_video_rotation_deg(video_path: str) -> int:
    """Read signed rotation degrees from video metadata (e.g. -90/0/90/180)."""

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_tags=rotate:stream_side_data=rotation",
                "-of",
                "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams")
        if not isinstance(streams, list) or not streams:
            return 0

        stream = streams[0]
        rotation = 0.0

        tags = stream.get("tags")
        if isinstance(tags, dict) and tags.get("rotate") is not None:
            rotation = float(tags["rotate"])

        if abs(rotation) < 1e-6:
            side_data_list = stream.get("side_data_list")
            if isinstance(side_data_list, list):
                for side_data in side_data_list:
                    if isinstance(side_data, dict) and side_data.get("rotation") is not None:
                        rotation = float(side_data["rotation"])
                        break

        normalized = int(round(rotation))
        while normalized > 180:
            normalized -= 360
        while normalized <= -180:
            normalized += 360
        if normalized in {-90, 0, 90}:
            return normalized
        if abs(normalized) == 180:
            return 180
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return 0
    return 0


def read_frame_rgb(video_path: str, frame_number: int):
    """Read a frame and convert BGR to RGB for Streamlit preview."""

    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rotation_deg = int(st.session_state.get("manual_trim_rotation_deg", 0) or 0)
    if rotation_deg == 90:
        frame_rgb = cv.rotate(frame_rgb, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_deg == -90:
        frame_rgb = cv.rotate(frame_rgb, cv.ROTATE_90_CLOCKWISE)
    elif abs(rotation_deg) == 180:
        frame_rgb = cv.rotate(frame_rgb, cv.ROTATE_180)
    return frame_rgb


def ensure_video_loaded() -> bool:
    """Load metadata for user video already stored in session state."""

    user_video_bytes = st.session_state.get("user_video_bytes")
    user_video_name = st.session_state.get(
        "user_video_name",
        "user_video.mp4",
    )

    if user_video_bytes is None:
        st.warning("先に動画入力ページで自分の動画をアップロードしてください。")
        if st.button("動画入力へ戻る", key="back_to_upload_page_manual"):
            st.switch_page("pages/upload_videos.py")
        return False

    file_id = f"{user_video_name}:{len(user_video_bytes)}"
    if st.session_state.manual_trim_source_file_id != file_id:
        source_path = save_video_bytes(user_video_bytes, user_video_name)
        total_frames, fps, width, height = get_video_metadata(source_path)
        rotation_deg = get_video_rotation_deg(source_path)

        st.session_state.manual_trim_source_file_id = file_id
        st.session_state.manual_trim_source_path = source_path
        st.session_state.manual_trim_total_frames = total_frames
        st.session_state.manual_trim_fps = fps
        st.session_state.manual_trim_width = width
        st.session_state.manual_trim_height = height
        st.session_state.manual_trim_rotation_deg = rotation_deg
        st.session_state.manual_trim_ranges = []
        st.session_state.manual_trimmed_videos = []

    return True


def get_template_for_manual_detection() -> tuple[bytes | None, str]:
    """Resolve selected template video bytes for manual detection."""

    ideal_video_bytes = st.session_state.get("ideal_video_bytes")
    ideal_video_name = st.session_state.get("ideal_video_name", "ideal_video.mp4")
    saved_clips: list[dict] = st.session_state.get("saved_clips", [])

    selected_template_video = ideal_video_bytes
    selected_template_name = ideal_video_name

    if saved_clips:
        clip_names = [c["name"] for c in saved_clips if "name" in c]
        selected = st.selectbox(
            "類似区間検出に使うお手本動画",
            options=["元のお手本動画", *clip_names],
            key="manual_template_select",
        )
        if selected != "元のお手本動画":
            clip = next((c for c in saved_clips if c.get("name") == selected), None)
            if clip and isinstance(clip.get("bytes"), (bytes, bytearray)):
                selected_template_video = bytes(clip["bytes"])
                selected_template_name = str(clip.get("name") or "template.mp4")

    if selected_template_video is not None:
        st.session_state["manual_template_video_bytes"] = selected_template_video
        st.session_state["manual_template_video_name"] = selected_template_name
    return selected_template_video, selected_template_name


def render_auto_detection_controls() -> None:
    """Detect similar ranges and prefill manual trim ranges."""

    st.markdown("---")
    st.subheader("類似区間の自動検出")
    exercise = st.session_state.get("exercise")
    exercise_task = EXERCISE_TO_TASK.get(exercise or "")

    template_video, template_name = get_template_for_manual_detection()
    user_video_bytes = st.session_state.get("user_video_bytes")
    user_video_name = st.session_state.get("user_video_name", "user_video.mp4")

    if template_video is None or user_video_bytes is None:
        st.info("お手本動画と自分の動画が必要です。")
        return

    if exercise_task not in {"squat", "deadlift"}:
        st.info("この種目では類似区間の自動検出は行わず、手動で範囲を追加してください。")
        return

    detect_signature = (
        f"{exercise_task}|{template_name}|{len(template_video)}|"
        f"{user_video_name}|{len(user_video_bytes)}"
    )
    if st.session_state.get("manual_detect_signature") != detect_signature:
        st.session_state.manual_detect_signature = detect_signature
        st.session_state.manual_trim_ranges = []
        st.session_state.manual_trimmed_videos = []
        st.session_state.pop("saved_user_clips", None)
        st.session_state.pop("manual_detect_log", None)

    if st.button("類似区間を自動検出", key="run_manual_auto_detect"):
        with st.spinner("お手本波形と類似する区間を探索しています..."):
            success, ranges, log_text = detect_similar_ranges(
                exercise_task=exercise_task,
                template_video=template_video,
                template_video_name=template_name,
                target_video=user_video_bytes,
                target_video_name=user_video_name,
            )
        st.session_state.manual_detect_log = log_text
        if success:
            if ranges:
                st.session_state.manual_trim_ranges = ranges
                st.success(f"{len(ranges)} 個の候補区間を検出しました。必要に応じて調整してください。")
            else:
                st.warning("候補区間が検出されませんでした。手動で範囲を追加してください。")
        else:
            st.error("自動検出に失敗しました。")

    log_text = st.session_state.get("manual_detect_log")
    if log_text:
        with st.expander("自動検出ログ"):
            st.code(log_text)


def add_range() -> None:
    """Append a new trimming range using current frame as a base."""

    total_frames = st.session_state.manual_trim_total_frames
    if total_frames <= 0:
        return

    start = 0
    end = min(start + 30, total_frames - 1)
    st.session_state.manual_trim_ranges.append({"start": start, "end": end})


@st.fragment
def render_ranges_editor() -> None:
    """Render range sliders with start/end frame previews."""

    st.subheader("切り取り範囲")
    if st.button("範囲を追加", key="add_manual_trim_range"):
        add_range()

    if not st.session_state.manual_trim_ranges:
        st.info("自動検出を実行すると候補範囲が表示されます。必要なら `範囲を追加` で手動追加してください。")
        return

    total_frames = st.session_state.manual_trim_total_frames
    fps = st.session_state.manual_trim_fps or 30.0
    max_frame = max(total_frames - 1, 0)
    video_path = st.session_state.manual_trim_source_path

    for idx, trim_range in enumerate(st.session_state.manual_trim_ranges):
        st.markdown(f"**範囲 {idx + 1}**")
        start_end = st.slider(
            f"範囲 {idx + 1} (開始, 終了)",
            min_value=0,
            max_value=max_frame,
            value=(trim_range["start"], trim_range["end"]),
            step=1,
            key=f"manual_trim_range_slider_{idx}",
        )
        start = start_end[0]
        end = start_end[1]
        st.session_state.manual_trim_ranges[idx]["start"] = start
        st.session_state.manual_trim_ranges[idx]["end"] = end

        col_start, col_end = st.columns(2)
        with col_start:
            st.caption(
                f"開始: Frame {start}"
                f" ({start / fps:.2f}s)"
            )
            frame_s = read_frame_rgb(video_path, start)
            if frame_s is not None:
                st.image(frame_s)
        with col_end:
            st.caption(
                f"終了: Frame {end}"
                f" ({end / fps:.2f}s)"
            )
            frame_e = read_frame_rgb(video_path, end)
            if frame_e is not None:
                st.image(frame_e)


def cut_videos() -> list[dict[str, object]]:
    """Extract selected frame ranges as individual video clips."""

    source_path = st.session_state.manual_trim_source_path
    fps = float(st.session_state.manual_trim_fps or 0.0)
    width = int(st.session_state.manual_trim_width or 0)
    height = int(st.session_state.manual_trim_height or 0)

    if fps <= 0:
        fps = 30.0

    clips: list[dict[str, object]] = []
    cap = cv.VideoCapture(source_path)

    if (width <= 0 or height <= 0) and cap.isOpened():
        ok, sample_frame = cap.read()
        if ok and sample_frame is not None:
            height, width = sample_frame.shape[:2]
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    if width <= 0 or height <= 0:
        cap.release()
        st.error("切り出しに必要な動画サイズを取得できませんでした。")
        return clips

    for idx, trim_range in enumerate(st.session_state.manual_trim_ranges):
        start = int(trim_range["start"])
        end = int(trim_range["end"])
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp.close()

        # Keep original display aspect ratio/orientation by trimming from source via ffmpeg first.
        start_sec = max(0.0, float(start) / float(fps))
        duration_sec = max(float(end - start + 1) / float(fps), 1.0 / float(fps))
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", source_path,
                    "-ss", f"{start_sec:.6f}",
                    "-t", f"{duration_sec:.6f}",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1",
                    "-movflags", "+faststart",
                    "-an",
                    output_temp.name,
                ],
                check=True,
                capture_output=True,
            )
            clip_bytes = Path(output_temp.name).read_bytes()
            if clip_bytes:
                clips.append(
                    {
                        "name": f"clip_{idx + 1:02d}_{start}_{end}.mp4",
                        "start": start,
                        "end": end,
                        "bytes": clip_bytes,
                        "path": output_temp.name,
                    }
                )
                continue
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            Path(output_temp.name).unlink(missing_ok=True)
        except OSError:
            pass

        # Prefer mp4v to avoid environment-dependent H264 encoder failures.
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(
            output_temp.name,
            fourcc,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            writer = None

        if writer is None:
            st.warning(f"範囲 {idx + 1} の出力ファイル作成に失敗しました。")
            continue

        cap.set(cv.CAP_PROP_POS_FRAMES, start)

        frame_idx = start
        written_count = 0
        while frame_idx <= end:
            success, frame = cap.read()
            if not success:
                break
            writer.write(frame)
            written_count += 1
            frame_idx += 1

        writer.release()

        if written_count == 0:
            st.warning(f"範囲 {idx + 1} はフレームが取得できずスキップしました。")
            continue

        clip_bytes = Path(output_temp.name).read_bytes()
        if not clip_bytes:
            st.warning(f"範囲 {idx + 1} の動画データが空のためスキップしました。")
            continue

        # Re-encode to H.264 so browsers can play it.
        h264_temp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4",
        )
        h264_temp.close()
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", output_temp.name,
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
            clip_bytes = Path(h264_temp.name).read_bytes()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Fall back to mp4v bytes already read.

        clips.append(
            {
                "name": f"clip_{idx + 1:02d}_{start}_{end}.mp4",
                "start": start,
                "end": end,
                "bytes": clip_bytes,
                "path": output_temp.name,
            }
        )

    cap.release()
    return clips


def render_cut_result() -> None:
    """Render clip output button and previews for extracted videos."""

    if st.button("動画を切り出す", key="run_manual_video_cut"):
        if not st.session_state.manual_trim_ranges:
            st.warning("切り取り範囲を1つ以上追加してください。")
        else:
            with st.spinner("動画を切り出しています..."):
                st.session_state.manual_trimmed_videos = cut_videos()
            st.success(
                f"{len(st.session_state.manual_trimmed_videos)} 本の動画を切り出しました。"
            )

    if st.session_state.manual_trimmed_videos:
        st.subheader("切り出し結果")
        for clip in st.session_state.manual_trimmed_videos:
            st.write(
                f"{clip['name']}"
                f" (Frame {clip['start']} - {clip['end']})"
            )
            st.video(clip["bytes"])

        # Persist clips for use on the results page.
        st.session_state["saved_user_clips"] = [
            {
                "name": c["name"],
                "start": c["start"],
                "end": c["end"],
                "bytes": c["bytes"],
            }
            for c in st.session_state.manual_trimmed_videos
        ]


def render_navigation() -> None:
    """Render navigation buttons."""

    step1, step2 = st.columns(2)
    with step1:
        if st.button("お手本動画の切り出しに戻る"):
            st.switch_page("pages/video_trimming.py")
    with step2:
        if st.button("評価画面へ進む"):
            if not st.session_state.get("saved_user_clips"):
                st.warning("先に切り出し動画を1本以上作成してください。")
                return
            st.session_state["evaluation_mode"] = "manual"
            st.switch_page("pages/form_results.py")


def main() -> None:
    """Render video trimming UI page."""

    st.title("自分の動画切り出し（手動評価用）")
    st.write("手動評価モードで使う自分の動画を切り出してください。")
    st.session_state["evaluation_mode"] = "manual"
    init_state()

    if not ensure_video_loaded():
        return

    render_auto_detection_controls()

    st.write(
        "総フレーム数: "
        f"{st.session_state.manual_trim_total_frames} | "
        f"FPS: {st.session_state.manual_trim_fps:.2f}"
    )
    render_ranges_editor()
    render_cut_result()
    render_navigation()


main()
