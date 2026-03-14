"""
Video trimming Page
"""
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member

from pathlib import Path
import subprocess
import tempfile

import cv2
import streamlit as st


st.set_page_config(
    page_title="KinNi Kun",
    page_icon="🏋️",
    layout="centered",
)

cv = cv2

EVAL_ROUTE_OPTIONS = [
    "自動切り出しでそのまま評価へ進む",
    "手動切り出しで自分の動画も切り出す",
]
EVAL_ROUTE_MAP = {
    "自動切り出しでそのまま評価へ進む": "auto",
    "手動切り出しで自分の動画も切り出す": "manual",
}


def init_state() -> None:
    """Initialize session state used by this page."""

    defaults = {
        "trim_source_file_id": None,
        "trim_source_path": None,
        "trim_total_frames": 0,
        "trim_fps": 0.0,
        "trim_width": 0,
        "trim_height": 0,
        "trim_ranges": [],
        "trimmed_videos": [],
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


def get_video_metadata(video_path: str) -> tuple[int, float, int, int]:
    """Read total frames, fps, width and height from the video."""

    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return total_frames, fps, width, height


def read_frame_rgb(video_path: str, frame_number: int):
    """Read a frame and convert BGR to RGB for Streamlit preview."""

    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


def ensure_video_loaded() -> bool:
    """Load metadata for ideal video already stored in session state."""

    ideal_video_bytes = st.session_state.get("ideal_video_bytes")
    ideal_video_name = st.session_state.get(
        "ideal_video_name",
        "ideal_video.mp4",
    )

    if ideal_video_bytes is None:
        st.warning("先に動画入力ページで理想フォーム動画をアップロードしてください。")
        if st.button("動画入力へ戻る", key="back_to_upload_page"):
            st.switch_page("pages/upload_videos.py")
        return False

    file_id = f"{ideal_video_name}:{len(ideal_video_bytes)}"
    if st.session_state.trim_source_file_id != file_id:
        source_path = save_video_bytes(ideal_video_bytes, ideal_video_name)
        total_frames, fps, width, height = get_video_metadata(source_path)

        st.session_state.trim_source_file_id = file_id
        st.session_state.trim_source_path = source_path
        st.session_state.trim_total_frames = total_frames
        st.session_state.trim_fps = fps
        st.session_state.trim_width = width
        st.session_state.trim_height = height
        st.session_state.trim_ranges = []
        st.session_state.trimmed_videos = []

    return True


def add_range() -> None:
    """Set the single trimming range (1 rep) using default bounds."""

    total_frames = st.session_state.trim_total_frames
    if total_frames <= 0:
        return

    start = 0
    end = min(start + 30, total_frames - 1)
    st.session_state.trim_ranges = [{"start": start, "end": end}]


@st.fragment
def render_ranges_editor() -> None:
    """Render range sliders with start/end frame previews."""

    if not st.session_state.trim_ranges:
        add_range()
    elif len(st.session_state.trim_ranges) > 1:
        st.session_state.trim_ranges = [st.session_state.trim_ranges[0]]

    total_frames = st.session_state.trim_total_frames
    fps = st.session_state.trim_fps or 30.0
    max_frame = max(total_frames - 1, 0)
    video_path = st.session_state.trim_source_path

    trim_range = st.session_state.trim_ranges[0]
    st.markdown("**切り取り範囲(開始, 終了)**")
    start_end = st.slider(
        "お手本動画の切り出しは 1rep 分のみ対応しています。",
        min_value=0,
        max_value=max_frame,
        value=(trim_range["start"], trim_range["end"]),
        step=1,
        key="trim_range_slider_0",
    )
    start = start_end[0]
    end = start_end[1]
    st.session_state.trim_ranges[0]["start"] = start
    st.session_state.trim_ranges[0]["end"] = end

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

    source_path = st.session_state.trim_source_path
    fps = float(st.session_state.trim_fps or 0.0)
    width = int(st.session_state.trim_width or 0)
    height = int(st.session_state.trim_height or 0)

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

    for idx, trim_range in enumerate(st.session_state.trim_ranges):
        start = int(trim_range["start"])
        end = int(trim_range["end"])
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp.close()

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

    if st.button("動画を切り出す", key="run_video_cut"):
        if not st.session_state.trim_ranges:
            st.warning("切り取り範囲（1rep）を設定してください。")
        else:
            with st.spinner("動画を切り出しています..."):
                st.session_state.trimmed_videos = cut_videos()
            st.success(
                f"{len(st.session_state.trimmed_videos)} 本の動画を切り出しました。"
            )

    if st.session_state.trimmed_videos:
        st.subheader("切り出し結果")
        for clip in st.session_state.trimmed_videos:
            st.write(
                f"{clip['name']}"
                f" (Frame {clip['start']} - {clip['end']})"
            )
            _, preview_col, _ = st.columns([1, 2, 1])
            with preview_col:
                st.video(clip["bytes"], autoplay=True)

        # Persist single template clip for use on the results page.
        st.session_state["saved_clips"] = [
            {
                "name": c["name"],
                "start": c["start"],
                "end": c["end"],
                "bytes": c["bytes"],
            }
            for c in st.session_state.trimmed_videos[:1]
        ]


def render_navigation() -> None:
    """Render navigation buttons."""

    st.markdown("---")
    st.subheader("次のステップ")
    selected_route = st.radio(
        "評価方法を選択してください",
        options=EVAL_ROUTE_OPTIONS,
        key="post_template_trim_mode",
        horizontal=True,
    )
    selected_mode = EVAL_ROUTE_MAP[selected_route]
    st.session_state["evaluation_mode"] = selected_mode

    step1, step2 = st.columns(2)
    with step1:
        if st.button("動画入力に戻る"):
            st.switch_page("pages/upload_videos.py")
    with step2:
        if st.button("次へ進む"):
            if selected_mode == "manual":
                st.switch_page("pages/video_trimming_manual.py")
            else:
                st.switch_page("pages/form_results.py")


def main() -> None:
    """Render video trimming UI page."""

    st.title("理想フォーム動画切り出し")
    init_state()

    if not ensure_video_loaded():
        return

    render_ranges_editor()
    render_cut_result()
    render_navigation()


main()
