"""
Video Upload Page
"""

from pathlib import Path

import streamlit as st


EXAMPLE_VIDEO_DIR = (
    Path(__file__).resolve().parent.parent / "menu_images" / "videos"
)
EXERCISE_TO_FILE_PREFIX = {
    "ベンチプレス": "bench_press",
    "デッドリフト": "deadlift",
    "スクワット": "squat",
}


def render_header() -> None:
    """Render the page title and selected exercise."""

    st.title("動画入力")
    st.write(
        f"今日のターゲットは．．． **{st.session_state.get('exercise', '未選択')}**!"
    )
    st.write("パワーー！！")


def init_video_state() -> None:
    """Initialize video-related session state keys once."""

    hold_videos = {
        "user_video_bytes": None,
        "user_video_name": None,
        "ideal_video_bytes": None,
        "ideal_video_name": None,
    }
    for key, value in hold_videos.items():
        if key not in st.session_state:
            st.session_state[key] = value


def list_example_videos() -> list[Path]:
    """Return sorted example videos under menu_images/videos."""

    if not EXAMPLE_VIDEO_DIR.exists():
        return []
    return sorted(
        path for path in EXAMPLE_VIDEO_DIR.iterdir() if path.is_file()
    )


def recommend_example_name(video_names: list[str], suffix: str) -> str:
    """Recommend a default example filename from selected exercise."""

    exercise = st.session_state.get("exercise")
    prefix = EXERCISE_TO_FILE_PREFIX.get(exercise)
    if not prefix:
        return "選択しない"

    target_stem = f"{prefix}_{suffix}"
    for name in video_names:
        if Path(name).stem.lower() == target_stem:
            return name
    return "選択しない"


def save_selected_example_video(filename: str, state_prefix: str) -> None:
    """Load selected example video and save it to session state."""

    video_path = EXAMPLE_VIDEO_DIR / filename
    if not video_path.exists():
        st.warning(f"例動画が見つかりません: {filename}")
        return

    st.session_state[f"{state_prefix}_video_bytes"] = video_path.read_bytes()
    st.session_state[f"{state_prefix}_video_name"] = video_path.name


def render_example_video_selector() -> None:
    """Render example video selectors and copy choices into session state."""

    example_videos = list_example_videos()
    if not example_videos:
        st.info("例動画フォルダに動画が見つかりませんでした。")
        return

    video_names = [path.name for path in example_videos]
    options = ["選択しない", *video_names]

    default_user = recommend_example_name(video_names, "mine")
    default_ideal = recommend_example_name(video_names, "ideal")

    st.subheader("例の動画を使う")
    st.caption("`menu_images/videos` の動画を選んで反映できます。")

    selected_user = st.selectbox(
        "自分の動画（例）",
        options=options,
        index=options.index(default_user),
        key="selected_example_user_video",
    )
    selected_ideal = st.selectbox(
        "理想フォーム動画（例）",
        options=options,
        index=options.index(default_ideal),
        key="selected_example_ideal_video",
    )

    if st.button("選択した例動画を反映", key="apply_example_videos"):
        applied = False

        if selected_user != "選択しない":
            save_selected_example_video(selected_user, "user")
            applied = True

        if selected_ideal != "選択しない":
            save_selected_example_video(selected_ideal, "ideal")
            applied = True

        if applied:
            st.success("例動画を反映しました。")
        else:
            st.warning("反映する例動画を選択してください。")


def render_uploaders() -> None:
    """Render file uploaders and persist uploaded content to session."""

    user_video = st.file_uploader(
        "自分の動画",
        type=["mp4", "mov", "avi"],
        key="user_video",
    )
    ideal_video = st.file_uploader(
        "理想フォーム動画",
        type=["mp4", "mov", "avi"],
        key="ideal_video",
    )

    if user_video is not None:
        st.session_state.user_video_bytes = user_video.getvalue()
        st.session_state.user_video_name = user_video.name
    if ideal_video is not None:
        st.session_state.ideal_video_bytes = ideal_video.getvalue()
        st.session_state.ideal_video_name = ideal_video.name

    render_example_video_selector()


def render_navigation() -> None:
    """Render readiness message and page navigation buttons."""

    is_ready = (
        st.session_state.user_video_bytes is not None
        and st.session_state.ideal_video_bytes is not None
    )

    if is_ready:
        st.success("動画がアップロードされました！動画から1動作を切り出して下さい")
        if st.button("動画切り出しへ"):
            st.switch_page("pages/video_trimming.py")
    else:
        st.warning("両方の動画をアップロードしてください。")

    if st.button("戻る"):
        st.switch_page("app.py")


def main() -> None:
    """Render the video upload page."""

    render_header()
    init_video_state()
    render_uploaders()
    render_navigation()


main()
