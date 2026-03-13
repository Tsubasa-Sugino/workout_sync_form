"""
Video Upload Page
"""

import streamlit as st


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
