"""
Form Results Page
"""
import streamlit as st


def get_inputs() -> tuple[bytes | None, bytes | None, str | None]:
    """Get page input values from session state."""

    user_video = st.session_state.get("user_video_bytes")
    ideal_video = st.session_state.get("ideal_video_bytes")
    exercise = st.session_state.get("exercise")
    return user_video, ideal_video, exercise


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


def render_videos(user_video: bytes, ideal_video: bytes) -> None:
    """Render uploaded and ideal videos side by side."""

    saved_clips: list[dict] = st.session_state.get(
        "saved_clips", [],
    )

    left, right = st.columns(2)
    with left:
        st.subheader("自分の動画")
        if saved_clips:
            clip_names = [c["name"] for c in saved_clips]
            video_slot = st.empty()
            selected = st.selectbox(
                "切り出し動画を選択",
                options=clip_names,
                key="results_clip_select",
            )
            clip = next(
                c for c in saved_clips
                if c["name"] == selected
            )
            video_slot.video(
                clip["bytes"],
                autoplay=True, loop=True, muted=True,
            )
        else:
            st.video(
                user_video,
                autoplay=True, loop=True, muted=True,
            )
    with right:
        st.subheader("理想のフォーム動画")
        st.video(
            ideal_video,
            autoplay=True, loop=True, muted=True,
        )


def render_result() -> None:
    """Render placeholder evaluation result."""

    st.markdown("---")
    st.subheader("フォーム評価結果")
    st.metric("一致スコア", "85%")
    st.write("フォームの一致度は高いですが、膝の位置が少し前に出ています。")
    st.write("スクワットの深さももう少し深くするとさらに良くなります。")


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

    user_video, ideal_video, exercise = get_inputs()
    guard_missing_videos(user_video, ideal_video)

    st.write(f"選択種目: **{exercise}**")
    render_videos(user_video, ideal_video)
    render_result()
    render_navigation()


main()
