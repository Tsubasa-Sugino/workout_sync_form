"""
筋トレフォーム評価アプリ - Workout Form Evaluation Application

動画やカメラ入力からフォームを可視化・評価するStreamlitアプリケーション。
"""
import streamlit as st


def clear_video_state() -> None:
    """Clear uploaded videos when exercise selection changes."""

    keys_to_clear = [
        "user_video",
        "ideal_video",
        "user_video_bytes",
        "user_video_name",
        "ideal_video_bytes",
        "ideal_video_name",
        "uploaded_user_video",
        "uploaded_ideal_video",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def main() -> None:
    """Render the exercise selection page and persist the selected value."""

    st.set_page_config(
        page_title="KinNi Kun",
        page_icon="🏋️",
        layout="centered",
    )

    st.title("KinNi Kun -筋トレフォーム評価-")
    st.write("筋トレのフォームを理想に近づけよう！")

    st.header("種目選択")
    st.write("評価したい筋トレの種目を選択してください。")

    if "exercise" not in st.session_state:
        st.session_state.exercise = None

    current_exercise = st.session_state.exercise

    left, middle, right = st.columns(3)
    if left.button("スクワット", width="stretch"):
        if current_exercise != "スクワット":
            clear_video_state()
        st.session_state.exercise = "スクワット"
        st.switch_page("pages/upload_videos.py")
    elif middle.button("ベンチプレス", width="stretch"):
        if current_exercise != "ベンチプレス":
            clear_video_state()
        st.session_state.exercise = "ベンチプレス"
        st.switch_page("pages/upload_videos.py")
    elif right.button("デッドリフト", width="stretch"):
        if current_exercise != "デッドリフト":
            clear_video_state()
        st.session_state.exercise = "デッドリフト"
        st.switch_page("pages/upload_videos.py")


main()
