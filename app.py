"""
筋トレフォーム評価アプリ - Workout Form Evaluation Application

動画やカメラ入力からフォームを可視化・評価するStreamlitアプリケーション。
"""
import streamlit as st
from streamlit_image_select import image_select


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

    if "exercise" not in st.session_state:
        st.session_state.exercise = None

    current_exercise = st.session_state.exercise

    exercise_options = [
        ("menu_images/bench_press.png", "ベンチプレス"),
        ("menu_images/deadlift.png", "デッドリフト"),
        ("menu_images/squat.png", "スクワット"),
    ]
    image_to_caption = {
        image_path: caption for image_path, caption in exercise_options
    }

    img = image_select(
        "評価したい筋トレの種目を選択してください。",
        images=[item[0] for item in exercise_options],
        captions=[item[1] for item in exercise_options],
    )
    selected_exercise = image_to_caption.get(img)
    if selected_exercise and selected_exercise != current_exercise:
        clear_video_state()
        st.session_state.exercise = selected_exercise

    if selected_exercise:
        st.info(f"選択中の種目: **{selected_exercise}**")

    if st.button("動画入力へ"):
        st.switch_page("pages/upload_videos.py")


main()
