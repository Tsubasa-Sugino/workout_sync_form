import streamlit as st

st.title("フォーム評価結果")

user_video = st.session_state.get("user_video_bytes")
ideal_video = st.session_state.get("ideal_video_bytes")
exercise = st.session_state.get("exercise")

if user_video is None or ideal_video is None:
    st.error("動画が未入力です。前の画面で2つの動画を入力してください。")
    if st.button("動画入力に戻る"):
        st.switch_page("pages/upload_videos.py")
    st.stop()

st.write(f"選択種目: **{exercise}**")

left, right = st.columns(2)
with left:
    st.subheader("自分の動画")
    st.video(user_video, autoplay=True, loop=True, muted=True)
with right:
    st.subheader("理想のフォーム動画")
    st.video(ideal_video, autoplay=True, loop=True, muted=True)

st.markdown("---")
st.subheader("フォーム評価結果")
st.metric("一致スコア", "85%")
st.write("フォームの一致度は高いですが、膝の位置が少し前に出ています。")
st.write("スクワットの深さももう少し深くするとさらに良くなります。")

step1, step2 = st.columns(2)
with step1:
    if st.button("種目選択に戻る"):
        st.switch_page("app.py")
with step2:
    if st.button("動画入力に戻る"):
        st.switch_page("pages/upload_videos.py")
