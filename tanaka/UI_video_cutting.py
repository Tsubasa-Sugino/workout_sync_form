import tempfile

import streamlit as st

# 旧（エラーが出る書き方）
# from moviepy.editor import VideoFileClip
# 新（MoviePy 2.0 以降の書き方）
from moviepy import VideoFileClip

st.title("動画トリミングツール")

# 1. ファイルのアップロード
uploaded_file = st.sidebar.file_uploader(
    "動画を選択してください．", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # 動画の読み込み
    video = VideoFileClip(tfile.name)
    duration = video.duration

    st.write(f"動画の長さ: {duration:.2f} 秒")

    # 2. UIで切り取り範囲を選択
    start_time, end_time = st.slider(
        "切り取る範囲を選択してください（秒）", 0.0, duration, (0.0, duration)
    )

    if st.button("カットを実行"):
        with st.spinner("処理中..."):
            # 3. 編集処理
            trimmed_video = video.subclipped(start_time, end_time)
            output_path = "trimmed_video.mp4"
            trimmed_video.write_videofile(output_path, codec="libx264")

            # 4. 結果の表示とダウンロード
            st.video(output_path)
            with open(output_path, "rb") as file:
                st.download_button("保存した動画をダウンロード", file, "trimmed.mp4")
