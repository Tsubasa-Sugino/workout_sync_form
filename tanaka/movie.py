import base64
import os
import subprocess

import cv2
from IPython.display import HTML, display


def save_and_view_video_clip(video_path, start_frame, end_frame, save_dir="clips"):
    # 1. 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 2. 動画情報の取得
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        print("Error: FPSが取得できませんでした．")
        return

    # 3. 出力ファイル名の設定（元のファイル名 + フレーム範囲）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_{start_frame}_{end_frame}.mp4"
    output_path = os.path.join(save_dir, output_filename)

    # 4. FFmpegで切り出し・保存
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path,
    ]
    subprocess.run(cmd)
    print(f"Saved clip to: {output_path}")

    # 5. ついでにノートブック上でも確認（任意）
    with open(output_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    display(
        HTML(
            f'<video width="400" controls><source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'
        )
    )
