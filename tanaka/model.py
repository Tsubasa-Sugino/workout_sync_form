import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseVisualizer3D:
    # 骨格を繋ぐ線のペア（クラス変数として定義）
    POSE_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]

    def __init__(
        self,
        model_path="pose_landmarker_heavy.task",
        num_poses=1,
        input_type="image",
        result_callback=None,
    ):
        """モデルの初期化を1回だけ行うよ．"""
        base_options = python.BaseOptions(model_asset_path=model_path)

        """入力タイプに応じて実行モードとコールバックを設定するよ．"""
        self.input_type = input_type.lower()
        if input_type == "image":
            running_mode = vision.RunningMode.IMAGE
        elif input_type == "video":
            running_mode = vision.RunningMode.VIDEO
        elif input_type == "live_stream":
            running_mode = vision.RunningMode.LIVE_STREAM
            # オンライン処理（live_stream）の場合は結果を受け取る関数が必須だよ！
            if result_callback is None:

                def print_result(result: vision.PoseLandmarkerResult):
                    print("pose landmarker result: {}".format(result))

                result_callback = print_result
        else:
            raise ValueError(
                "input_typeは 'image', 'video', 'live_stream' のどれかにしてね．"
            )

        # optionsの設定（live_streamのときだけコールバックを登録するよ）
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=num_poses,
            result_callback=result_callback if input_type == "live_stream" else None,
        )

        # Landmarkerをインスタンス変数として保持して使い回す
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def format_input(self, source):
        """
        input_typeに合わせてMediaPipe用の画像フォーマットに変換するよ．
        - imageのとき: sourceには「画像のファイルパス（文字列）」を渡す
        - video / live_streamのとき: sourceには「OpenCVで読み込んだ1フレーム（numpy配列）」を渡す
        """
        if self.input_type == "image":
            cv_image = cv2.imread(source)
            if cv_image is None:
                raise FileNotFoundError(f"画像 '{source}' が読み込めなかったよ．")

        elif self.input_type in ["video", "live_stream"]:
            cv_image = source
            if cv_image is None:
                raise ValueError("フレームデータが空だよ．")
        else:
            raise ValueError("未対応のinput_typeだよ．")

        # 共通の変換処理（BGR -> RGB -> MediaPipe Image）
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    def detect_pose(self, mp_image, timestamp_ms=None):
        """
        input_typeに合わせて適切な推論メソッドを呼び出すよ．
        動画やライブ配信のときは，タイムスタンプ（timestamp_ms）が必要になるよ．
        """
        if self.input_type == "image":
            return self.landmarker.detect(mp_image)

        elif self.input_type == "video":
            if timestamp_ms is None:
                raise ValueError("videoモードでは timestamp_ms の指定が必須だよ．")
            # 動画用の推論メソッド
            return self.landmarker.detect_for_video(mp_image, timestamp_ms)

        elif self.input_type == "live_stream":
            if timestamp_ms is None:
                raise ValueError(
                    "live_streamモードでは timestamp_ms の指定が必須だよ．"
                )
            # ライブ用の非同期推論メソッド（結果は戻り値ではなく，コールバック関数に送られるよ）
            self.landmarker.detect_async(mp_image, timestamp_ms)
            return None

    def process_video(self, source_path, output_path="output_video.mp4"):
        """動画ファイルやカメラ映像を読み込んで，骨格を重ねた動画を保存するよ．"""
        if self.input_type != "video":
            print("エラー: input_typeが 'video' に設定されていないよ！")
            return False

        # source_pathが 0 ならWebカメラ，文字列なら動画ファイルを開くよ
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"動画 '{source_path}' が開けなかったよ．")
            return False

        # 動画のプロパティ（幅，高さ，FPS）を取得して保存の準備をするよ
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # Webカメラ等でFPSが取れない場合の保険
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # mp4形式で保存する設定
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"動画の処理を始めるよ！（出力先: {output_path}）")

        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("動画が最後まで終わったか，フレームが途切れたよ．")
                break

            # 動画モードに必須の「タイムスタンプ（ミリ秒）」を計算するよ
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                # タイムスタンプがうまく取れない場合はフレーム数から無理やり計算！
                timestamp_ms = int((frame_index / fps) * 1000)

            # 万が一タイムスタンプが被るとエラーになるから，必ず増えるようにする安全対策
            if frame_index > 0 and timestamp_ms <= getattr(self, "last_timestamp", -1):
                timestamp_ms = self.last_timestamp + 1
            self.last_timestamp = timestamp_ms

            # クラス内の各メソッドを連携させて処理！
            mp_image = self.format_input(frame)
            result = self.detect_pose(mp_image, timestamp_ms=timestamp_ms)

            annotated_frame = np.copy(frame)
            if result and result.pose_landmarks:
                for pose_landmarks in result.pose_landmarks:
                    # 2D描画メソッドを呼び出し
                    self.draw_2d_landmarks(annotated_frame, pose_landmarks)

            # 処理したフレームを動画に書き込む
            out.write(annotated_frame)

            frame_index += 1
            if frame_index % 100 == 0:
                print(f"{frame_index} フレーム目を処理中...")

        # 後片付けをしっかりやるよ
        cap.release()
        out.release()
        print(f"完了！ '{output_path}' に動画を保存したよ．")
        return True

    def print_landmarks(self, world_landmarks):
        """ランドマークの座標をコンソールに出力するよ．"""
        print("--- ランドマークの座標 ---")
        for i, lm in enumerate(world_landmarks):
            print(f"Landmark {i:2}: (x={lm.x:6.3f}, y={lm.y:6.3f}, z={lm.z:6.3f})")

    def draw_2d_landmarks(self, image, pose_landmarks):
        """画像（フレーム）の上に2Dの骨格を直接描画するよ．"""
        height, width, _ = image.shape
        landmark_px = {}

        # 1. 各関節の座標を計算して緑の点を打つ
        for idx, landmark in enumerate(pose_landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_px[idx] = (x, y)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # 2. 定義したペアにしたがって青い線を結ぶ
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            if start_idx in landmark_px and end_idx in landmark_px:
                cv2.line(
                    image, landmark_px[start_idx], landmark_px[end_idx], (255, 0, 0), 2
                )

        return image

    def create_3d_figure(self, world_landmarks):
        """Plotlyの3Dグラフオブジェクトを生成するよ．"""
        fig = go.Figure()

        # 座標の抽出 (Yは上下反転)
        xs = [lm.x for lm in world_landmarks]
        ys = [lm.z for lm in world_landmarks]
        zs = [-lm.y for lm in world_landmarks]

        # 関節の点をプロット
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(size=5, color="green"),
                name="Landmarks",
            )
        )

        # 骨格の線をプロット
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            fig.add_trace(
                go.Scatter3d(
                    x=[xs[start_idx], xs[end_idx]],
                    y=[ys[start_idx], ys[end_idx]],
                    z=[zs[start_idx], zs[end_idx]],
                    mode="lines",
                    line=dict(color="blue", width=4),
                    showlegend=False,
                )
            )

        # レイアウト設定
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X (Side)", range=[-1, 1]),
                yaxis=dict(title="Z (Depth)", range=[-1, 1]),
                zaxis=dict(title="Y (Up/Down)", range=[-1, 1]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            title="Interactive 3D Pose Visualization",
        )
        return fig

    def process_and_save(self, image_path, output_html="pose_3d_interactive.html"):
        """画像読み込みから保存までを一気にやる便利メソッドだよ．"""
        try:
            mp_image = self.format_input(image_path)
            result = self.detect_pose(mp_image)

            if not result.pose_world_landmarks:
                print("人が見つからなかったよ．")
                return False

            # 最初の1人分のデータを使う
            world_landmarks = result.pose_world_landmarks[0]

            # 各機能を呼び出す
            self.print_landmarks(world_landmarks)
            fig = self.create_3d_figure(world_landmarks)

            # HTMLに保存
            fig.write_html(output_html)
            print(f"HTMLファイル '{output_html}' を保存したよ．")
            return True

        except Exception as e:
            print(f"エラーが発生したよ: {e}")
            return False


# ==========================================
# クラスの使い方（実行部分）
# ==========================================
if __name__ == "__main__":
    # クラスをインスタンス化（ここでモデルが読み込まれる）
    visualizer = PoseVisualizer3D(
        model_path="pose_landmarker_heavy.task", input_type="video"
    )

    # 処理を実行
    visualizer.process_video(
        source_path="sandbox/IMG_7940.MOV", output_path="output_video.mp4"
    )  # 0はWebカメラを意味するよ
