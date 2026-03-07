import copy
import json

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter


class PoseVisualizer3D:
    # 骨格を繋ぐ線のペア
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

    def __init__(self, model_path="pose_landmarker_heavy.task", input_type="video"):
        """モデルの読み込みと設定を行うよ（初期化担当）．"""
        self.input_type = input_type.lower()

        if self.input_type == "image":
            running_mode = vision.RunningMode.IMAGE
        elif self.input_type == "video":
            running_mode = vision.RunningMode.VIDEO
        else:
            raise ValueError("input_type は 'image' か 'video' にしてね．")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    # ==========================================
    # 1. 解析機能（MediaPipeの処理とデータ構造化）
    # ==========================================
    def analyze(self, frame_or_image, timestamp_ms=None):
        """
        画像またはフレームを解析し，構造化された数値データを返すよ．
        戻り値: {"2d": [{"x":..., "y":..., "z":...}, ...], "3d": [...] } または None
        """
        # BGRからRGBに変換してMediaPipeフォーマットに
        rgb_image = cv2.cvtColor(frame_or_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # モードに応じた推論の実行
        if self.input_type == "image":
            result = self.landmarker.detect(mp_image)
        elif self.input_type == "video":
            if timestamp_ms is None:
                raise ValueError("videoモードでは timestamp_ms が必要だよ．")
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # 検出されなかった場合はNoneを返す
        if not result or not result.pose_landmarks:
            return None

        # 構造化データ（辞書）に変換して返す
        return self._format_result(result)

    def _format_result(self, result):
        """MediaPipeの結果を扱いやすいPythonの辞書リストに変換する内部メソッドだよ．"""
        pose_2d = result.pose_landmarks[0]  # 画像への描画用（0.0〜1.0）
        pose_3d = result.pose_world_landmarks[
            0
        ]  # 3Dプロット・角度計算用（現実のメートル単位）

        # visibility（見えている確率）と presence（画面内に存在している確率）を追加！
        data_2d = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
            for lm in pose_2d
        ]

        data_3d = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
            for lm in pose_3d
        ]

        return {"2d": data_2d, "3d": data_3d}

    # ==========================================
    # 2. 出力・可視化機能
    # ==========================================
    def print_data(self, pose_data):
        """結果を見やすい形でコンソールに出力するよ．"""
        if not pose_data:
            return
        print("--- 3D ランドマーク座標 ---")
        for i, lm in enumerate(pose_data["3d"]):
            print(
                f"Landmark {i:2}: (x={lm['x']:6.3f}, y={lm['y']:6.3f}, z={lm['z']:6.3f})"
            )

    def draw_2d(self, image, pose_data):
        """画像に2Dの骨格をプロットして返すよ．"""
        if not pose_data:
            return image

        annotated_image = np.copy(image)
        height, width, _ = annotated_image.shape
        landmark_px = {}

        # 点を打つ
        for idx, lm in enumerate(pose_data["2d"]):
            px_x, px_y = int(lm["x"] * width), int(lm["y"] * height)
            landmark_px[idx] = (px_x, px_y)
            cv2.circle(annotated_image, (px_x, px_y), 5, (0, 255, 0), -1)

        # 線を引く
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            if start_idx in landmark_px and end_idx in landmark_px:
                cv2.line(
                    annotated_image,
                    landmark_px[start_idx],
                    landmark_px[end_idx],
                    (255, 0, 0),
                    2,
                )

        return annotated_image

    def plot_3d(self, pose_data_list, title="3D Pose Animation"):
        """
        3次元座標をアニメーションプロットするよ．
        全フレームのリストを受け取って，Playボタンで動くHTMLを生成するよ．
        """
        if not pose_data_list:
            return None

        # 1. ベースとなる最初のフレーム（点と線）を作成
        first_pose = next((item for item in pose_data_list if item is not None), None)
        if not first_pose:
            return None

        fig = go.Figure()

        def get_coords(pose_data):
            xs = [lm["x"] for lm in pose_data["3d"]]
            ys = [lm["z"] for lm in pose_data["3d"]]
            zs = [-lm["y"] for lm in pose_data["3d"]]
            return xs, ys, zs

        # 初期描画（点）
        x0, y0, z0 = get_coords(first_pose)
        fig.add_trace(
            go.Scatter3d(
                x=x0,
                y=y0,
                z=z0,
                mode="markers",
                marker=dict(size=5, color="green"),
                name="Landmarks",
            )
        )

        # 初期描画（線：各ペアを1つのトレースとして追加）
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            fig.add_trace(
                go.Scatter3d(
                    x=[x0[start_idx], x0[end_idx]],
                    y=[y0[start_idx], y0[end_idx]],
                    z=[z0[start_idx], z0[end_idx]],
                    mode="lines",
                    line=dict(color="blue", width=4),
                    showlegend=False,
                )
            )

        # 2. 各フレームのデータを作成
        frames = []
        for i, pose_data in enumerate(pose_data_list):
            if pose_data is None:
                continue

            xi, yi, zi = get_coords(pose_data)
            frame_traces = []

            # 点の更新データ
            frame_traces.append(go.Scatter3d(x=xi, y=yi, z=zi))

            # 線の更新データ（初期描画と同じ順序で追加）
            for start_idx, end_idx in self.POSE_CONNECTIONS:
                frame_traces.append(
                    go.Scatter3d(
                        x=[xi[start_idx], xi[end_idx]],
                        y=[yi[start_idx], yi[end_idx]],
                        z=[zi[start_idx], zi[end_idx]],
                    )
                )

            frames.append(go.Frame(data=frame_traces, name=str(i)))

        fig.frames = frames

        # 3. アニメーションとスライダーの設定
        # durationを少し長め(100ms)に設定して，描画が追いつくようにするよ
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", range=[-1, 1]),
                yaxis=dict(title="Z", range=[-1, 1]),
                zaxis=dict(title="Y", range=[-1, 1]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i, f in enumerate(frames)
                    ]
                }
            ],
        )

        return fig

    def save_json(self, pose_data, output_path):
        """解析結果をJSONファイルに保存するよ．"""
        if not pose_data:
            print("保存するデータがないよ．")
            return False

        try:
            with open(output_path, "w") as f:
                json.dump(pose_data, f, indent=4)
            print(f"解析結果を '{output_path}' に保存したよ．")
            return True
        except Exception as e:
            print(f"データの保存に失敗したよ: {e}")
            return False

    # ==========================================
    # 3. 全体コントロール（司令塔）
    # ==========================================
    def process_file(
        self,
        source_path,
        output_base_name="output",
        show_console=False,
        save_2d=False,
        show_3d=False,
        save_json=False,
    ):
        """入力タイプを自動判別して適切な処理フローに流すまとめ役だよ．"""
        if self.input_type == "image":
            return self._process_image(
                source_path, output_base_name, show_console, save_2d, show_3d, save_json
            )
        elif self.input_type == "video":
            return self._process_video(
                source_path, output_base_name, show_console, save_2d, show_3d, save_json
            )
        else:
            print("エラー: 未対応のモードだよ．")
            return False

    def _process_image(
        self, source_path, output_base_name, show_console, save_2d, show_3d, save_json
    ):
        """静止画専用の処理フローだよ．"""
        image = cv2.imread(source_path)
        if image is None:
            print(f"画像 '{source_path}' が読み込めなかったよ．")
            return False

        pose_data = self.analyze(image)
        if not pose_data:
            print("人が検出されなかったよ．")
            return False

        if show_console:
            self.print_data(pose_data)

        if save_2d:
            out_img_path = f"{output_base_name}.jpg"
            cv2.imwrite(out_img_path, self.draw_2d(image, pose_data))
            print(f"2Dプロット画像を '{out_img_path}' に保存したよ．")

        if show_3d:
            fig = self.plot_3d(pose_data)
            if fig:
                out_html_path = f"{output_base_name}_3d.html"
                fig.write_html(out_html_path)
                print(f"3Dプロットを '{out_html_path}' に保存したよ．")
        if save_json:
            json_path = f"{output_base_name}.json"
            self.save_json(pose_data, json_path)

        return True

    def _process_video(
        self,
        source_path,
        output_base_name,
        show_console,
        save_2d,
        show_3d,
        save_json=False,
    ):
        """動画専用の処理フローだよ．全フレームの3Dアニメーションに対応！"""
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"動画 '{source_path}' が開けなかったよ．")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if save_2d:
            out_vid_path = f"{output_base_name}.mp4"
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(out_vid_path, fourcc, fps, (width, height))

        print(f"動画処理スタート！（入力: {source_path}）")
        frame_index = 0

        # 【変更点1】全フレームのデータを貯め込むリストを用意するよ
        all_frames_data = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = int((frame_index / fps) * 1000)
            if frame_index > 0 and timestamp_ms <= getattr(self, "last_timestamp", -1):
                timestamp_ms = self.last_timestamp + 1
            self.last_timestamp = timestamp_ms

            pose_data = self.analyze(frame, timestamp_ms)

            # 【変更点2】解析結果（Noneの場合も含めて）をリストに順番に追加していくよ
            all_frames_data.append(pose_data)

            if pose_data:
                # ログが埋まらないように1秒（fpsフレーム）に1回だけ出力
                if show_console and frame_index % int(fps) == 0:
                    print(f"\n--- Frame {frame_index} (Time: {timestamp_ms}ms) ---")
                    self.print_data(pose_data)

            frame_index += 1
            if frame_index % 100 == 0:
                print(f"{frame_index} フレーム完了...")

        # ここでデータのクリーニング職人を呼び出す！
        print("データのノイズ除去と補間を行っているよ...")
        smoother = PoseDataSmoother(visibility_threshold=0.0)
        all_frames_data = smoother.process(all_frames_data)
        out = None
        if save_2d:
            print("\n【ステップ3/3】補正されたデータで2D動画を作成中...")
            out_vid_path = f"{output_base_name}.mp4"
            # 警告が出にくい書き方に直しておくね
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(out_vid_path, fourcc, fps, (width, height))

            # 【重要】動画の再生位置を最初のフレーム(0)に戻す！
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frame_index = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 綺麗になったデータをリストから取り出す
                if frame_index < len(all_frames_data):
                    pose_data = all_frames_data[frame_index]
                else:
                    pose_data = None

                if pose_data:
                    # コンソール出力
                    if show_console and frame_index % int(fps) == 0:
                        print(f"--- Frame {frame_index} ---")
                        self.print_data(pose_data)

                # 描画して書き込み
                if out is not None:
                    annotated_frame = (
                        self.draw_2d(frame, pose_data) if pose_data else frame
                    )
                    out.write(annotated_frame)

                frame_index += 1
                if frame_index % 100 == 0:
                    print(f"動画出力中... {frame_index} フレーム完了")

        cap.release()
        if out is not None:
            out.release()
            print(f"\n2D動画の保存が完了したよ！（出力: {out_vid_path}）")

        # 【変更点3】動画のループが終わった「後」に，貯まったデータを一気に3Dプロットに渡すよ
        if show_3d and all_frames_data:
            print("3Dアニメーションを生成中...")
            fig = self.plot_3d(all_frames_data)
            if fig:
                out_html_path = f"{output_base_name}_3d.html"
                fig.write_html(out_html_path)
                print(
                    f"全フレームの3Dアニメーションを '{out_html_path}' に保存したよ．"
                )
        else:
            print("すべての動画処理が完了したよ！")

        if save_json:
            json_path = f"{output_base_name}.json"
            self.save_json(all_frames_data, json_path)

        return True


class PoseDataSmoother:
    def __init__(self, visibility_threshold=0.5, window_length=11, polyorder=3):
        """
        データのノイズ除去と補間を行うクラスだよ．
        - visibility_threshold: これより低い確率の座標は「隠れている」とみなして補間する
        - window_length: スムージングの窓サイズ（奇数．大きいほど滑らかになるけど遅れる）
        - polyorder: 曲線フィットの次数（通常は3でOK）
        """
        self.threshold = visibility_threshold
        self.window_length = window_length
        self.polyorder = polyorder

    def process(self, all_frames_data):
        """全フレームのデータを受け取って，綺麗に補正したデータを返すよ．"""
        if not all_frames_data:
            return []

        # 完全に人が見つからなかったフレーム(None)の対策
        # 最初の有効なフレームを探して，それをベースに「全部のvisibilityが0」のダミーフレームを作るよ
        first_valid = next((p for p in all_frames_data if p is not None), None)
        if not first_valid:
            return all_frames_data  # 全フレームに人がいなかったらそのまま返す

        cleaned_data = []
        for frame in all_frames_data:
            if frame is None:
                # 欠損フレームは，座標0・visibility0のダミーデータにする（後で補間される）
                dummy = copy.deepcopy(first_valid)
                for dim in ["2d", "3d"]:
                    for lm in dummy[dim]:
                        lm["x"], lm["y"], lm["z"] = 0.0, 0.0, 0.0
                        lm["visibility"] = 0.0
                cleaned_data.append(dummy)
            else:
                cleaned_data.append(copy.deepcopy(frame))

        num_frames = len(cleaned_data)
        num_landmarks = len(cleaned_data[0]["3d"])

        # 2Dと3Dのデータを，関節ごとに縦串で取り出して処理するよ
        for dimension in ["2d", "3d"]:
            for lm_idx in range(num_landmarks):
                # 1. 1つの関節の全フレーム分のデータを取り出す
                x_list = [f[dimension][lm_idx]["x"] for f in cleaned_data]
                y_list = [f[dimension][lm_idx]["y"] for f in cleaned_data]
                z_list = [f[dimension][lm_idx]["z"] for f in cleaned_data]
                v_list = [f[dimension][lm_idx]["visibility"] for f in cleaned_data]

                df = pd.DataFrame({"x": x_list, "y": y_list, "z": z_list, "v": v_list})

                # 2. マスク処理：visibilityが低いところを NaN（欠損値）にする
                mask = df["v"] < self.threshold
                df.loc[mask, ["x", "y", "z"]] = np.nan

                # 3. 補間処理：スプライン（3次曲線）で滑らかに穴埋めする
                # 前後がNaNだった時のために，前後の値で引っ張る bfill/ffill もかけておくよ
                # 3次曲線（cubic）だとデータが少なすぎる時にエラーになりやすいから，まずは線形（linear）で繋ぐよ
                df[["x", "y", "z"]] = df[["x", "y", "z"]].interpolate(
                    method="linear", limit_direction="both"
                )

                # 前後を引っ張って埋める
                df[["x", "y", "z"]] = df[["x", "y", "z"]].ffill().bfill()

                # 【重要】もし「動画の最初から最後まで一度もその関節が見えなかった」などの理由で
                # まだNaNが残っていたら，強制的に 0.0 で埋める安全装置！
                df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

                # 4. 平滑化処理：Savitzky-Golayフィルタで細かいブレを消す
                wl = min(self.window_length, num_frames)
                if wl % 2 == 0:
                    wl -= 1  # 窓サイズは必ず奇数にする必要があるんだ

                # データがちゃんと揃っている時だけフィルタをかける
                if wl > self.polyorder:
                    df["x"] = savgol_filter(
                        df["x"], window_length=wl, polyorder=self.polyorder
                    )
                    df["y"] = savgol_filter(
                        df["y"], window_length=wl, polyorder=self.polyorder
                    )
                    df["z"] = savgol_filter(
                        df["z"], window_length=wl, polyorder=self.polyorder
                    )

                # 5. 綺麗になったデータを cleaned_data に戻す
                for frame_idx in range(num_frames):
                    cleaned_data[frame_idx][dimension][lm_idx]["x"] = df.at[
                        frame_idx, "x"
                    ]
                    cleaned_data[frame_idx][dimension][lm_idx]["y"] = df.at[
                        frame_idx, "y"
                    ]
                    cleaned_data[frame_idx][dimension][lm_idx]["z"] = df.at[
                        frame_idx, "z"
                    ]

        return cleaned_data


# ==========================================
# 実行部分
# ==========================================
# ==========================================
# 実行部分の例
# ==========================================
if __name__ == "__main__":
    # 画像モードで実行したい場合
    # visualizer = PoseVisualizer3D(
    #     model_path="pose_landmarker_heavy.task", input_type="image"
    # )
    # visualizer.process_file(
    #     source_path="sandbox/IMG_7677.jpeg",
    #     output_base_name="output_result",
    #     show_console=True,
    #     save_2d=True,
    #     show_3d=True
    # )

    # 動画モードで実行したい場合
    visualizer = PoseVisualizer3D(
        model_path="pose_landmarker_heavy.task", input_type="video"
    )
    visualizer.process_file(
        source_path="video/squad2.MOV",
        output_base_name="result/output_video_result2",
        show_console=False,  # 座標を見たい時はTrueにしてね
        save_2d=True,  # 骨格が描画された動画を保存するよ
        show_3d=True,  # 最初のフレームの3Dグラフを保存するよ
        save_json=True,  # 解析結果をJSONファイルに保存するよ
    )
