import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Optional, Tuple


class PoseLandmarkerRunner:
    """
    Video -> List[pose_landmarks or None]
    pose_landmarks: 33 landmarks (NormalizedLandmark)
    """

    def __init__(self, model_path: str, num_poses: int = 1):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        # detect_for_video は landmarker インスタンス単位で timestamp の単調増加を要求する
        self._last_video_ts_ms = -1

    def iter_video_landmarks(
        self,
        video_path: str,
        visibility_th: float = 0.5,
        min_visible_keypoints: int = 4,
        prefer_person: int = 0,
        max_frames: Optional[int] = None,
    ) -> Tuple[float, List[Optional[list]]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video cannot be opened: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        poses: List[Optional[list]] = []
        frame_index = 0
        local_last_ts = -1

        # visibilityチェックに使う主要関節
        key_ids = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles
        min_visible = max(1, min(int(min_visible_keypoints), len(key_ids)))

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ts <= 0:
                ts = int((frame_index / fps) * 1000)
            if frame_index > 0 and ts <= local_last_ts:
                ts = local_last_ts + 1
            local_last_ts = ts

            # 2本目以降の動画でも landmarker 全体で単調増加にする
            if ts <= self._last_video_ts_ms:
                ts = self._last_video_ts_ms + 1
            self._last_video_ts_ms = ts

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, ts)

            if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose = result.pose_landmarks[prefer_person]
                visible_count = sum(1 for i in key_ids if pose[i].visibility >= visibility_th)
                if visible_count >= min_visible:
                    poses.append(pose)
                else:
                    poses.append(None)
            else:
                poses.append(None)

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break

        cap.release()
        return fps, poses
