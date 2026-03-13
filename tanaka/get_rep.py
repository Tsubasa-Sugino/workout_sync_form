import numpy as np
from dynamic_time_qarping import DTWActionSegmenter
from model import PoseVisualizer3D
from movie import save_and_view_video_clip
from pca import (
    format_data_for_pca,
    normalize_pose_landmarks,
    perform_pca,
    visualize_pca_wave,
)


def get_rep(teacher_video_path_name, student_video_path_name):
    # 1. 動画から骨格データを抽出
    teacher_frames_data = PoseVisualizer3D().process_file(
        source_path=f"video/{teacher_video_path_name}.MOV",
    )

    student_frames_data = PoseVisualizer3D().process_file(
        source_path=f"video/{student_video_path_name}.MOV",
    )

    if not teacher_frames_data or not student_frames_data:
        print("骨格データの抽出に失敗したよ。")
        return None

    # 2. PCAで特徴的な波形を抽出
    teacher_frames_data = normalize_pose_landmarks(teacher_frames_data)
    student_frames_data = normalize_pose_landmarks(student_frames_data)

    pc1_teacher, explained_variance, components = perform_pca(
        format_data_for_pca(teacher_frames_data)
    )
    # 2. studentのデータ（行列）を準備
    student_matrix = format_data_for_pca(student_frames_data)

    # 3. teacherの係数を使って，studentのPC1を計算する
    # (データ - 平均) × 係数 で算出できるよ
    student_centered = student_matrix - np.mean(student_matrix, axis=0)
    pc1_student = np.dot(student_centered, components)
    # 3. 波形を可視化して、動作のチェックポイントを検出
    peaks = visualize_pca_wave(pc1_teacher, explained_variance)

    # 4. 動的時間伸縮（DTW）で代表的な動作のタイミングを特定
    """
    ここでは, teacherのPC1のピーク[1]〜[2]の範囲を「お手本の動作」として切り出し, studentのPC1全体と比較して類似した動作を探すよ.
    """
    dtw = DTWActionSegmenter(template=pc1_teacher[peaks[1] : peaks[2]], threshold=0.5)
    results, all_distances = dtw.search(pc1_student)

    # 5. 結果を表示
    save_and_view_video_clip(
        video_path=f"video/{teacher_video_path_name}.MOV",
        start_frame=peaks[1],
        end_frame=peaks[2],
    )
    for res in results:
        save_and_view_video_clip(
            video_path=f"video/{student_video_path_name}.MOV",
            start_frame=res["start"],
            end_frame=res["end"],
        )

        print(
            f"検出された動作セグメント: 開始フレーム {res['start']}, 終了フレーム {res['end']}, DTW距離 {res['distance']:.3f}"
        )
    return results


if __name__ == "__main__":
    # path setting
    teacher_video_path_name = "bench3"
    student_video_path_name = "bench4"

    get_rep(teacher_video_path_name, student_video_path_name)
