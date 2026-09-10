import cv2
import numpy as np
import os
import landmark_extraction as le
import grader as gr
import mediapipe as mp

def play_video_with_landmarks_and_reps(path, landmarks, rep_starts, sim_list, exercise_name=None):
    """
    Displays a video with:
    - the landmarks
    - the segments (MediaPipe skeleton)
    - rep counter and score
    - the exercise name and its icon (icons/<exercise_name>.png), if available
    """

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to open video")

    icon = None
    if exercise_name is not None:
        icon_path = os.path.join("icons", f"{exercise_name}.png")
        if os.path.exists(icon_path):
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            icon_h, icon_w = icon.shape[:2]
            max_icon_height = 80
            if icon_h > max_icon_height:
                scale = max_icon_height / icon_h
                icon = cv2.resize(icon, (int(icon_w * scale), int(icon_h * scale)), interpolation=cv2.INTER_AREA)

    frame_idx = 0
    rep_idx = 0
    num_frames = len(landmarks)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps))  # ms par frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        h, w, _ = frame.shape

        # --- Draw segments (skeleton) ---
        for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
            x1, y1, z1 = landmarks[frame_idx][start_idx]
            x2, y2, z2 = landmarks[frame_idx][end_idx]
            if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # --- Draw landmark points ---
        for (x, y, z) in landmarks[frame_idx]:
            if not np.isnan(x) and not np.isnan(y):
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # --- Rep counter ---
        while rep_idx + 1 < len(rep_starts) and frame_idx >= rep_starts[rep_idx + 1]:
            rep_idx += 1

        # --- Text overlay ---
        panel_width = 280
        panel_height = 150 if exercise_name is not None else 110
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)

        text_y = 40
        if exercise_name is not None:
            cv2.putText(frame, f"Exercise: {exercise_name}",
                        (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 40

        cv2.putText(frame, f"Reps: {rep_idx + 1}/{len(rep_starts)}",
                    (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        text_y += 40

        if rep_idx < len(sim_list):
            cv2.putText(frame, f"Score: {sim_list[rep_idx]:.2f}",
                        (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # --- Icon (top-right corner) ---
        if icon is not None:
            ih, iw = icon.shape[:2]
            frame_h, frame_w = frame.shape[:2]
            x_offset = frame_w - iw - 15
            y_offset = 15

            overlay = frame.copy()
            cv2.rectangle(overlay, (x_offset - 5, y_offset - 5), (x_offset + iw + 5, y_offset + ih + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            if icon.shape[2] == 4:  # RGBA: alpha-blend onto the frame
                alpha_s = icon[:, :, 3] / 255.0
                for c in range(3):
                    frame[y_offset:y_offset+ih, x_offset:x_offset+iw, c] = (
                        alpha_s * icon[:, :, c] + (1 - alpha_s) * frame[y_offset:y_offset+ih, x_offset:x_offset+iw, c]
                    )
            else:  # RGB
                frame[y_offset:y_offset+ih, x_offset:x_offset+iw] = icon

        # --- Display ---
        cv2.imshow("Video + Landmarks + Reps", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    filename ="data/data-btc/push-up/push-up_test1.mp4"
    landmarks, df = le.pipe_extract_landmark(filename)
    rep_starts, sim_list = gr.compute_repgrade(df, "push-up")
    play_video_with_landmarks_and_reps(filename, landmarks, rep_starts, sim_list, "push-up")