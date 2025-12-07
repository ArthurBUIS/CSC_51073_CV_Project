import cv2
import numpy as np
import landmark_extraction as le
import grader as gr
import mediapipe as mp

def play_video_with_landmarks_and_reps(path, landmarks, rep_starts, sim_list):
    """
    Affiche une vidéo avec :
    - les landmarks
    - les segments (squelette MediaPipe)
    - compteur de reps et score
    """

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to open video")

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

        # --- Dessin des segments (squelette) ---
        for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
            x1, y1, z1 = landmarks[frame_idx][start_idx]
            x2, y2, z2 = landmarks[frame_idx][end_idx]
            if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # --- Dessin des points landmarks ---
        for (x, y, z) in landmarks[frame_idx]:
            if not np.isnan(x) and not np.isnan(y):
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # --- Compteur de reps ---
        while rep_idx + 1 < len(rep_starts) and frame_idx >= rep_starts[rep_idx + 1]:
            rep_idx += 1

        # --- Encarts texte ---
        panel_width = 280
        cv2.rectangle(frame, (10, 10), (panel_width, 110), (0, 0, 0), -1)

        cv2.putText(frame, f"Reps: {rep_idx + 1}/{len(rep_starts)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if rep_idx < len(sim_list):
            cv2.putText(frame, f"Score: {sim_list[rep_idx]:.2f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # --- Affichage ---
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
    play_video_with_landmarks_and_reps(filename, landmarks, rep_starts, sim_list)