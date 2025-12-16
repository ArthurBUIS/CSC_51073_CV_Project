import cv2
import numpy as np
import landmark_extraction as le
import grader as gr
import mediapipe as mp
import os

def play_video_with_landmarks_and_reps(path, landmarks, rep_starts, sim_list, exercise_name):
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

    icon_path = os.path.join("icons", f"{exercise_name}.png")
    if os.path.exists(icon_path):
        icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        icon_h, icon_w = icon.shape[:2]
        # Resize icon
        max_icon_height = 80
        if icon_h > max_icon_height:
            scale = max_icon_height / icon_h
            icon = cv2.resize(icon, (int(icon_w * scale), int(icon_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        icon = None
    
    frame_idx = 0
    rep_idx = 0
    num_frames = len(landmarks)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps))  # ms per frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        h, w, _ = frame.shape

        # Segments for skeleton
        for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
            x1, y1, z1 = landmarks[frame_idx][start_idx]
            x2, y2, z2 = landmarks[frame_idx][end_idx]
            if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

        # Landmarks points
        for (x, y, z) in landmarks[frame_idx]:
            if not np.isnan(x) and not np.isnan(y):
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Reps and score
        while rep_idx + 1 < len(rep_starts) and frame_idx >= rep_starts[rep_idx + 1]:
            rep_idx += 1

        panel_w, panel_h = 220, 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        if frame_idx < rep_starts[0] or frame_idx >= rep_starts[-1]:
            current_rep = None
        else:
            current_rep = np.searchsorted(rep_starts, frame_idx, side="right") - 1

        # Text with shadow
        def put_text_shadow(img, text, pos, font, scale, color, shadow_color=(0,0,0), thickness=2):
            x, y = pos
            cv2.putText(img, text, (x+1, y+1), font, scale, shadow_color, thickness+1, cv2.LINE_AA)
            cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

        put_text_shadow(frame, f"Exercise: {exercise_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
        if current_rep is None:
            rep_text = "Reps: -"
        else:
            rep_text = f"Reps: {current_rep + 1}/{len(rep_starts) - 1}"

        put_text_shadow(frame, rep_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255))
        
        if current_rep is None or current_rep >= len(sim_list):
            score_text = "Score: -"
        else:
            score_text = f"Score: {sim_list[current_rep]:.2f}"

        put_text_shadow(frame,score_text,(20, 110),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 0))


        # Display icon
        if icon is not None:
            ih, iw = icon.shape[:2]
            frame_h, frame_w = frame.shape[:2]
            x_offset = frame_w - iw - 15
            y_offset = 15

            # Draw shadow rectangle
            shadow_color = (0, 0, 0)
            alpha_shadow = 0.6
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_offset-5, y_offset-5), (x_offset+iw+5, y_offset+ih+5), shadow_color, -1)
            cv2.addWeighted(overlay, alpha_shadow, frame, 1 - alpha_shadow, 0, frame)

            # Paste icon
            if icon.shape[2] == 4:  # RGBA
                alpha_s = icon[:, :, 3] / 255.0
                for c in range(3):
                    frame[y_offset:y_offset+ih, x_offset:x_offset+iw, c] = (
                        alpha_s * icon[:, :, c] + (1 - alpha_s) * frame[y_offset:y_offset+ih, x_offset:x_offset+iw, c]
                    )
            else:  # RGB
                frame[y_offset:y_offset+ih, x_offset:x_offset+iw] = icon


        # Final display
        cv2.imshow("Video + Landmarks + Reps", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    filename ="data/push-up_test1.mp4"
    landmarks, df = le.pipe_extract_landmark(filename)
    rep_starts, sim_list = gr.compute_repgrade(df, "push-up")
    play_video_with_landmarks_and_reps(filename, landmarks, rep_starts, sim_list)