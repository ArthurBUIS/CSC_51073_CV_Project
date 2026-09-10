import cv2
import mediapipe as mp
import numpy as np
import math
from tqdm import tqdm

from landmark_extraction import extract_pose_from_video_interpolated, normalize_landmarks

def extract_pose_from_image (filename):
    """
    ------------------------------------------------------------
    OBJECTIVE :
        Detect and screen in real time the position of a person detected on a single image using MediaPipe.
        The function also returns the coordinate of the pose landmarks.

    INPUT :
        - file : str
            Path to the image file from which the pose is extracted

    OUTPUT :
        - results.pose_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            List of the pose landmarks detected on the image

    EXCEPTIONS :
        - FileNotFoundError : if the provided file path is invalid or unreadable.
        - ValueError : if the image cannot be processed by MediaPipe.
        - None returned if no pose is detected.
    ------------------------------------------------------------
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode = True, model_complexity = 2, enable_segmentation = False, min_detection_confidence = 0.3, min_tracking_confidence=0.3) # True for a single frame
    mp_drawing = mp.solutions.drawing_utils #Draw landamrks
    
    image = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert from BGR to RGB for mediaPipe
    
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print("No pose landmarks detected.")
        return None
    else: 
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            print(f"Point {id}: ({cx}, {cy})")
        
        cv2.imshow("Pose Estimation", annotated_image)
        print("Press 'Esc' to close the window.")
        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # Quit the view with 'Esc'
                break
        cv2.destroyAllWindows()
        return results.pose_landmarks
    



def extract_pose_from_video(filename): 
    """
    ------------------------------------------------------------
    OBJECTIVE :
        Detect and screen in real time the position of a person detected on a video using MediaPipe.
        The function also returns the coordinate of the pose landmarks.

    INPUT :
        - file : str
            Path to the video file from which the pose is extracted

    OUTPUT :
        - 

    EXCEPTIONS :
        - FileNotFoundError : if the provided file path is invalid or unreadable.
        - ValueError : if the image cannot be processed by MediaPipe.
        - None returned if no pose is detected.
    ------------------------------------------------------------
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode = False, model_complexity = 2, enable_segmentation = False, min_detection_confidence = 0.3, min_tracking_confidence=0.3)
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
            raise ValueError("Unable to open the video file.")
    
    results = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        cv2.imshow("Pose Estimation (Video)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("===== End of program =====")
    return results.pose_landmarks if results else None
    
    

def display_normalized_skeleton(
    landmarks_norm,
    size=600,
    out_path="normalized_skeleton.avi",
    fps=30,
    show=True
):
    mp_pose = mp.solutions.pose
    connections = mp_pose.POSE_CONNECTIONS

    # === Camera rotation (static here) ===
    R_view = np.eye(3)

    # === Video writer (ROBUST) ===
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (size, size))

    scale = size * 0.25
    center = size // 2
    origin = np.array([center, center])

    for pts in landmarks_norm:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        # Rotate
        pts_rot = (R_view @ pts.T).T

        # Project
        pts_2d = pts_rot[:, :2] * scale
        pts_2d[:, 1] *= -1
        pts_2d += center

        # Skeleton (yellow)
        for i, j in connections:
            p1 = pts_2d[i].astype(int)
            p2 = pts_2d[j].astype(int)
            cv2.line(canvas, tuple(p1), tuple(p2), (0, 255, 255), 2)

        # Points (green)
        for p in pts_2d:
            cv2.circle(canvas, tuple(p.astype(int)), 3, (0, 255, 0), -1)

        # Axes
        axes = np.eye(3) * scale * 0.4
        axes = (R_view @ axes.T).T
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        labels = ["X", "Y", "Z"]

        for ax, col, lab in zip(axes, colors, labels):
            end = origin + ax[:2] * np.array([1, -1])
            cv2.arrowedLine(canvas, tuple(origin), tuple(end.astype(int)), col, 2)
            cv2.putText(canvas, lab, tuple(end.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # Write frame
        writer.write(canvas)

        if show:
            cv2.imshow("Normalized Skeleton", canvas)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break

    writer.release()
    cv2.destroyAllWindows()
    print(f"🎥 Video saved: {out_path}")



def extract_pose_from_webcam():
    """
    ------------------------------------------------------------
    OBJECTIVE :
        Detect and screen in real time the position of a person detected on the webcam using MediaPipe.

    INPUT :
        - None

    OUTPUT :
        - Video with the pose landmarks

    EXCEPTIONS :
        - ValueError : if the webcam is undetcectable or if there is an error in video capture.
    ------------------------------------------------------------
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils

    #Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Webcam not accessible")

    while True:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Video capture error")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        cv2.imshow("Pose Estimation (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()
    print("===== End of program =====")



if __name__ == "__main__":
    landmarks = extract_pose_from_video_interpolated("data/data-btc/squat/squat_test1.mp4", show=False)
    landmarks_norm = normalize_landmarks(landmarks)
    display_normalized_skeleton(landmarks_norm)