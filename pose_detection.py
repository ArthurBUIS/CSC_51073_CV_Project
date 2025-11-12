import cv2
import mediapipe as mp
import numpy as np
import math
from tqdm import tqdm

def interpolate_landmarks(landmarks_sequence):
    """
    Interpolates missing landmarks (NaN) over time for each landmark.
    landmarks_sequence : np.ndarray of shape (n_frames, 33, 3)
    """
    arr = landmarks_sequence.copy()
    n_frames, n_landmarks, _ = arr.shape

    for j in range(n_landmarks):  # pour chaque landmark
        for k in range(3):  # x, y, z
            values = arr[:, j, k]
            mask = np.isnan(values)
            if np.any(~mask):
                valid = np.where(~mask)[0]
                arr[:, j, k] = np.interp(np.arange(n_frames), valid, values[valid])
    return arr

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
        print("Appuie sur la touche 'Esc' pour fermer la fenêtre.")
        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # Quit the view with 'Esc'
                break
        cv2.destroyAllWindows()
        return results.pose_landmarks
    




def extract_pose_from_video_interpolated(filename, show_interpolated=True): 
    """
    Detect pose landmarks from a video using MediaPipe.
    Missing landmarks are interpolated.
    Optionally visualize the interpolated landmarks.
    """
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError("Unable to open the video file.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pour la barre tqdm
    all_landmarks = []
    frames = []

    # === Lecture vidéo avec barre de progression ===
    print("🔍 Processing video frames...")
    for _ in tqdm(range(total_frames), desc="Reading frames", ncols=80):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        frame_landmarks = np.full((33, 3), np.nan)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[i] = [lm.x, lm.y, lm.z]
        
        all_landmarks.append(frame_landmarks)
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    if not all_landmarks:
        print("⚠️ No pose landmarks detected in any frame.")
        return None

    all_landmarks = np.array(all_landmarks)
    interpolated_landmarks = interpolate_landmarks(all_landmarks)
    print("✅ Interpolation done.")

    # === Sauvegarde TXT ===
    np.savetxt(
        "test/interpolated_landmarks.txt",
        interpolated_landmarks.reshape(-1, 3), 
        fmt="%.6f",
        header="x y z (flattened over all frames)",
        comments=""
    )

    # === Affichage (optionnel) ===
    if show_interpolated:
        print("▶ Replaying with interpolated landmarks...")

        for frame, landmarks in tqdm(zip(frames, interpolated_landmarks), total=len(frames), desc="Displaying", ncols=80):
            h, w, _ = frame.shape

            # Dessiner les segments
            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                x1, y1, z1 = landmarks[start_idx]
                x2, y2, z2 = landmarks[end_idx]
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    pt1 = (int(x1 * w), int(y1 * h))
                    pt2 = (int(x2 * w), int(y2 * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            # Dessiner les points
            for (x, y, z) in landmarks:
                if not np.isnan(x) and not np.isnan(y):
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            cv2.imshow("Pose Estimation (Interpolated)", frame)
            if cv2.waitKey(20) & 0xFF == 27:  # ESC pour quitter
                break

        cv2.destroyAllWindows()
        print("===== End of visualization =====")

    return interpolated_landmarks



def extract_pose_from_webcam():
    """
    ------------------------------------------------------------
    OBJECTIF :
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
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.3, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    #Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Webcam non accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Erreur de capture vidéo")
            break

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
    print("=====Fin du programme=====")



if __name__ == "__main__":
    # filename = "data/squat.jpg"  
    # extract_pose_from_image(filename)
    extract_pose_from_video_interpolated("data/data-crawl/chest fly machine/chest fly machine_8.mp4")
    # extract_pose_from_webcam()