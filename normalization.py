import pose_detection as pdec
import DataClasses as dc
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
import csv

POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # bras gauche
    (12, 14), (14, 16),  # bras droit
    (11, 12),            # épaules
    (23, 24),            # hanches
    (11, 23), (12, 24),  # tronc
    (23, 25), (25, 27),  # jambe gauche
    (24, 26), (26, 28),  # jambe droite
    (27, 29), (29, 31),  # pied gauche
    (28, 30), (30, 32),  # pied droit
    (15, 17), (15, 19), (15, 21),  # main gauche
    (16, 18), (16, 20), (16, 22),  # main droite
]

def show_video_with_landmarks(video_path, landmarks):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(landmarks):
            break
        
        h, w, _ = frame.shape
        pts = landmarks[frame_idx]   # shape (33, 3)

        # -- 1) Dessiner les segments du squelette --
        for a, b in POSE_CONNECTIONS:
            x1, y1, _ = pts[a]
            x2, y2, _ = pts[b]
            
            # Vérifier que les points sont valides
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

        # -- 2) Dessiner chaque landmark --
        for (x, y, z) in pts:
            if x > 0 and y > 0:
                px = int(x * w)
                py = int(y * h)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # -- 3) Affichage --
        cv2.imshow("Video with Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def show_video_with_frames(video_path, frames):
    """
    Affiche la vidéo avec superposition des landmarks provenant de frames: List[FrameData]
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frames):
            break

        h, w, _ = frame.shape

        # Récupérer les landmarks du frame courant
        pts = frames[frame_idx].landmarks  # (33, 3)

        # -- 1) Dessiner les segments du squelette --
        for a, b in POSE_CONNECTIONS:
            x1, y1, _ = pts[a]
            x2, y2, _ = pts[b]

            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

        # -- 2) Dessiner chaque landmark --
        for (x, y, z) in pts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

        # -- 3) Affichage --
        cv2.imshow("Video with Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    
def play_normalized_frames(frames, window_name="Normalized Video", size=512):
    """
    Affiche les frames normalisés (landmarks uniquement) sous forme d'animation squelette.
    frames : List[FrameData]
    size : taille de l'image carrée (pixels)
    """

    while True:
        for frame_data in frames:
            pts = frame_data.landmarks  # shape (33, 3)

            # créer une image noire
            canvas = np.zeros((size, size, 3), dtype=np.uint8)

            # --- 1) dessiner les segments ---
            for a, b in POSE_CONNECTIONS:
                x1, y1, _ = pts[a]
                x2, y2, _ = pts[b]
                
                # les landmarks normalisés tournent autour de (0,0)
                # donc on les remet dans l'image
                p1 = (int((x1 * 100 + size/2)), int((y1 * 100 + size/2)))
                p2 = (int((x2 * 100 + size/2)), int((y2 * 100 + size/2)))

                cv2.line(canvas, p1, p2, (0, 255, 255), 2)

            # --- 2) dessiner les points ---
            for (x, y, z) in pts:
                px = int((x * 100 + size/2))
                py = int((y * 100 + size/2))
                cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)

            # --- 3) afficher ---
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return


def play_normalized_video_from_csv(csv_file, video_name, window_name="Normalized Video", size=512, scale=100):
    """
    Affiche toutes les frames normalisées d'une vidéo sous forme de squelette animé.
    
    csv_file : chemin vers le CSV
    video_name : nom de la vidéo
    window_name : nom de la fenêtre OpenCV
    size : taille de l'image carrée (pixels)
    scale : facteur pour agrandir les coordonnées normalisées
    """
    df = pd.read_csv(csv_file)
    
    # Filtrer le dataframe pour cette vidéo
    video_df = df[df['video_name'] == video_name].sort_values('frame_number')
    if len(video_df) == 0:
        print(f"Aucune ligne trouvée pour {video_name}")
        return
    
    # Colonnes des landmarks
    lm_cols = [col for col in df.columns if col.startswith("lm_")]
    
    # Convertir en liste de frames (chaque frame = array (33,3))
    frames = [row[lm_cols].values.astype(float).reshape(33,3) for _, row in video_df.iterrows()]
    
    while True:
        for pts in frames:
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            
            # --- 1) Dessiner les segments ---
            for a,b in POSE_CONNECTIONS:
                x1, y1, _ = pts[a]
                x2, y2, _ = pts[b]
                p1 = (int(x1*scale + size/2), int(y1*scale + size/2))
                p2 = (int(x2*scale + size/2), int(y2*scale + size/2))
                cv2.line(canvas, p1, p2, (0,255,255), 2)
            
            # --- 2) Dessiner les landmarks ---
            for x,y,z in pts:
                px, py = int(x*scale + size/2), int(y*scale + size/2)
                cv2.circle(canvas, (px, py), 4, (0,255,0), -1)
            
            # --- 3) Afficher ---
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms ≈ 33 FPS
                cv2.destroyAllWindows()
                return



if __name__ == "__main__":
    
    filename_ = "data/data-btc/barbell biceps curl/barbell biceps curl_52.mp4"
    # cap = cv2.VideoCapture(filename_)
    # fps_ = cap.get(cv2.CAP_PROP_FPS)
    # cap.release()

    # landmarks_ = pdec.extract_pose_from_video_interpolated(filename_, show_interpolated=False)
    
    video = dc.VideoData(
        filename=filename_,
        landmarks=[],
        fps=30,
        predicted_class="push up",
        confidence=0.99
    )
    
    # video.landmark_estimation()
    # video.normalize()
    # video.rotate()
    # play_normalized_frames(video.frames)
    
    play_normalized_video_from_csv("data/data-btc/full_landmarks_dataset_features.csv", "barbell biceps curl_52.mp4")

    
