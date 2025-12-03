import pose_detection as pdec
import DataClasses as dc
import numpy as np
import mediapipe as mp
import cv2
import os
from tqdm import tqdm
import pandas as pd
import csv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from ExerciseClasses import EXERCISES

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

ANGLE_TRIPLETS = [
    # Bras
    (11, 13, 15),  # épaule g - coude g - poignet g
    (12, 14, 16),  # épaule d - coude d - poignet d

    # Jambes
    (23, 25, 27),  # hanche g - genou g - cheville g
    (24, 26, 28),  # hanche d - genou d - cheville d

    # Orientation tronc
    (11, 23, 25),  # épaule g - hanche g - genou g
    (12, 24, 26),  # épaule d - hanche d - genou d

    # Épaules
    (13, 11, 12),  # coude g - épaule g - épaule d
    (14, 12, 11),  # coude d - épaule d - épaule g
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


def play_normalized_video_from_csv(csv_file, video_name, window_name="Normalized Video",
                                   size=512, scale=100, axis_length=80):
    """
    Affiche les frames normalisées + les axes X,Y,Z sur le canvas.
    """

    df = pd.read_csv(csv_file)

    # Filtrer la vidéo
    video_df = df[df['video_name'] == video_name].sort_values('frame_number')
    if len(video_df) == 0:
        print(f"Aucune ligne trouvée pour {video_name}")
        return

    # Colonnes des landmarks
    lm_cols = [col for col in df.columns if col.startswith("lm_")]

    # Convertir en frames
    frames = [
        row[lm_cols].values.astype(float).reshape(33,3)
        for _, row in video_df.iterrows()
    ]

    center = (size // 2, size // 2)

    while True:
        for pts in frames:
            canvas = np.zeros((size, size, 3), dtype=np.uint8)

            # ---------------------------
            # 1) AFFICHAGE DES AXES
            # ---------------------------
            # Axe X = rouge
            cv2.arrowedLine(canvas, center,
                            (center[0] + axis_length, center[1]),
                            (0, 0, 255), 2, tipLength=0.2)

            # Axe Y = vert
            cv2.arrowedLine(canvas, center,
                (center[0], center[1] - axis_length),
                (0, 255, 0), 2, tipLength=0.2)

            # Axe Z = bleu (diagonale)
            cv2.arrowedLine(canvas, center,
                            (center[0] - axis_length, center[1] - axis_length),
                            (255, 0, 0), 2, tipLength=0.2)

            # Labels
            cv2.putText(canvas, "X", (center[0] + axis_length + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(canvas, "Y", (center[0], center[1] + axis_length + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(canvas, "Z", (center[0] - axis_length - 20, center[1] - axis_length - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            # ---------------------------
            # 2) LIGNES DU POSE
            # ---------------------------
            for a, b in POSE_CONNECTIONS:
                x1, y1, _ = pts[a]
                x2, y2, _ = pts[b]
                p1 = (int(x1*scale + size/2), int(-y1*scale + size/2))
                p2 = (int(x2*scale + size/2), int(-y2*scale + size/2))
                cv2.line(canvas, p1, p2, (0,255,255), 2)

            # ---------------------------
            # 3) LANDMARKS
            # ---------------------------
            for x, y, z in pts:
                px = int(x*scale + size/2)
                py = int(-y*scale + size/2)
                # Z positif = bleu clair, Z négatif = bleu foncé
                color = (255, 0, max(0, min(255, int(255*(z+1)/2))))
                cv2.circle(canvas, (px, py), 4, color, -1)

            # ---------------------------
            # 4) AFFICHAGE
            # ---------------------------
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return




def find_best_start_frame(X1, X2, window=10, start_v1=10):
    # indices dans X
    landmark_ids=[k for k in range(33)]  # utiliser tous les landmarks
    landmark_dim_idx = []
    for lm in landmark_ids:
        base = lm * 3
        landmark_dim_idx.extend([base, base+1, base+2])

    # positions
    X1_filt = X1[:, landmark_dim_idx]
    X2_filt = X2[:, landmark_dim_idx]

    # vitesses
    V1 = np.diff(X1_filt, axis=0, prepend=X1_filt[:1])
    V2 = np.diff(X2_filt, axis=0, prepend=X2_filt[:1])

    # fenêtre de V1
    win1_pos = X1_filt[start_v1:start_v1+window]
    win1_vel = V1[start_v1:start_v1+window]
    feat1 = np.hstack([win1_pos, win1_vel])

    best_dist = float("inf")
    best_frame = 0

    for start in range(0, len(X2) - window - 1):
        win2_pos = X2_filt[start:start+window]
        win2_vel = V2[start:start+window]

        # --- correction : détecter inversion ---
        corr = np.mean(np.sum(win1_vel * win2_vel, axis=1))
        if corr < 0:
            win2_vel = -win2_vel  # inversion automatique

        feat2 = np.hstack([win2_pos, win2_vel])

        dist = np.linalg.norm(feat1 - feat2) / window

        if dist < best_dist:
            best_dist = dist
            best_frame = start

    return best_frame, best_dist


#================ Détection des répétitions

def detect_repetitions_ex(X, exercise, min_dist=40):
    """
    Détection des répétitions basée sur :
        • landmark fourni par exercise.landmark_id
        • axis (0=x,1=y,2=z)
        • opti : -1 → minima, 1 → maxima

    Args:
        X : tableau [frames, 99] = 33 landmarks * 3 coords
        exercise : ExerciseConfig
    """

    idx = exercise.landmark_id
    axis = exercise.axis        # 0,1 ou 2
    opti = exercise.opti        # -1 pour minima, +1 pour maxima

    # ---- 1) Extraction du signal ciblé ----
    coord = X[:, idx * 3 + axis]
    # ---- 2) Lissage ----
    coord_smooth = savgol_filter(coord, window_length=21, polyorder=3)

    # ---- 3) Détection des peaks ----
    # si opti = -1 → minima → on inverse le signal - - si opti =  1 → maxima → normal
    signal = coord_smooth * opti

    peaks, _ = find_peaks(signal, distance=min_dist)
    peak_values = signal[peaks]       

    min_amp = exercise.sensibility * (coord_smooth.max() - coord_smooth.min())
    # print(peaks)
    # ---- 4) Filtrage amplitude ----
    good_peaks = [0]
    for p in peaks:
        # print(f"max{peak_values.max()}, cord{coord_smooth[p]}, min {min_amp}")
        if peak_values.max() - signal[p] < min_amp:
            good_peaks.append(p)

    # plt.figure(figsize=(12,5))
    # plt.plot(coord, label="coord originale", alpha=0.5)
    # plt.plot(signal, label="coord lissée")
    # plt.scatter(good_peaks, coord_smooth[good_peaks], color='red', label='Peaks détectés')
    # plt.title(f"Détection des répétitions pour {exercise.name}")
    # plt.xlabel("Frame")
    # plt.ylabel("Position sur axe choisi")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return np.array(good_peaks)


def compute_dtw_for_rep(X1, X2, start_v1, end_v1, start_v2):
    """
    Recalcule le DTW entre :
      - segment V1 [start_v1 : end_v1]
      - V2 à partir de start_v2

    Retourne un dict : frame_v1 → frame_v2
    """
    X1_seg = X1[start_v1:end_v1]
    X2_seg = X2[start_v2:]

    vel1 = np.diff(X1_seg, axis=0)
    vel2 = np.diff(X2_seg, axis=0)

    _, path = fastdtw(vel1, vel2, dist=euclidean)

    # IMPORTANT : le mapping doit être global dans la timeline de V1
    dtw_dict = {start_v1 + i1: start_v2 + i2 for i1, i2 in path}
    return dtw_dict


def cosine_sim(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den if den > 0 else 0


def compute_rep_cosine_similarity(X1, X2, dtw_dict, rep_start, rep_end):
    sims = []

    for i in range(rep_start, rep_end):
        if i not in dtw_dict:
            continue
        j = dtw_dict[i]
        if j >= len(X2):
            continue

        # On compare l’ensemble des landmarks (33×3 = 99 dims)
        v1 = X1[i]
        v2 = X2[j]

        sims.append(cosine_sim(v1, v2))

    if len(sims) == 0:
        return 0.0

    return np.mean(sims)


def joint_angle(a, b, c):
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-6:
        return 0.0
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)  # radians

def frame_to_angle_vector(frame):
    angles = []
    for (a, b, c) in ANGLE_TRIPLETS:
        angles.append(joint_angle(frame[a], frame[b], frame[c]))
    return np.array(angles, dtype=np.float32)

def angle_similarity(a1, a2):
    return cosine_sim(a1, a2)


def compute_rep_angle_similarity(X1, X2, dtw_dict, rep_start, rep_end):
    sims = []

    for i in range(rep_start, rep_end):
        if i not in dtw_dict:
            continue
        j = dtw_dict[i]
        if j >= len(X2):
            continue

        # Convertir les frames en vecteurs d'angles
        ang1 = frame_to_angle_vector(X1[i])
        ang2 = frame_to_angle_vector(X2[j])

        # Similarité cosinus sur les angles
        sims.append(angle_similarity(ang1, ang2))

    if len(sims) == 0:
        return 0.0

    return float(np.mean(sims))


def play_videos_dtw_2(csv_file, name_exercise, n_v1, n_v2, mode="video"):
    video1_file = "data/data-btc/"+name_exercise+"/"+name_exercise+"_"+n_v1+".mp4"
    video2_file = "data/data-btc/"+name_exercise+"/"+name_exercise+"_"+n_v2+".mp4"
    name_v1 =name_exercise+"_"+n_v1+".mp4"
    name_v2 =name_exercise+"_"+n_v2+".mp4"

    # --- 1) Lecture des landmarks ---
    df = pd.read_csv(csv_file)
    v1 = df[df['video_name'] == name_v1].sort_values('frame_number')
    v2 = df[df['video_name'] == name_v2].sort_values('frame_number')

    landmark_cols = [f"lm_{i}_{axis}" for i in range(33) for axis in ['x','y','z']]
    X1 = v1[landmark_cols].to_numpy(float)
    X2 = v2[landmark_cols].to_numpy(float)

    # --- 2) Alignement initial ---
    start_frame_v2, _ = find_best_start_frame(X1, X2, window=10)

    # --- 3) Détection des répétitions ---
    rep_starts = detect_repetitions_ex(X1,EXERCISES[name_exercise])
    print("Detected repetitions:", rep_starts)

    if len(rep_starts) == 0:
        rep_starts = [0]

    # On ajoute la fin de la vidéo pour faciliter les boucles
    rep_starts = list(rep_starts) + [len(X1)]

    # --- 4) Préparation vidéos ---
    cap1 = cv2.VideoCapture(video1_file)
    cap2 = cv2.VideoCapture(video2_file)

    # --- 5) Initialiser première répétition ---
    current_rep = 0
    rep_start_v1 = rep_starts[0]
    rep_end_v1   = rep_starts[1]
    
    if(rep_starts[1]-rep_starts[0] <2):
        rep_end_v1 +=1
        
    dtw_dict = compute_dtw_for_rep(X1, X2, rep_start_v1, rep_end_v1, start_frame_v2)

    frame_idx = 0

    while True:

        ret1, frame1 = cap1.read()
        if not ret1:
            break

        # --- Nouvelle répétition détectée ---
        if frame_idx == rep_end_v1:

            current_rep += 1
            if current_rep >= len(rep_starts) - 1:
                break  # plus de reps

            rep_start_v1 = rep_starts[current_rep]
            rep_end_v1 = rep_starts[current_rep + 1]

            
            print(f">>> Nouvelle répétition #{current_rep} : frames V1 {rep_start_v1} → {rep_end_v1}")
            if current_rep > 0:  #Calcul de la cosine similarity
                sim = compute_rep_cosine_similarity(
                    X1, X2, dtw_dict, rep_starts[current_rep-1], rep_starts[current_rep]
                )
                print(f"Cosine similarity repetition {current_rep-1}: {sim:.4f}")

            # Reset V2 au début
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Recalcul DTW de la répétition courante
            dtw_dict = compute_dtw_for_rep(X1, X2, rep_start_v1, rep_end_v1, 0)

        # --- Mapping local DTW ---
        i_v2 = dtw_dict.get(frame_idx, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i_v2)

        ret2, frame2 = cap2.read()
        if not ret2:
            frame2 = np.zeros_like(frame1)

        # --- Resize pour la cohérence ---
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        if h1 != h2:
            frame2 = cv2.resize(frame2, (int(w2*(h1/h2)), h1))

        # --- Dessin des landmarks ---
        for x, y, _ in X1[frame_idx].reshape(-1, 3):
            cv2.circle(frame1, (int(x*w1), int(y*h1)), 3, (0,255,0), -1)

        if i_v2 < len(X2):
            for x, y, _ in X2[i_v2].reshape(-1, 3):
                cv2.circle(frame2, (int(x*w2), int(y*h2)), 3, (0,0,255), -1)

        combined = np.hstack([frame1, frame2])
        cv2.imshow("V1 / V2 DTW with reps", combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
def draw_skeleton(pts, canvas, size=512):
    pts = pts.reshape(-1, 3)
    h, w = canvas.shape[:2]

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        x1, y1, _ = pts[a]
        x2, y2, _ = pts[b]

        p1 = (int((x1 * 100 + w/2)), int((y1 * 100 + h/2)))
        p2 = (int((x2 * 100 + w/2)), int((y2 * 100 + h/2)))

        cv2.line(canvas, p1, p2, (0, 255, 255), 2)

    # Draw points
    for (x, y, z) in pts:
        px = int((x * 100 + w/2))
        py = int((y * 100 + h/2))
        cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)

    return canvas


    
def play_videos_dtw(csv_file, name_exercise, n_v1, n_v2, mode="video"):
    """
    mode = "video"     → vidéos + landmarks
    mode = "skeleton"  → squelette normalisé uniquement
    mode = "none"      → aucune visualisation
    """

    video1_file = f"data/data-btc/{name_exercise}/{name_exercise}_{n_v1}.mp4"
    video2_file = f"data/data-btc/{name_exercise}/{name_exercise}_{n_v2}.mp4"
    name_v1 = f"{name_exercise}_{n_v1}.mp4"
    name_v2 = f"{name_exercise}_{n_v2}.mp4"

    # --- Lecture CSV ---
    df = pd.read_csv(csv_file)
    v1 = df[df['video_name'] == name_v1].sort_values('frame_number')
    v2 = df[df['video_name'] == name_v2].sort_values('frame_number')

    # Extraction landmarks
    landmark_cols = [f"lm_{i}_{axis}" for i in range(33) for axis in ['x','y','z']]
    X1 = v1[landmark_cols].to_numpy(float)
    X2 = v2[landmark_cols].to_numpy(float)

    # Alignement
    start_frame_v2, _ = find_best_start_frame(X1, X2, window=10)

    # Détection répétitions
    rep_starts = detect_repetitions_ex(X1, EXERCISES[name_exercise])
    print("Detected repetitions:", rep_starts)

    if len(rep_starts) == 0:
        rep_starts = [0]

    rep_starts = list(rep_starts) + [len(X1)]

    # Vidéos si mode="video"
    if mode == "video":
        cap1 = cv2.VideoCapture(video1_file)
        cap2 = cv2.VideoCapture(video2_file)

    # Première répétition
    current_rep = 0
    rep_start_v1 = rep_starts[0]
    rep_end_v1 = rep_starts[1]

    if rep_end_v1 - rep_start_v1 < 2:
        rep_end_v1 += 1

    dtw_dict = compute_dtw_for_rep(X1, X2, rep_start_v1, rep_end_v1, start_frame_v2)

    frame_idx = 0

    # --- BOUCLE D'AFFICHAGE ---
    while True:
        # MODE : squelette → recréer une frame seulement avec landmarks
        if mode == "skeleton":
            frame1 = np.zeros((512, 512, 3), dtype=np.uint8)
            frame2 = np.zeros((512, 512, 3), dtype=np.uint8)

        # MODE : vidéo → lire les vraies frames
        elif mode == "video":
            ret1, frame1 = cap1.read()
            if not ret1:
                break

        # MODE : none → ne calcule que l’alignement
        elif mode == "none":
            frame_idx += 1
            if frame_idx >= len(X1):
                break
            continue

        # Si on dépasse la rep
        if frame_idx == rep_end_v1:
            current_rep += 1
            if current_rep >= len(rep_starts) - 1:
                break
            if current_rep > 0:
                sim = compute_rep_angle_similarity(
                    X1, X2, dtw_dict,
                    rep_starts[current_rep - 1],   # début rep précédente
                    rep_starts[current_rep]        # début rep actuelle
                )
                print(f"ANgle cosine similarity repetition {current_rep-1}: {sim:.4f}")
                rep_start_v1 = rep_starts[current_rep]
                rep_end_v1 = rep_starts[current_rep+1]

                print(f">>> Nouvelle répétition #{current_rep}: {rep_start_v1} → {rep_end_v1}")

            dtw_dict = compute_dtw_for_rep(X1, X2, rep_start_v1, rep_end_v1, 0)

        # Frame correspondante dans V2 après DTW
        i_v2 = dtw_dict.get(frame_idx, 0)

        # --- MODE VIDEO ---
        if mode == "video":
            cap2.set(cv2.CAP_PROP_POS_FRAMES, i_v2)
            ret2, frame2 = cap2.read()
            if not ret2:
                frame2 = np.zeros_like(frame1)

        # --- MODE SKELETON ---
        if mode == "skeleton":
            frame1 = draw_skeleton(X1[frame_idx], frame1)
            frame2 = draw_skeleton(X2[i_v2], frame2)

        if mode in ("video", "skeleton"):
            combined = np.hstack([frame1, frame2])
            cv2.imshow("DTW Comparison", combined)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        frame_idx += 1

    if mode == "video":
        cap1.release()
        cap2.release()
    cv2.destroyAllWindows()




def plot_landmark_velocity(csv_file, video_file, video_name, landmark_id=15):
    """
    Plot vx, vy, vz over time for a specific landmark using the normalized CSV landmarks.

    Parameters
    ----------
    csv_file : str
        Path to CSV containing landmarks.
    video_file : str
        Path to the actual video file (to extract FPS).
    video_name : str
        Name of the video as stored in the CSV.
    landmark_id : int
        Landmark index (default = 15 = main gauche).
    """

    # --- Load CSV ---
    df = pd.read_csv(csv_file)
    df = df[df["video_name"] == video_name].sort_values("frame_number")

    if len(df) == 0:
        print("Aucune donnée pour cette vidéo.")
        return

    # --- Extract landmark coords ---
    cols = [f"lm_{landmark_id}_x", f"lm_{landmark_id}_y", f"lm_{landmark_id}_z"]
    XYZ = df[cols].to_numpy(float)

    # --- Compute velocities ---
    V = np.diff(XYZ, axis=0, prepend=XYZ[:1])
    vx, vy, vz = V[:,0], V[:,1], V[:,2]

    # --- Load FPS (time axis) ---
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    t = np.arange(len(vx)) / fps

    # --- Plot ---
    plt.figure(figsize=(12,5))
    plt.plot(t, vx, label="vx")
    plt.plot(t, vy, label="vy")
    plt.plot(t, vz, label="vz")

    plt.title(f"Vitesses du landmark {landmark_id} ({video_name})")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (unités normalisées/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_landmark_position(csv_file, video_file, video_name, landmark_id=15):
    """
    Plot x(t), y(t), z(t) over time for a specific landmark using the normalized CSV landmarks.

    Parameters
    ----------
    csv_file : str
        Path to CSV containing landmarks.
    video_file : str
        Path to the actual video file (to extract FPS).
    video_name : str
        Name of the video in the CSV.
    landmark_id : int
        Landmark index (default = 15 = main gauche).
    """

    # --- Load CSV ---
    df = pd.read_csv(csv_file)
    df = df[df["video_name"] == video_name].sort_values("frame_number")

    if len(df) == 0:
        print("Aucune donnée pour cette vidéo.")
        return

    # --- Extract landmark positions ---
    cols = [f"lm_{landmark_id}_x", f"lm_{landmark_id}_y", f"lm_{landmark_id}_z"]
    XYZ = df[cols].to_numpy(float)
    x, y, z = XYZ[:,0], XYZ[:,1], XYZ[:,2]

    # --- Load FPS → time axis ---
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    t = np.arange(len(x)) / fps

    # --- Plot ---
    plt.figure(figsize=(12,5))
    plt.plot(t, x, label="x")
    plt.plot(t, y, label="y")
    plt.plot(t, z, label="z")

    plt.title(f"Positions du landmark {landmark_id} ({video_name})")
    plt.xlabel("Temps (s)")
    plt.ylabel("Position (unités normalisées)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

    ##### ==== TEST FOR BARCELL BICEPS
    # play_normalized_video_from_csv("data/data-btc/full_landmarks_dataset_features2.csv", "leg extension_8.mp4")
    # play_normalized_video_from_csv("data/data-btc/full_landmarks_dataset_features2.csv", "leg extension_22.mp4")
    # play_videos_dtw_2("data/data-btc/full_landmarks_dataset_features.csv","data/data-btc/barbell biceps curl/barbell biceps curl_34.mp4","data/data-btc/barbell biceps curl/barbell biceps curl_27.mp4", "barbell biceps curl_34.mp4", "barbell biceps curl_27.mp4")
    # play_videos_dtw("data/data-btc/full_landmarks_dataset_features.csv","barbell biceps curl","34", "36", mode = "video")
    play_videos_dtw("data/data-btc/full_landmarks_dataset_features.csv","leg extension","19", "22", mode = "video")
    
    # play_videos_dtw_2("data/data-btc/full_landmarks_dataset_features.csv","leg extension","11", "22")
    # play_videos_dtw_2("data/data-btc/full_landmarks_dataset_features.csv","leg extension","19", "22")
    
    #Marche super pour v1 = 27 et v2 = 36
    # Tester pour 33 et 36 -> L'un a ses variations selon vx, et pas vz.
    # Marche quand commence en bas, pas quand commence en haut
    # plot_landmark_position("data/data-btc/full_landmarks_dataset_features.csv", "data/data-btc/bench press/bench press_2.mp4", "bench press_2.mp4", landmark_id=15)
    # plot_landmark_position("data/data-btc/full_landmarks_dataset_features.csv", "data/data-btc/leg extension/leg extension_22.mp4", "leg extension_22.mp4", landmark_id=29)
    # plot_landmark_position("data/data-btc/full_landmarks_dataset_features.csv", "data/data-btc/leg extension/leg extension_8.mp4", "leg extension_8.mp4", landmark_id=29)

    ##### ==== TEST FOR PLANK
    # play_videos_dtw("data/data-btc/full_landmarks_dataset_features.csv","data/data-btc/lateral raise/lateral raise_5.mp4","data/data-btc/lateral raise/lateral raise_35.mp4", "lateral raise_5.mp4", "lateral raise_35.mp4")
    #->> Même problème
    
    #=====-> Compter le nombre de reps et donner un score par reps, pour pouvoir ensuite en afficher un graphique
    #=====-> Mieux analyser le nombre de reps
    #=====-> Pour 12 et 22, j'ai encore des vidéos désynchronisées
    #. ===== -> Beaucoup de variations multi axes de. lavitesse, peut-être pas le meilleur truc à étudier
