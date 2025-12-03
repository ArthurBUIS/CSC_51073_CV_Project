import pose_detection as pdec
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
import csv


def missing_landmarks_ratio(landmarks):
    """
    Calcule le pourcentage moyen de points manquants dans la vidéo.
    """
    total_points = np.prod(landmarks.shape)
    missing_points = np.isnan(landmarks).sum()
    return missing_points / total_points


def count_total_frames_in_folder(folder_path, extensions=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Compte le nombre total de frames dans toutes les vidéos d'un dossier.

    Parameters
    ----------
    folder_path : str
        Chemin du dossier contenant les vidéos.
    extensions : tuple
        Extensions de fichiers à considérer.

    Returns
    -------
    total_frames : int
        Nombre total de frames dans toutes les vidéos valides.
    details : dict
        Détails par fichier {nom_fichier: nombre_de_frames}
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"❌ Dossier introuvable : {folder_path}")

    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    if not video_files:
        print("⚠️ Aucune vidéo trouvée dans le dossier.")
        return 0, {}

    total_frames = 0
    details = {}

    print(f"🔍 Analyse de {len(video_files)} vidéos dans '{folder_path}'...\n")

    for file in tqdm(video_files, desc="Counting frames", ncols=80):
        path = os.path.join(folder_path, file)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"⚠️ Impossible d'ouvrir : {file}")
            continue

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames += frames
        details[file] = frames
        cap.release()

    print(f"\n✅ Total : {total_frames:,} frames détectées dans {len(details)} vidéos.")
    return total_frames, details

import numpy as np

def pose_cleanliness_score(landmarks, weights=None):
    """
    Évalue la propreté d'une séquence de landmarks MediaPipe.

    Parameters
    ----------
    landmarks : np.ndarray
        Tableau de forme (frames, 33, 3) contenant les coordonnées normalisées [0,1].
        Peut contenir des NaN si la détection échoue sur certaines frames.
    weights : dict (optionnel)
        Pondération des critères, ex: {"missing": 0.4, "out_of_bounds": 0.4, "stability": 0.2}

    Returns
    -------
    score : float
        Score global de qualité entre 0 et 1.
    details : dict
        Détails des sous-scores {"missing_ratio": ..., "out_of_bounds_ratio": ..., "instability": ...}
    """

    if landmarks.ndim != 3 or landmarks.shape[1] != 33 or landmarks.shape[2] != 3:
        raise ValueError("landmarks doit être de forme (frames, 33, 3).")

    # Valeurs par défaut des pondérations
    weights = weights or {"missing": 0.4, "out_of_bounds": 0.4, "stability": 0.2}

    # === 1. Points manquants (NaN) ===
    missing_ratio = np.isnan(landmarks).sum() / landmarks.size

    # === 2. Points hors cadre ===
    # (on ignore les NaN pour ce test)
    valid = ~np.isnan(landmarks)
    out_of_bounds = ((landmarks < 0) | (landmarks > 1)) & valid
    out_of_bounds_ratio = out_of_bounds.sum() / valid.sum() if valid.sum() > 0 else 1.0

    # === 3. Instabilité (vitesse moyenne des points d'une frame à l'autre) ===
    diffs = np.diff(landmarks, axis=0)
    diffs[np.isnan(diffs)] = 0
    frame_motion = np.linalg.norm(diffs, axis=2)
    frame_motion = np.mean(frame_motion ** 5, axis=1) ** (1/5) # Use of RMS to show strong cha
    instability = np.clip(frame_motion.mean() * 10, 0, 1)  # normalisation grossière

    # === 4. Score global ===
    sub_scores = {
        "missing_ratio": missing_ratio,
        "out_of_bounds_ratio": out_of_bounds_ratio,
        "instability": instability
    }

    score = 1 - (
        weights["missing"] * missing_ratio +
        weights["out_of_bounds"] * out_of_bounds_ratio +
        weights["stability"] * instability
    )

    score = np.clip(score, 0, 1)
    return score, sub_scores


def build_landmark_dataset(computed_folders,root_dir="data-btc"):
    """
    Parcourt tous les sous-dossiers du dataset, filtre les vidéos
    par cleanliness_score > 0.4, extrait les landmarks interpolés,
    et ajoute les données dans full_landmarks_dataset.csv.

    Le CSV final contient :
        video_name, total_frames, frame_number, 33×(x, y, z)
    """

    output_csv = os.path.join("data/data-btc/full_landmarks_dataset.csv")

    #Création du csv si n'existe pas
    if not os.path.exists(output_csv):
        print("📄 Création du fichier CSV global...")
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            header = ["video_name", "total_frames", "frame_number"]
            for i in range(33):
                header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"]
            writer.writerow(header)

    #Parcours des dossiers
    for exercise_folder in os.listdir(root_dir):
        if(exercise_folder) not in computed_folders:
            exercise_path = os.path.join(root_dir, exercise_folder)
            if not os.path.isdir(exercise_path):
                continue

            report_path = os.path.join(exercise_path, "landmarks_quality_report.csv")
            if not os.path.exists(report_path):
                print(f"❌ Pas de report dans {exercise_folder}")
                continue

            df_report = pd.read_csv(report_path)

            # Filtrer les vidéos suffisamment propres
            valid_videos = df_report[df_report["cleanliness_score"] > 0.4]
            if valid_videos.empty:
                print(f"⚠️ Aucune vidéo propre dans {exercise_folder}")
                continue

            # Traitement des vidéos valides
            for _, row in valid_videos.iterrows():

                video_name = row["video"]
                video_path = os.path.join(exercise_path, video_name)

                if not os.path.exists(video_path):
                    print(f"❌ Vidéo introuvable : {video_path}")
                    continue

                print(f"\n➡️ Traitement de  : {video_name}")

                # Appel de ta fonction (renvoie shape = [frames, 33, 3])
                landmarks = pdec.extract_pose_from_video_interpolated(
                    video_path,
                    show_interpolated=False
                )
                
                cap = cv2.VideoCapture(video_path)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if landmarks is None:
                    print(f"⚠️ Aucun landmark détecté dans {video_name}")
                    continue

                total_frames = landmarks.shape[0]

                with open(output_csv, "a", newline="") as f:
                    writer = csv.writer(f)

                    for frame_idx in range(total_frames):

                        row_out = [video_name, total_frames, frame_idx, width, height, exercise_folder]
                        frame_lm = landmarks[frame_idx]  # shape (33,3)

                        for (x, y, z) in frame_lm:
                            row_out.extend([x, y, z])

                        writer.writerow(row_out)
            print(f"\n✅ Dataset complet pour {exercise_folder} !")
    print("\n✅ Dataset complet construit avec succès !")
    
    
def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return np.arccos(cos_theta) * 180/np.pi


def compute_dataset_features(input_csv,output_csv,first_line=0):
    """
    Fonction pour calculer des features supplémentaires sur le dataset de landmarks. Et normalise + rotate les landmarks.
    """
    
    L_SH, R_SH = 11, 12
    L_HP, R_HP = 23, 24
    L_ELB, R_ELB = 13, 14
    L_WR, R_WR = 15, 16
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    HEAD = 0  # point tête
    TORSO = 12

    meta_cols = ['video_name','total_frames','frame_number','width','height','label']
    lm_cols = [f"lm_{i}_{c}" for i in range(33) for c in ['x','y','z']]

    # --- Lecture du CSV d'origine en chunks pour grande taille ---
    chunk_size = 1000  # ajustable selon RAM

    with open(output_csv, mode='w', newline='') as f_out:
        # Colonnes des features
        feature_cols = [
            'a_elb_L','a_elb_R',        # angle coude gauche / droit
            'a_sh_L','a_sh_R',          # angle épaule gauche / droit
            'a_kn_L','a_kn_R',          # angle genou gauche / droit
            'a_hp_L','a_hp_R',          # angle hanche gauche / droit
            'a_wr_L','a_wr_R',          # angle poignet gauche / droit
            'a_an_L','a_an_R',          # angle cheville gauche / droit
            'd_sh','d_hp',               # distance épaules, hanches
            'd_head_sh','d_wr_hp',       # distance tête-épaule, main-hanche
            'd_ank_hp','d_wr','d_kn'    # distance pied-hanche, main-main, genou-genou
        ]
        
        writer = csv.DictWriter(f_out, fieldnames=meta_cols + lm_cols + feature_cols)
        writer.writeheader()
        
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
            grouped = chunk.groupby('video_name')
            
            for video_name, video_df in grouped:
                T = len(video_df)
                # reconstruire array (T,33,3)
                lm_cols = [col for col in video_df.columns if col.startswith("lm_")]
                landmarks = video_df[lm_cols].astype(float).values.reshape(T, 33, 3)
                
                # --- Normalize ---
                hip_center = landmarks[:, [23,24], :].mean(axis=1)
                landmarks = landmarks - hip_center[:, None, :]
                
                shoulders = landmarks[:, [11,12], :].mean(axis=1)
                torso_len = np.linalg.norm(shoulders - hip_center, axis=1)
                torso_len[torso_len == 0] = 1
                landmarks = landmarks / torso_len[:, None, None]
                
                # --- Rotate ---
                for t in range(T):
                    pts = landmarks[t]
                    left_shoulder  = pts[L_SH]
                    right_shoulder = pts[R_SH]
                    left_hip       = pts[L_HP]
                    right_hip      = pts[R_HP]
                    shoulder_center = (left_shoulder + right_shoulder)/2
                    hip_center_frame = (left_hip + right_hip)/2

                    x_axis = right_shoulder - left_shoulder
                    x_axis /= np.linalg.norm(x_axis)
                    torso = shoulder_center - hip_center_frame
                    torso /= np.linalg.norm(torso)
                    z_axis = np.cross(x_axis, torso)
                    z_norm = np.linalg.norm(z_axis)
                    if z_norm < 1e-6:
                        z_axis = np.array([0,0,1])
                    else:
                        z_axis /= z_norm
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)
                    if np.dot(y_axis, torso) > 0:
                        y_axis = -y_axis
                    R_mat = np.vstack([x_axis, y_axis, z_axis]).T
                    landmarks[t] = np.array([R_mat @ (p - hip_center_frame) for p in pts])
                
                # --- Calcul des features et écriture ligne par ligne ---
                for i, (_, row) in enumerate(video_df.iterrows()):
                    pts = landmarks[i]
                    new_row = {col: row[col] for col in meta_cols}
                    
                    # Landmarks
                    for j in range(33):
                        new_row[f'lm_{j}_x'] = pts[j,0]
                        new_row[f'lm_{j}_y'] = pts[j,1]
                        new_row[f'lm_{j}_z'] = pts[j,2]
                    
                    # Angles
                    new_row['a_elb_L'] = angle_between(pts[L_SH], pts[L_ELB], pts[L_WR])
                    new_row['a_elb_R'] = angle_between(pts[R_SH], pts[R_ELB], pts[R_WR])
                    new_row['a_sh_L']  = angle_between(pts[L_ELB], pts[L_SH], pts[TORSO])
                    new_row['a_sh_R']  = angle_between(pts[R_ELB], pts[R_SH], pts[TORSO])
                    new_row['a_kn_L']  = angle_between(pts[L_HP], pts[L_KNEE], pts[L_ANK])
                    new_row['a_kn_R']  = angle_between(pts[R_HP], pts[R_KNEE], pts[R_ANK])
                    new_row['a_hp_L']  = angle_between(pts[L_SH], pts[L_HP], pts[L_KNEE])
                    new_row['a_hp_R']  = angle_between(pts[R_SH], pts[R_HP], pts[R_KNEE])
                    new_row['a_wr_L']  = angle_between(pts[L_ELB], pts[L_WR], pts[L_WR]+np.array([1,0,0]))
                    new_row['a_wr_R']  = angle_between(pts[R_ELB], pts[R_WR], pts[R_WR]+np.array([1,0,0]))
                    new_row['a_an_L']  = angle_between(pts[L_KNEE], pts[L_ANK], pts[L_ANK]+np.array([0,0,1]))
                    new_row['a_an_R']  = angle_between(pts[R_KNEE], pts[R_ANK], pts[R_ANK]+np.array([0,0,1]))

                    # Distances
                    new_row['d_sh']       = np.linalg.norm(pts[L_SH]-pts[R_SH])
                    new_row['d_hp']       = np.linalg.norm(pts[L_HP]-pts[R_HP])
                    new_row['d_head_sh']  = np.linalg.norm(pts[HEAD]-((pts[L_SH]+pts[R_SH])/2))
                    new_row['d_wr_hp']    = np.linalg.norm(pts[L_WR]-pts[L_HP])
                    new_row['d_ank_hp']   = np.linalg.norm(pts[L_ANK]-((pts[L_HP]+pts[R_HP])/2))
                    new_row['d_wr']       = np.linalg.norm(pts[L_WR]-pts[R_WR])
                    new_row['d_kn']       = np.linalg.norm(pts[L_KNEE]-pts[R_KNEE])

                    writer.writerow(new_row)

def compute_dataset_features_2(input_csv,output_csv,first_line=0):
    """
    Fonction pour calculer des features supplémentaires sur le dataset de landmarks. Et normalise + rotate les landmarks.
    """
    
    L_SH, R_SH = 11, 12
    L_HP, R_HP = 23, 24
    L_ELB, R_ELB = 13, 14
    L_WR, R_WR = 15, 16
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    HEAD = 0  # point tête
    TORSO = 12

    meta_cols = ['video_name','total_frames','frame_number','width','height','label']
    lm_cols = [f"lm_{i}_{c}" for i in range(33) for c in ['x','y','z']]

    # --- Lecture du CSV d'origine en chunks pour grande taille ---
    chunk_size = 1000  # ajustable selon RAM

    with open(output_csv, mode='w', newline='') as f_out:
        # Colonnes des features
        feature_cols = [
            'a_elb_L','a_elb_R',        # angle coude gauche / droit
            'a_sh_L','a_sh_R',          # angle épaule gauche / droit
            'a_kn_L','a_kn_R',          # angle genou gauche / droit
            'a_hp_L','a_hp_R',          # angle hanche gauche / droit
            'a_wr_L','a_wr_R',          # angle poignet gauche / droit
            'a_an_L','a_an_R',          # angle cheville gauche / droit
            'd_sh','d_hp',               # distance épaules, hanches
            'd_head_sh','d_wr_hp',       # distance tête-épaule, main-hanche
            'd_ank_hp','d_wr','d_kn'    # distance pied-hanche, main-main, genou-genou
        ]
        
        writer = csv.DictWriter(f_out, fieldnames=meta_cols + lm_cols + feature_cols)
        writer.writeheader()
        
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
            grouped = chunk.groupby('video_name')
            
            for video_name, video_df in grouped:
                T = len(video_df)
                # reconstruire array (T,33,3)
                lm_cols = [col for col in video_df.columns if col.startswith("lm_")]
                landmarks = video_df[lm_cols].astype(float).values.reshape(T, 33, 3)
                
                # --- Normalize ---
                hip_center = landmarks[:, [23,24], :].mean(axis=1)
                landmarks = landmarks - hip_center[:, None, :]
                
                shoulders = landmarks[:, [11,12], :].mean(axis=1)
                torso_len = np.linalg.norm(shoulders - hip_center, axis=1)
                torso_len[torso_len == 0] = 1
                landmarks = landmarks / torso_len[:, None, None]
                
                # --- Rotate ---
                for t in range(T):
                    pts = landmarks[t]
                    left_shoulder  = pts[L_SH]
                    right_shoulder = pts[R_SH]
                    left_hip       = pts[L_HP]
                    right_hip      = pts[R_HP]
                    shoulder_center = (left_shoulder + right_shoulder)/2
                    hip_center_frame = (left_hip + right_hip)/2

                    y_axis = hip_center_frame - shoulder_center  
                    y_axis /= np.linalg.norm(y_axis)
                    shoulders_line = right_shoulder - left_shoulder
                    shoulders_line /= np.linalg.norm(shoulders_line)
                    z_axis = np.cross(shoulders_line, y_axis)
                    z_norm = np.linalg.norm(z_axis)
                    if z_norm < 1e-6:
                        z_axis = np.array([0,0,1])
                    else:
                        z_axis /= z_norm
                    x_axis = np.cross(z_axis, y_axis)
                    x_axis /= np.linalg.norm(x_axis)
                    if np.dot(y_axis, shoulders_line) < 0:
                        x_axis = -x_axis
                    R_mat = np.vstack([x_axis, y_axis, z_axis]).T
                    landmarks[t] = np.array([R_mat @ (p - hip_center_frame) for p in pts])
                
                # --- Calcul des features et écriture ligne par ligne ---
                for i, (_, row) in enumerate(video_df.iterrows()):
                    pts = landmarks[i]
                    new_row = {col: row[col] for col in meta_cols}
                    
                    # Landmarks
                    for j in range(33):
                        new_row[f'lm_{j}_x'] = pts[j,0]
                        new_row[f'lm_{j}_y'] = pts[j,1]
                        new_row[f'lm_{j}_z'] = pts[j,2]
                    
                    # Angles
                    new_row['a_elb_L'] = angle_between(pts[L_SH], pts[L_ELB], pts[L_WR])
                    new_row['a_elb_R'] = angle_between(pts[R_SH], pts[R_ELB], pts[R_WR])
                    new_row['a_sh_L']  = angle_between(pts[L_ELB], pts[L_SH], pts[TORSO])
                    new_row['a_sh_R']  = angle_between(pts[R_ELB], pts[R_SH], pts[TORSO])
                    new_row['a_kn_L']  = angle_between(pts[L_HP], pts[L_KNEE], pts[L_ANK])
                    new_row['a_kn_R']  = angle_between(pts[R_HP], pts[R_KNEE], pts[R_ANK])
                    new_row['a_hp_L']  = angle_between(pts[L_SH], pts[L_HP], pts[L_KNEE])
                    new_row['a_hp_R']  = angle_between(pts[R_SH], pts[R_HP], pts[R_KNEE])
                    new_row['a_wr_L']  = angle_between(pts[L_ELB], pts[L_WR], pts[L_WR]+np.array([1,0,0]))
                    new_row['a_wr_R']  = angle_between(pts[R_ELB], pts[R_WR], pts[R_WR]+np.array([1,0,0]))
                    new_row['a_an_L']  = angle_between(pts[L_KNEE], pts[L_ANK], pts[L_ANK]+np.array([0,0,1]))
                    new_row['a_an_R']  = angle_between(pts[R_KNEE], pts[R_ANK], pts[R_ANK]+np.array([0,0,1]))

                    # Distances
                    new_row['d_sh']       = np.linalg.norm(pts[L_SH]-pts[R_SH])
                    new_row['d_hp']       = np.linalg.norm(pts[L_HP]-pts[R_HP])
                    new_row['d_head_sh']  = np.linalg.norm(pts[HEAD]-((pts[L_SH]+pts[R_SH])/2))
                    new_row['d_wr_hp']    = np.linalg.norm(pts[L_WR]-pts[L_HP])
                    new_row['d_ank_hp']   = np.linalg.norm(pts[L_ANK]-((pts[L_HP]+pts[R_HP])/2))
                    new_row['d_wr']       = np.linalg.norm(pts[L_WR]-pts[R_WR])
                    new_row['d_kn']       = np.linalg.norm(pts[L_KNEE]-pts[R_KNEE])

                    writer.writerow(new_row)

if __name__ == "__main__":
    # filename = "data/squat.jpg"  
    # extract_pose_from_image(filename)
    #count_total_frames_in_folder("dataset/barbell biceps curl")

    # landmarks = pdec.extract_pose_from_video_interpolated("data/data-btc/barbell biceps curl/barbell biceps curl_52.mp4", False)
    # print(pose_cleanliness_score(landmarks, weights={"missing": 0, "out_of_bounds": 0.3, "stability": 0.7}))
    # parent_folder = "data/data-crawl"
    # results = []

    # # Lister uniquement les dossiers valides
    # to_remove = ["barbell biceps curl", "russian twist", "romanian deadlift", "hip thrust","plank","leg raises", "pull Up","bench press", "deadlift", "t bar row","incline bench press","decline bench press", "leg extension","hammer curl", "push-up", "squat", "tricep dips", "tricep Pushdown", "lat pulldown","chest fly machine", "shoulder press"]  # dossiers à ignorer
    # folder_list = [
    #     f for f in os.listdir(parent_folder)
    #     if os.path.isdir(os.path.join(parent_folder, f)) and f not in to_remove
    # ]
    

    # for folder in folder_list:
    #     folder_path = os.path.join(parent_folder, folder)
    #     print("Processing folder:", folder_path)

    #     # Lister uniquement les fichiers vidéo
    #     video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi", ".mov"))]

    #     for file in tqdm(video_files, desc="Processing videos", ncols=80):
    #         path = os.path.join(folder_path, file)
    #         try:
    #             landmarks = pdec.extract_pose_from_video_interpolated(path, show_interpolated=False)
    #             if landmarks is None:
    #                 continue

    #             score, details = pose_cleanliness_score(
    #                 landmarks, weights={"missing":0, "out_of_bounds":0.3, "stability":0.7}
    #             )

    #             results.append({
    #                 "video": file,
    #                 "folder": folder,
    #                 "cleanliness_score": score,
    #                 "missing_ratio": details["missing_ratio"],
    #                 "out_of_bounds_ratio": details["out_of_bounds_ratio"],
    #                 "instability": details["instability"]
    #             })
    #         except Exception as e:
    #             print(f"⚠️ Erreur avec {file}: {e}")
    #     # Convertir en DataFrame et sauvegarder
    #     df = pd.DataFrame(results)
    #     csv_path = os.path.join(folder_path, "landmarks_quality_report.csv")
    #     df.to_csv(csv_path, index=False)
    #     results = []
    #     print(f"\n✅ Résultats sauvegardés dans {csv_path}")
    # computed_folders = ['deadlift', 'hammer curl', 'tricep Pushdown', 'squat', 'tricep dips','lat pulldown', 'push up', 'barbell biceps curl','chest fly machine','incline bench press', 'leg extension', 'shoulder press', 't bar row', 'decline bench press', 'bench press', 'lateral raise']
    # build_landmark_dataset(computed_folders, "data/data-btc")
    
    
    # compute_dataset_features(input_csv="data/data-btc/full_landmarks_dataset.csv", output_csv="data/data-btc/full_landmarks_dataset_features.csv")
    compute_dataset_features_2(input_csv="data/data-btc/full_landmarks_dataset.csv", output_csv="data/data-btc/full_landmarks_dataset_features2.csv")
    
    
    # df = pd.read_csv("data/data-btc/full_landmarks_dataset.csv")
    # df_filtered = df[df["label"] == "barbell biceps curl"]
    # videos_uniques = df_filtered["video_name"].unique()
    # print(videos_uniques)


