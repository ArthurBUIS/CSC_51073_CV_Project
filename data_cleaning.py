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
    df = pd.read_csv("data/data-btc/full_landmarks_dataset.csv")
    df_filtered = df[df["label"] == "barbell biceps curl"]
    videos_uniques = df_filtered["video_name"].unique()
    print(videos_uniques)


