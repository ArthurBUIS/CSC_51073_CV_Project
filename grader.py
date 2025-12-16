from ExerciseClasses import EXERCISES
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import metric
import landmark_extraction as le

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
    good_peaks = []
    for p in peaks:
        # print(f"max{peak_values.max()}, cord{coord_smooth[p]}, min {min_amp}")
        if peak_values.max() - signal[p] < min_amp:
            good_peaks.append(p)
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


def compute_repgrade(df, name_exercise):
    """
    """

    # --- 1. Sélection des vidéos ---
    v1 = df
    
    csv_path_ref = "ref/"+name_exercise+".csv"
    df_ref = pd.read_csv(csv_path_ref)
    v2 = df_ref.sort_values('frame_number')

    # --- 2. Extraction landmarks ---
    landmark_cols = [f"lm_{i}_{axis}" for i in range(33) for axis in ['x','y','z']]
    X1 = v1[landmark_cols].to_numpy(float)
    X2 = v2[landmark_cols].to_numpy(float)

    # --- 3. Alignement ---
    start_frame_v2, _ = find_best_start_frame(X1, X2, window=10)

    # --- 4. Détection répétitions ---
    rep_starts = detect_repetitions_ex(X1, EXERCISES[name_exercise]) #Tableau des répétitions

    if len(rep_starts) == 0:
        rep_starts = [0]

    rep_starts = list(rep_starts)

    # --- 6. DTW sur la première répétition ---
    current_rep = 0
    rep_start_v1 = rep_starts[0]
    rep_end_v1   = rep_starts[1]

    if rep_end_v1 - rep_start_v1 < 2:
        rep_end_v1 += 1

    dtw_dict = compute_dtw_for_rep(
        X1, X2,
        rep_start_v1,
        rep_end_v1,
        start_frame_v2
    )

    frame_idx = 0
    sim_list= []
    
    # --- 7. Boucle d'affichage / calcul ---
    while True:
        if frame_idx >= len(X1):
            break
        # --- Nouvelle répétition ---
        elif frame_idx == rep_end_v1:
            current_rep += 1
            if current_rep >= len(rep_starts) - 1:
                break

            # Similarité angulaire
            sim = metric.compute_rep_angle_similarity(
                X1, X2, dtw_dict,
                rep_starts[current_rep - 1],
                rep_starts[current_rep],
                EXERCISES[name_exercise]
            )
            sim_list.append(sim)

            rep_start_v1 = rep_starts[current_rep]
            rep_end_v1   = rep_starts[current_rep + 1]

            dtw_dict = compute_dtw_for_rep(X1, X2, rep_start_v1, rep_end_v1, 0)

        frame_idx += 1
        
    # --- Dernière répétition (sinon elle manque !) ---
    last_rep = len(rep_starts) - 2
    sim = metric.compute_rep_angle_similarity(
        X1, X2, dtw_dict,
        rep_starts[last_rep],
        rep_starts[last_rep + 1],
        EXERCISES[name_exercise]
    )
    sim_list.append(sim)
    return rep_starts, sim_list
    
    
if __name__ == "__main__":
    filename ="data/data-btc/push-up/push-up_test0.mp4"
    landmarks, df = le.pipe_extract_landmark(filename)
    rep_starts, sim_list = compute_repgrade(df, "push-up")
    print(rep_starts)
    print(sim_list)
    