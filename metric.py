import numpy as np
from ExerciseClasses import EXERCISES


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

landmark_weights = np.array([
    # 0–10 : tête et tronc haut → peu utile pour les pompes
    0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
    # 11–16 : épaules, coudes, poignets → pivot principal du mouvement
    1.0,  # 11 épaule g
    1.0,  # 12 épaule d
    1.0,  # 13 coude g
    1.0,  # 14 coude d
    0.9,  # 15 poignet g
    0.9,  # 16 poignet d
    # 17–22 : tronc bas → utile
    0.5,0.5,0.5,0.5,0.5,0.5,
    # 23–28 : hanches, genoux, chevilles → posture importante
    0.8,  # hanche g
    0.8,  # hanche d
    0.6,0.6,0.4,0.4,  # genoux / chevilles
    # 29–32 : pieds → peu utile
    0.2,0.2,0.2,0.2
], dtype=np.float32)

def cosine_sim(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den if den > 0 else 0

def compute_rep_cosine_similarity(X1, X2, dtw_dict, rep_start, rep_end):
    sims = []

    # weights = np.repeat(landmark_weights, 3)
    weights = np.ones(99, dtype=np.float32)
    
    for i in range(rep_start, rep_end):
        if i not in dtw_dict:
            continue
        j = dtw_dict[i]
        if j >= len(X2):
            continue

        # On compare l’ensemble des landmarks (33×3 = 99 dims)
        v1 = X1[i] * weights
        v2 = X2[j] * weights

        sims.append(cosine_sim(v1, v2))

    if len(sims) == 0:
        return 0.0

    return np.mean(sims)


def joint_angle(A, B, C, debug=False):
    """
    Calcule l'angle ABC en radians.
    B est le sommet.
    """
    BA = A - B
    BC = C - B
    
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    cos_angle = np.dot(BA, BC) / (norm_BA * norm_BC)

    # Clamp numérique
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return angle


def point(frame, i):
    return frame[3*i : 3*i + 3]  

def angle_similarity(a1, a2, exercise, strictness=2, weight_scale=1):
    diff = a1 - a2
    weighted = (exercise.angle_weights / weight_scale) * (np.abs(diff) ** strictness)
    score = np.exp(-np.sum(weighted)) #For discrimination
    
    score = (score - 0.08)*100  # factor ajuste la pente
    score = 1.2 * 1 / (1 + np.exp(-score))
    return score
    return score



def frame_to_angle_vector(frame):
    angles = []
    for (a, b, c) in ANGLE_TRIPLETS:
        A = point(frame, a)
        B = point(frame, b)
        C = point(frame, c)
        angles.append(joint_angle(A, B, C))
    return np.array(angles, dtype=np.float32)



def compute_rep_angle_similarity(X1, X2, dtw_dict, rep_start, rep_end, exercise):
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
        sims.append(angle_similarity(ang1, ang2, exercise))

    if len(sims) == 0:
        return 0.0
    final_score = np.clip(float(np.mean(sims)),0,1)
    return final_score