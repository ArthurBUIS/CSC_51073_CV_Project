import numpy as np
from ExerciseClasses import EXERCISES


ANGLE_TRIPLETS = [
    # Arms
    (11, 13, 15),  # L shoulder - L elbow - L wrist
    (12, 14, 16),  # R shoulder - R elbow - R wrist

    # Legs
    (23, 25, 27),  # L hip - L knee - L ankle
    (24, 26, 28),  # R hip - R knee - R ankle

    # Torso orientation
    (11, 23, 25),  # L shoulder - L hip - L knee
    (12, 24, 26),  # R shoulder - R hip - R knee

    # Shoulders
    (13, 11, 12),  # L elbow - L shoulder - R shoulder
    (14, 12, 11),  # R elbow - R shoulder - L shoulder
]

landmark_weights = np.array([
    # 0-10: head and upper torso -> not very useful for push-ups
    0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
    # 11-16: shoulders, elbows, wrists -> main pivot of the movement
    1.0,  # 11 L shoulder
    1.0,  # 12 R shoulder
    1.0,  # 13 L elbow
    1.0,  # 14 R elbow
    0.9,  # 15 L wrist
    0.9,  # 16 R wrist
    # 17-22: lower torso -> useful
    0.5,0.5,0.5,0.5,0.5,0.5,
    # 23-28: hips, knees, ankles -> important for posture
    0.8,  # L hip
    0.8,  # R hip
    0.6,0.6,0.4,0.4,  # knees / ankles
    # 29-32: feet -> not very useful
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

        # Compare the full set of landmarks (33×3 = 99 dims)
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

    # Numerical clamp
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return angle


def point(frame, i):
    return frame[3*i : 3*i + 3]  

def angle_similarity(a1, a2, exercise, strictness=2, weight_scale=1):
    diff = a1 - a2
    weighted = (exercise.angle_weights / weight_scale) * (np.abs(diff) ** strictness)
    score = np.exp(-np.sum(weighted)) #For discrimination
    
    score = (score - exercise.c)*exercise.b  # Center around 0
    score = exercise.a * 1 / (1 + np.exp(-score))
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

        # Convert frames to angle vectors
        ang1 = frame_to_angle_vector(X1[i])
        ang2 = frame_to_angle_vector(X2[j])
        # Cosine similarity on the angles
        sims.append(angle_similarity(ang1, ang2, exercise))

    if len(sims) == 0:
        return 0.0
    final_score = np.clip(float(np.mean(sims)),0,1)
    return final_score