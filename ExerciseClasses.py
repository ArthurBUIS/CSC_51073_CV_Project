from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ExerciseConfig:
    def __init__(
        self, 
        name: str,
        axis: int, #0=x, 1=y, 2=z
        opti: int, #1 = max, -1 = min
        landmark_id: int,
        sensibility: int,
        angle_weights: Optional[np.ndarray] = None, 
        a : float = 1.0,
        b : float = 1.0,
        c : float = 0.0,
    ):
        self.name = name
        self.axis = axis
        self.opti = opti
        self.landmark_id = landmark_id
        self.sensibility = sensibility
        if angle_weights is None:
            self.angle_weights = np.ones(8, dtype=np.float32)
        else:
            self.angle_weights = angle_weights
        self.a = a  # default scaling factor
        self.b = b  # default scaling factor
        self.c = c  # default offset


# ANGLE_WEIGHTS = np.array([ #For push-ups
#     # Bras
#     3.0,  # bras gauche
#     3.0,  # bras droit
#     1.0,  # jambe gauche
#     1.0,  # jambe droite
#     2.0,  # tronc gauche
#     2.0,  # tronc droit
#     2.5,  # épaules gauche
#     2.5,  # épaules droite
# ], dtype=np.float32)

# Exemple de dictionnaire d’exercices
EXERCISES = {
    "barbell biceps curl": ExerciseConfig(name="barbell biceps curl", axis=1, opti=-1, landmark_id=15, sensibility=0.3, angle_weights=np.array([
    3.5,   # bras gauche
    3.5,   # bras droit
    0.1,   # jambe gauche
    0.1,   # jambe droite
    1.5,   # tronc gauche
    1.5,   # tronc droit
    0.1,   # épaules gauche
    0.1,   # épaules droite
], dtype=np.float32), a =2.5, b=500, c = 0.035),
    
    "leg extension": ExerciseConfig(name="leg extension", axis=2, opti=-1, landmark_id=29, sensibility=0.35,angle_weights=np.array([
    0.5,  # bras gauche
    0.5,  # bras droit
    3.5,  # jambe gauche
    3.5,  # jambe droite
    2.0,  # tronc gauche
    2.0,  # tronc droit
    0.75,  # épaules gauche
    0.75,  # épaules droite
], dtype=np.float32), a =1.2, b=25, c = 0.37,),
    
    "push-up": ExerciseConfig(name="push-up", axis=2, opti=-1, landmark_id=15, sensibility=0.35, angle_weights=np.array([
    3.0,  # bras gauche
    3.0,  # bras droit
    1.0,  # jambe gauche
    1.0,  # jambe droite
    2.0,  # tronc gauche
    2.0,  # tronc droit
    2.5,  # épaules gauche
    2.5,  # épaules droite
], dtype=np.float32), a =1.15, b=100, c = 0.08), #-> ref = video 11, commence en bas

    
    "squat": ExerciseConfig(name="squat", axis=1, opti=1, landmark_id=30, sensibility=0.3,angle_weights=np.array([
    0.1,  # bras gauche
    0.1,  # bras droit
    3.5,  # jambe gauche
    3.5,  # jambe droite
    2.0,  # tronc gauche
    2.0,  # tronc droit
    0.2,  # épaules gauche
    0.2,  # épaules droite
], dtype=np.float32), a =2.2, b=200, c = 0.072,),
}