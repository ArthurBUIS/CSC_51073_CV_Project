from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ExerciseConfig:
    """
    Per-exercise configuration used by grader.py (repetition detection) and
    metric.py (scoring).

    axis, opti and landmark_id define which landmark/axis is tracked to
    detect repetitions (see grader.detect_repetitions_ex): the signal is
    landmark_id's coordinate on `axis`, and `opti` says whether a rep peaks
    at a minimum (-1) or a maximum (+1) of that signal.

    angle_weights, a, b and c calibrate metric.angle_similarity's scoring
    curve for this exercise.
    """

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


# Exercise configuration dictionary, keyed by exercise name
EXERCISES = {
    "barbell biceps curl": ExerciseConfig(name="barbell biceps curl", axis=1, opti=-1, landmark_id=15, sensibility=0.3, angle_weights=np.array([
    3.5,   # left arm
    3.5,   # right arm
    0.1,   # left leg
    0.1,   # right leg
    1.5,   # left torso
    1.5,   # right torso
    0.1,   # left shoulder
    0.1,   # right shoulder
], dtype=np.float32), a =2.5, b=50000, c = 0.035),
    
    "leg extension": ExerciseConfig(name="leg extension", axis=2, opti=-1, landmark_id=29, sensibility=0.35,angle_weights=np.array([
    0.5,  # left arm
    0.5,  # right arm
    3.5,  # left leg
    3.5,  # right leg
    2.0,  # left torso
    2.0,  # right torso
    0.75,  # left shoulder
    0.75,  # right shoulder
], dtype=np.float32), a =1.2, b=25, c = 0.37,),
    
    "push-up": ExerciseConfig(name="push-up", axis=2, opti=-1, landmark_id=15, sensibility=0.35, angle_weights=np.array([
    3.0,  # left arm
    3.0,  # right arm
    1.0,  # left leg
    1.0,  # right leg
    2.0,  # left torso
    2.0,  # right torso
    2.5,  # left shoulder
    2.5,  # right shoulder
], dtype=np.float32), a =1.15, b=100, c = 0.08), #-> ref = video 11, starts at the bottom
    
    # "bench press": ExerciseConfig(name="bench press", axis=2, opti=-1, landmark_id=15, sensibility=0.4), #-> a bit broken, sync does not work great
    
    "squat": ExerciseConfig(name="squat", axis=1, opti=1, landmark_id=30, sensibility=0.3,angle_weights=np.array([
    0.1,  # left arm
    0.1,  # right arm
    3.5,  # left leg
    3.5,  # right leg
    2.0,  # left torso
    2.0,  # right torso
    0.2,  # left shoulder
    0.2,  # right shoulder
], dtype=np.float32), a =2.2, b=200, c = 0.072,),
}