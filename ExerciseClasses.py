from dataclasses import dataclass

@dataclass
class ExerciseConfig:
    name: str
    axis: int          # 0=x, 1=y, 2=z
    opti: int          # -1 -> détecter min, +1 -> détecter max
    landmark_id: int   # index du landmark à suivre
    sensibility: int

# Exemple de dictionnaire d’exercices
EXERCISES = {
    "barbell biceps curl": ExerciseConfig(name="barbell biceps curl", axis=1, opti=1, landmark_id=15, sensibility=0.3),
    "leg extension": ExerciseConfig(name="leg extension", axis=2, opti=1, landmark_id=30, sensibility=0.3),
    "push-up": ExerciseConfig(name="push-up", axis=0, opti=1, landmark_id=15, sensibility=0.35),
    "bench press": ExerciseConfig(name="bench press", axis=0, opti=1, landmark_id=15, sensibility=0.4), #-> un peu broken, marche pas hyper bien niveau synchro
}