from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


class Exercise(Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    BENCH_PRESS = "bench_press"


@dataclass
class PoseSample:
    """
    Unified data structure for pose ML pipeline.

    filename : path to the associated video
    frames : np.ndarray (T, 33, 3) pose landmarks or None if not yet processed
    predicted_class : predicted exercise or None
    confidence : confidence score of prediction
    scores : full probability distribution over classes or None
    ground_truth : actual label for training or None
    fps : frames per second
    """

    filename: str
    frames: Optional[np.ndarray] = None
    predicted_class: Optional[Exercise] = None
    confidence: Optional[float] = None
    scores: Optional[np.ndarray] = None
    ground_truth: Optional[Exercise] = None
    fps: Optional[int] = None

    def num_frames(self) -> int:
        if self.frames is None:
            return 0
        return self.frames.shape[0]

    def normalize(self) -> None:
        """
        Center skeleton on hips + normalize by torso length.
        Applies directly to self.frames.
        """
        if self.frames is None:
            return

        hip_center = self.frames[:, [23, 24], :].mean(axis=1)  # left & right hips
        self.frames = self.frames - hip_center[:, None, :]

        # Normalize by shoulder-to-hip distance as scale reference
        shoulders = self.frames[:, [11, 12], :].mean(axis=1)
        torso_len = np.linalg.norm(shoulders - hip_center, axis=1)
        torso_len[torso_len == 0] = 1
        self.frames = self.frames / torso_len[:, None, None]
