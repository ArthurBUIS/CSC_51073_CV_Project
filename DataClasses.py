from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from pose_detection import extract_pose_from_video_interpolated
import tensorflow as tf


class Exercise(Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    BENCH_PRESS = "bench_press"
    # Rajouter les autres plus tard

@dataclass
class FrameData:
    """
    Data structure for a single video frame.

    filename : path to the associated video
    frame_index : index of the frame in the video
    landmarks : np.ndarray (33, 3) pose landmarks or None if not detected
    predicted_class : predicted exercise or None
    confidence : confidence score of prediction
    scores : full probability distribution over classes or None
    ground_truth : for training data, the actual label, or None for test data
    """
    
    filename: str
    frame_index: int
    landmarks: Optional[np.ndarray] = None
    predicted_class: Optional[Exercise] = None
    confidence: Optional[float] = None
    scores: Optional[np.ndarray] = None
    ground_truth: Optional[Exercise] = None


@dataclass
class VideoData:
    """
    Data structure for an entire video.

    filename : path to the associated video
    frames : np.ndarray (T, 33, 3) pose landmarks or None if not yet processed
    predicted_class : predicted exercise or None
    confidence : confidence score of prediction
    scores : full probability distribution over classes or None
    ground_truth : actual label for training or None
    fps : frames per second
    """

    filename: str
    frames: Optional[List[FrameData]] = None
    landmarks: Optional[np.ndarray] = None
    predicted_class: Optional[Exercise] = None
    confidence: Optional[float] = None
    scores: Optional[np.ndarray] = None
    ground_truth: Optional[Exercise] = None
    fps: Optional[int] = None

    def num_frames(self) -> int:
        """Return number of frames in the video."""
        if self.frames is None:
            return 0
        return len(self.frames)
    
    def normalize(self) -> None:
        """
        Center skeleton on hips + normalize by torso length.
        Applies directly to self.frames.
        """
        if self.frames is None:
            return
        
        array_frames = np.array([frame.landmarks for frame in self.frames])  # (T, 33, 3)

        hip_center = array_frames[:, [23, 24], :].mean(axis=1)  # left & right hips
        array_frames = array_frames - hip_center[:, None, :]

        # Normalize by shoulder-to-hip distance as scale reference
        shoulders = array_frames[:, [11, 12], :].mean(axis=1)
        torso_len = np.linalg.norm(shoulders - hip_center, axis=1)
        torso_len[torso_len == 0] = 1
        array_frames = array_frames / torso_len[:, None, None]
    
        for i, frame in enumerate(self.frames):
            frame.landmarks = array_frames[i]
            
    def landmark_estimation(self) -> None:
        """
        Run pose estimation on the video to fill self.frames and self.landmarks.
        Uses interpolation for missing landmarks.
        """
        if self.filename is None:
            return
        
        landmarks = extract_pose_from_video_interpolated(self.filename)
        if landmarks is None:
            return
        
        self.landmarks = landmarks  # (T, 33, 3)
        self.frames = []
        for i in range(landmarks.shape[0]):
            frame_data = FrameData(
                filename=self.filename,
                frame_index=i,
                landmarks=landmarks[i]
            )
            self.frames.append(frame_data)

        
@dataclass
class Dataset:
    """
    Container for a collection of FrameData objects.
    Can represent a training, validation or test dataset.
    """

    datas: Optional[List[FrameData]] = None

    def __post_init__(self):
        if self.datas is None:
            self.datas = []
            
    def add_data(self, data: FrameData) -> None:
        """Add a single FrameData object to the dataset."""
        self.datas.append(data)
        
    def add_datas(self, datas: List[FrameData]) -> None:
        """Add multiple FrameData objects to the dataset."""
        self.datas.extend(datas)
        
    def add_video_data(self, video_data: VideoData) -> None:
        """Convert VideoData to FrameData objects and add to dataset."""
        if video_data.frames is None:
            return
        
        for frame_data in video_data.frames:
            frame_data.predicted_class = video_data.predicted_class
            frame_data.confidence = video_data.confidence
            frame_data.scores = video_data.scores
            frame_data.ground_truth = video_data.ground_truth
            self.add_data(frame_data)

    def __len__(self) -> int:
        """Return number of frames in the dataset."""
        return len(self.datas)
    
    def __getitem__(self, idx: int) -> FrameData:
        """Allow index-based access (dataset[i])."""
        return self.datas[idx]

    def get_data_arrays(self):
        """
        Returns tuple (X, y) for ML pipelines:
        X: np.ndarray of shape (N, T, 33, 3)
        y: np.ndarray of class indices
        Only includes frames with both landmarks and labels.
        """
        X, y = [], []
        for frame_data in self.datas:
            if frame_data.landmarks is None or frame_data.ground_truth is None:
                continue
            X.append(frame_data.landmarks)
            y.append(list(Exercise).index(frame_data.ground_truth))
        return np.array(X), np.array(y)

    def split(self, train_ratio: float = 0.8, shuffle: bool = True, seed: Optional[int] = None):
        """
        Split dataset into train/test subsets.
        Returns (train_dataset, test_dataset)
        """
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.datas)

        split_idx = int(len(self.datas) * train_ratio)
        train_datas = self.datas[:split_idx]
        test_datas = self.datas[split_idx:]
        return Dataset(train_datas), Dataset(test_datas)






def build_tf_dataset(dataset: Dataset, num_classes: int, batch_size: int = 32) -> tf.data.Dataset:
    """
    Convert a Dataset (collection of FrameData) into a TensorFlow-compatible dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing FrameData elements (each must have landmarks and ground_truth).
    num_classes : int
        Number of exercise classes (for one-hot encoding).
    batch_size : int, optional
        Batch size for the tf.data.Dataset.

    Returns
    -------
    tf.data.Dataset
        A batched and shuffled TensorFlow dataset ready for model training.
        Each element is a tuple (X, y) with shapes:
            X -> (33, 3)
            y -> (num_classes,)
    """

    # Récupération des données sous forme de tableaux numpy
    X, y = dataset.get_data_arrays()

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Dataset is empty or missing labels/landmarks.")

    # Encodage one-hot des labels
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # Conversion en tf.data.Dataset
    tf_ds = tf.data.Dataset.from_tensor_slices((X, y))

    # Optimisations classiques pour l'entraînement
    tf_ds = (
        tf_ds
        .shuffle(buffer_size=len(X))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return tf_ds