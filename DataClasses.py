from enum import Enum
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from pose_detection import extract_pose_from_video_interpolated
import tensorflow as tf


class Exercise(Enum):
    BARBELL_BICEPS_CURL = "barbell_biceps_curl"
    BENCH_PRESS = "bench_press"
    CHEST_FLY_MACHINE = "chest_fly_machine"
    DEADLIFT = "deadlift"
    DECLINE_BENCH_PRESS = "decline_bench_press"
    HAMMER_CURL = "hammer_curl"
    HIP_THRUST = "hip_thrust"
    INCLINE_BENCH_PRESS = "incline_bench_press"
    LAT_PULLDOWN = "lat_pulldown"
    LATERAL_RAISE = "lateral_raise"
    LEG_EXTENSION = "leg_extension"
    LEG_RAISES = "leg_raises"
    PLANK = "plank"
    PULL_UP = "pull_up"
    PUSH_UP = "push_up"
    ROMANIAN_DEADLIFT = "romanian_deadlift"
    RUSSIAN_TWIST = "russian_twist"
    SHOULDER_PRESS = "shoulder_press"
    SQUAT = "squat"
    T_BAR_ROW = "t_bar_row"
    TRICEP_DIPS = "tricep_dips"
    TRICEP_PUSHDOWN = "tricep_pushdown"

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
        X: np.ndarray of shape (N, 33, 3)
        y: np.ndarray of class indices
        Only includes frames with both landmarks and labels.
        """
        X, y = [], []
        for frame_data in self.datas:
            if frame_data.landmarks is None or frame_data.ground_truth is None:
                continue
                
            # Vérifier que landmarks a la bonne forme et ne contient pas de NaN
            landmarks = frame_data.landmarks
            if landmarks.shape != (33, 3):
                print(f"Warning: Skipping frame with invalid landmarks shape {landmarks.shape}")
                continue
            
                
            X.append(landmarks.astype(np.float32))
            
            # Obtenir l'index de la classe
            try:
                class_idx = list(Exercise).index(frame_data.ground_truth)
                y.append(class_idx)
            except ValueError:
                print(f"Warning: Unknown exercise type {frame_data.ground_truth}")
                continue
                
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def split(self, train_ratio: float = 0.8):
        """
        Split the dataset into train and test sets.
        
        Parameters
        ----------
        train_ratio : float
            Ratio of data to use for training (default: 0.8)
        
        Returns
        -------
        tuple[Dataset, Dataset]
            (train_dataset, test_dataset)
        """
        from sklearn.model_selection import train_test_split
        
        # Grouper par vidéo pour éviter le data leakage
        videos = {}
        for frame_data in self.datas:
            video_name = frame_data.filename
            if video_name not in videos:
                videos[video_name] = []
            videos[video_name].append(frame_data)
        
        # Split par vidéo
        video_names = list(videos.keys())
        train_videos, test_videos = train_test_split(
            video_names, 
            train_size=train_ratio, 
            random_state=42,
            stratify=None  
        )
        
        # Créer les datasets
        train_dataset = Dataset()
        test_dataset = Dataset()
        
        for video_name in train_videos:
            for frame_data in videos[video_name]:
                train_dataset.add_data(frame_data)
        
        for video_name in test_videos:
            for frame_data in videos[video_name]:
                test_dataset.add_data(frame_data)
        
        print(f"Split completed:")
        print(f"  Training: {len(train_dataset.datas)} frames from {len(train_videos)} videos")
        print(f"  Testing: {len(test_dataset.datas)} frames from {len(test_videos)} videos")
        
        return train_dataset, test_dataset


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

    # CORRECTIONS : Conversion explicite des types et gestion des valeurs manquantes
    
    # 1. Nettoyer et convertir X
    X = np.array(X, dtype=np.float32)  # Force le type float32
    
    # Remplacer les NaN et valeurs infinies par 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Nettoyer et convertir y
    y = np.array(y, dtype=np.int32)  # Force le type int32
    
    # Vérifier que les indices de classes sont valides
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError(f"Invalid class indices found. Expected [0, {num_classes-1}], got [{y.min()}, {y.max()}]")

    print(f"Dataset shapes: X={X.shape}, y={y.shape}")
    print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
    
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