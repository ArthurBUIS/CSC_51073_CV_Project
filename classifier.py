from abc import ABC, abstractmethod
from random import sample
from typing import List
import numpy as np
from pose_data_classes import PoseSample, Exercise
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ExerciseClassifierBase(ABC):
    """
    Base abstract class for exercise classification models.
    Defines the API for training, prediction, saving and loading.
    """

    @abstractmethod
    def train(self, dataset: List[PoseSample]) -> None:
        """
        Train the model using a list of labeled PoseSample objects.
        Each PoseSample must have ground_truth not None.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, sample: PoseSample) -> PoseSample:
        """
        Predict the exercise and confidence score for a PoseSample.
        Must update sample.predicted_class, sample.confidence, sample.scores.
        Return the updated PoseSample.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model weights/config."""
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load model weights/config."""
        raise NotImplementedError

    def _apply_prediction(self, sample: PoseSample, probabilities: np.ndarray) -> PoseSample:
        """
        Helper to convert model probability vector to predicted class + confidence.
        """
        sample.scores = probabilities
        best_idx = int(np.argmax(probabilities))
        sample.predicted_class = list(Exercise)[best_idx]
        sample.confidence = float(probabilities[best_idx])
        return sample




class RandomForestExerciseClassifier(ExerciseClassifierBase):
    """
        Baseline classifier using a RandomForest model.
        Feature = mean over time of normalized landmarks.
    """


    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)
        self.scaler = StandardScaler()


    def _extract_features(self, sample: PoseSample) -> np.ndarray:
        if sample.frames is None:
            raise ValueError("frames required for feature extraction")
        sample.normalize() # ensure normalized skeleton
        flat = sample.frames.reshape(sample.num_frames(), -1) # (T, 99)
        features = flat.mean(axis=0) # time pooling
        return features


    def train(self, dataset: List[PoseSample]) -> None:
        X, y = [], []
        for s in dataset:
            if s.ground_truth is None:
                continue
        X.append(self._extract_features(s))
        y.append(list(Exercise).index(s.ground_truth))


        X = np.array(X)
        y = np.array(y)


        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)


    def predict(self, sample: PoseSample) -> PoseSample:
        features = self._extract_features(sample)
        X_scaled = self.scaler.transform([features])
        probs = self.model.predict_proba(X_scaled)[0]
        return self._apply_prediction(sample, probs)


    def save(self, filepath: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler}, filepath)


    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]