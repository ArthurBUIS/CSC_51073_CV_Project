import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DataClasses import FrameData, VideoData, Dataset, Exercise


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    Regularizer enforcing orthogonality in transformation matrices (T-Net).
    """

    def __init__(self, num_features: int, l2reg: float = 0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        """
        Computes the orthogonal regularization term.
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.matmul(x, x, transpose_b=True)
        diff = xxt - self.eye
        return tf.reduce_sum(self.l2reg * tf.square(diff))


class ClassifierModel:
    """
    Wrapper class to construct and manage a classification model
    for pose-based or point-based data.
    """

    def __init__(self, n_classes: int = 4, batch_size: int = 128):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.model = None

    # ------------------------------
    # Utility building blocks
    # ------------------------------
    @staticmethod
    def conv_bn(x, filters: int):
        """
        1D convolution followed by batch normalization and ReLU.
        """
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    @staticmethod
    def dense_bn(x, filters: int):
        """
        Dense layer followed by batch normalization and ReLU.
        """
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def tnet(self, inputs, num_features: int):
        """
        Builds a transformation network (T-Net) block.
        """
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = self.conv_bn(inputs, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = self.dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)

        feat_T = layers.Reshape((num_features, num_features))(x)
        transformed = layers.Dot(axes=(2, 1))([inputs, feat_T])
        return transformed

    def build_model(self):
        """
        Constructs the full classification model.
        """
        inputs = keras.Input(shape=(33, 3)) # Format des inputs (tracking des vidéos avec 33 points 3D)
    
        x = self.tnet(inputs, 3)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 32)
        x = self.tnet(x, 32)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = self.dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        self.model.summary()
        return self.model
    
    def train_model(self, train_dataset, test_dataset, epochs: int = 10, lr: float = 0.001):
        """
        Train the model with the given datasets.
        
        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Training dataset
        test_dataset : tf.data.Dataset
            Testing dataset
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        """
        # Compilation du modèle
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks pour améliorer l'entraînement
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Entraînement
        print("🚀 Début de l'entraînement...")
        history = self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation finale
        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=0)
        print(f"\n📊 Résultats finaux:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return history
        
    def visualize_predictions(model, test_dataset, class_names):
        data = test_dataset.take(3)

        points, labels = list(data)[1]
        points = points[:8,]
        labels = labels[:8,]
        labels = pd.DataFrame(labels.numpy()).idxmax(axis=1)

        # run test data through model
        preds = model.predict(points)
        preds = tf.math.argmax(preds, -1) # retrieve class with highest probability
        points = points.numpy()

        # plot points with predicted class and label
        fig = plt.figure(figsize=(15,10))
        for i in range(8):
            ax = fig.add_subplot(2,4,i+1, projection = "3d")
            ax.scatter(points[i,:,0], points[i,:,1],points[i,:,2])
            ax.set_title(f"pred: {class_names[preds[i].numpy()]}, label: {class_names[labels[i]]}")
        plt.show()
        
    def predict_frame(self, data: FrameData) -> (int, float):
        """
        Predict the class of a single frame and store results in the FrameData object.
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        
        # Vérification des landmarks
        if data.landmarks is None:
            raise ValueError("No landmarks found in FrameData")
        
        # Nettoyage des données (comme dans build_tf_dataset)
        landmarks_clean = np.nan_to_num(data.landmarks, nan=0.0, posinf=0.0, neginf=0.0)
        input_data = np.expand_dims(landmarks_clean.astype(np.float32), axis=0)  # Ajouter dimension batch
        
        # Prédiction
        predictions = self.model.predict(input_data, verbose=0)  # verbose=0 pour éviter les logs
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]
        
        # Convertir l'index de classe en Exercise enum
        exercises_list = list(Exercise)
        if predicted_class_idx < len(exercises_list):
            data.predicted_class = exercises_list[predicted_class_idx]
        else:
            data.predicted_class = None
        
        data.confidence = float(confidence)
        data.scores = predictions[0]  # Stocker toutes les probabilités
        
        return predicted_class_idx, confidence
    
def normalize_label(label: str) -> str:
    res = ""
    for l in label:
        if l.isalnum():
            res += l.lower()
        elif l == " ":
            res += "_"
    return res

if __name__ == "__main__":

    clf = ClassifierModel(n_classes=3)
    model = clf.build_model()
    model.summary()

    # === 2️⃣ Préparation du dataset ===
    from DataClasses import VideoData, Dataset, Exercise, FrameData
    from pose_detection import extract_pose_from_video_interpolated
    from DataClasses import build_tf_dataset 

    # Conversion en dataset de frames
    dataset = Dataset()
    
    from tqdm import tqdm

    dataset_raw = pd.read_csv("data/full_landmarks_dataset.csv")  # Pour charger le dataset complet
    num_classes = 22

    # Ajout d'une barre de chargement avec tqdm
    for index, data in tqdm(dataset_raw.iterrows(), total=len(dataset_raw), desc="Chargement du dataset"):
        frame_data = FrameData(
            filename=data["video_name"], 
            frame_index=data["frame_number"],
            landmarks=np.array(data.drop(["video_name","total_frames","frame_number","width","height","label"])).reshape(-1, 3),
            predicted_class=None,
            confidence=None,
            scores=None,
            ground_truth=Exercise(normalize_label(data["label"]))
        )
        dataset.add_data(frame_data)

    print(f"Dataset chargé avec succès ! Total: {len(dataset.datas)} frames")

    # Split train / test
    train_dataset, test_dataset = dataset.split(train_ratio=0.8)

    # === 3️⃣ Construction des datasets TensorFlow ===
    train_dataset = build_tf_dataset(train_dataset, num_classes=num_classes, batch_size=32)
    test_dataset = build_tf_dataset(test_dataset, num_classes=num_classes, batch_size=32)

    # === 4️⃣ Entraînement ===
    clf.train_model(train_dataset, test_dataset, epochs=10, lr=0.001)