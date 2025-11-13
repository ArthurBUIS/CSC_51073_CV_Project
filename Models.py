import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        model.summary()
        return model
    
    def train_model(self, train_dataset, test_dataset, epochs: int = 20, lr: float = 0.001):
        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
        history = self.model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
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
    

if __name__ == "__main__":
    clf = ClassifierModel(n_classes=3)
    model = clf.build_model()
    model.summary()
    train_dataset = ...
    test_dataset = ...
    clf.train_model(train_dataset, test_dataset, epochs=10, lr=0.001)