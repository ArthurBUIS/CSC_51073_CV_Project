import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import joblib

# Paramètres

SEQ_LEN = 10
N_COMPONENTS = 3
DEVICE = "cpu"

MODEL_PATH = "models/cnn_model.pth"
PCA_PATH = "models/pca.pkl"

LABEL_MAP = {
    0: "barbell biceps curl",
    1: "leg extension",
    2: "push-up",
    3: "squat"
}

# CNN 1D

class CNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 2, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Chargement du modèle et de la PCA

def load_model_and_pca():
    model = CNN1D(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    pca = joblib.load(PCA_PATH)

    return model, pca

# Echantillonnage des séquences

def build_sequence(features):
    """Retourne une séquence de shape (SEQ_LEN, 3) pour le CNN 1D."""

    if len(features) >= SEQ_LEN:
        indices = np.linspace(0, len(features) - 1, SEQ_LEN).astype(int)
        X = features[indices]
    else:
        pad = np.zeros((SEQ_LEN - len(features), features.shape[1]), dtype=features.dtype)
        X = np.vstack([features, pad])

    return X.astype(np.float32)
        

# Prétraitement dataframe

def preprocess_df(df, pca):
    """
    df : dataframe contenant uniquement les colonnes numériques des landmarks
    """
    data = df.select_dtypes(include=[np.number]).to_numpy()
    data_pca = pca.transform(data)

    sequence = build_sequence(data_pca)

    X = torch.tensor(sequence, dtype=torch.float32)
    X = X.unsqueeze(0)  # (1, SEQ_LEN, 3)
    X = X.permute(0, 2, 1)  # (1, 3, SEQ_LEN)
    X = X.to(DEVICE)

    return X

# Prédiction finale

def predict_exercise(df):
    model, pca = load_model_and_pca()
    df = df.drop(['video_name','total_frames','frame_number','width','height','label'], axis=1, errors="ignore")
    X = preprocess_df(df, pca)

    with torch.no_grad():
        outputs = model(X)
        pred_id = int(torch.argmax(outputs, dim=1).cpu().item())

    return LABEL_MAP[pred_id]
