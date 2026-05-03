from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import product
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Sequence
import numpy as np

# =============================================================================
# Models: KNN, SVM, 1D-CNN, LSTM
# All classifiers predict bolt tightening condition: 0=0ftlb, 1=25ftlb, 2=50ftlb
# =============================================================================


# --- Shallow learners ---

def knn_sweep(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_range: Sequence[int] = range(1, 11),
) -> list[tuple]:
    """Grid search over k (1-10), weight scheme, and distance metric for KNN.
    Weights: 'uniform' vs 'distance'; Distance metrics: 'euclidean' vs 'manhattan'.
    
    Returns
    -------
    results : list of (k, weights, metric, accuracy) sorted by accuracy descending
    """
    results = []
    for k, w, m in product(k_range, ['uniform', 'distance'], ['euclidean', 'manhattan']):
        knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        results.append((k, w, m, acc))
    results.sort(key=lambda x: x[3], reverse=True)
    return results


def svm_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernels: list[str] = ['linear', 'rbf', 'poly'],
) -> list[tuple]:
    """Evaluate SVM with multiple kernel functions.
    linear: standard linear SVM 
    rbf: radial basis function kernel 
    poly: polynomial kernel (degree=3 by default).
    
    Returns
    -------
    results : list of (kernel, accuracy) sorted by accuracy descending
    """
    results = []
    for kernel in kernels:
        svm = SVC(kernel=kernel)
        svm.fit(X_train, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test))
        results.append((kernel, acc))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# --- Deep learners ---

class CNN1D(nn.Module):
    """1D Convolutional Neural Network for raw waveform classification.

    Input shape: (batch, 1, splice_len) — single-channel waveform.
    Three conv blocks progressively downsample the time axis; AdaptiveAvgPool1d
    at the end collapses whatever remains to a fixed 64-D feature vector,
    so the model works regardless of the exact input length.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Block 1: broad receptive field to capture low-frequency tap onset
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 2: intermediate features
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 3: high-level features; global average pool collapses time → 1
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),  # 3 tightening classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # (batch, 64, 1) → (batch, 64)
        return self.classifier(x)


def cnn_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-2,
) -> tuple[float, np.ndarray]:
    """Train CNN1D and return test accuracy and confusion matrix.

    Learning rate is halved every 50 epochs (StepLR scheduler with step_size=50, gamma=0.5). 

    Parameters
    ----------
    X_train / X_test : (N, splice_len) z-score normalized waveform arrays
    y_train / y_test : (N,) integer label arrays (0, 1, or 2)

    Returns
    -------
    accuracy       : float
    confusion_matrix : (3, 3) int array
    """
    # Add channel dimension: (N, splice_len) → (N, 1, splice_len) (single-channel waveform)
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    torch.manual_seed(42)
    model = CNN1D()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).argmax(dim=1).numpy()

    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)


class LSTMModel(nn.Module):
    """2-layer LSTM for sequential MFCC input classification.

    Input shape: (batch, time_frames, n_mfcc=13) — batch_first convention.
    Only the final layer's last hidden state is fed to the classifier;
    the full output sequence is discarded.
    """

    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,  # applied between stacked LSTM layers, not after the last
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_size)
        out = h_n[-1]               # last layer's final hidden state, feeds into classifier
        return self.classifier(out)


def lstm_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """Train LSTMModel and return test accuracy and confusion matrix.

    Parameters
    ----------
    X_train / X_test : (N, time_frames, 13) MFCC sequence arrays
    y_train / y_test : (N,) integer label arrays (0, 1, or 2)

    Returns
    -------
    accuracy         : float
    confusion_matrix : (3, 3) int array
    """
    # LSTM expects (batch, seq_len, features) — no extra channel dim needed (unlike CNN)
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

    torch.manual_seed(42)
    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).argmax(dim=1).numpy()

    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)