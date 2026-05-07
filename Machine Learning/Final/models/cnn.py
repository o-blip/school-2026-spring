import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

class CNN1D(nn.Module):
    """1D Convolutional Neural Network for raw waveform classification.

    Input shape: (batch, 1, splice_len) — single-channel waveform.
    Three conv blocks progressively downsample the time axis; AdaptiveAvgPool1d
    at the end collapses whatever remains to a fixed 64-D feature vector,
    so the model works regardless of the exact input length.
    """

    def __init__(self, num_classes: int = 3) -> None:
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
            nn.Linear(32, num_classes),  # 2 or 3 classes optional depending on how you set up the labels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # (batch, 64, 1) → (batch, 64)
        return self.classifier(x)




def cnn_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-2,
    num_classes: int = 3,
) -> CNN1D:
    """Train CNN1D on all provided data and return the fitted model.

    Parameters
    ----------
    X_train : (N, splice_len) z-score normalized waveform arrays
    y_train : (N,) integer label arrays (0, 1, or 2)

    Returns
    -------
    model : trained CNN1D in eval mode
    loss_history : list of training loss values per epoch for visualization
    """
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    torch.manual_seed(42)
    model = CNN1D(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        scheduler.step()
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        if early_stop_check(loss_history):
            print(f"Early stopping at epoch {epoch+1} with avg loss={np.mean(loss_history[-20:]):.4f}")
            break
    model.eval()
    return model, loss_history


def cnn_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-2,
    num_classes: int = 3,
) -> tuple[float, np.ndarray]:
    """Train CNN1D and return test accuracy and confusion matrix."""
    model, loss_history = cnn_train(X_train, y_train, epochs, lr, num_classes)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(X_te_t).argmax(dim=1).numpy()
    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), loss_history


def early_stop_check(loss_history, patience=20, min_delta=1e-4, min_epochs=80, loss_threshold=0.05):
    if len(loss_history) < max(patience * 2, min_epochs):
        return False
    recent_avg = np.mean(loss_history[-patience:])
    if recent_avg > loss_threshold:
        return False
    previous_avg = np.mean(loss_history[-2*patience:-patience])
    return previous_avg - recent_avg < min_delta

