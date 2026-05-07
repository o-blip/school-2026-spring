import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix



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


def lstm_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    num_classes: int = 3,
) -> LSTMModel:
    """Train LSTMModel on all provided data and return the fitted model.

    Parameters
    ----------
    X_train : (N, time_frames, 13) MFCC sequence arrays
    y_train : (N,) integer label arrays (0, 1, or 2)

    Returns
    -------
    model : trained LSTMModel in eval mode
    loss_history : list of training loss values per epoch for visualization
    """
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    torch.manual_seed(42)
    model = LSTMModel(num_classes=num_classes)
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
            scheduler.step()
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        if early_stop_check(loss_history):
            print(f"Early stopping at epoch {epoch+1} with avg loss={np.mean(loss_history[-20:]):.4f}")
            break
    model.eval()
    return model, loss_history


def lstm_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    num_classes: int = 3,
) -> tuple[float, np.ndarray]:
    """Train LSTMModel and return test accuracy and confusion matrix."""
    model, loss_history = lstm_train(X_train, y_train, epochs, lr, num_classes)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
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

