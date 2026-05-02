from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import product
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# KNN
def knn_sweep(X_train, X_test, y_train, y_test, k_range=range(1, 11)):
    results = []
    for k, w, m in product(k_range, ['uniform', 'distance'], ['euclidean', 'manhattan']):
        knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        results.append((k, w, m, acc))
    results.sort(key=lambda x: x[3], reverse=True)
    return results

# SVM
def svm_eval(X_train, X_test, y_train, y_test, kernels=['linear', 'rbf', 'poly']):
    results = []
    for kernel in kernels:
        svm = SVC(kernel=kernel)
        svm.fit(X_train, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test))
        results.append((kernel, acc))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 1D-CNN Model
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # (batch, 64, 1) → (batch, 64)
        x = self.classifier(x)
        return x
def cnn_eval(X_train, X_test, y_train, y_test, epochs=200, lr=1e-2):
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
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

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=32, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)    # h_n: (num_layers, batch, hidden)
        out = h_n[-1]                  # take last layer's hidden state
        return self.classifier(out)
    
def lstm_eval(X_train, X_test, y_train, y_test, epochs=200, lr=1e-3):
    # Reshape to (batch, seq_len, 1) instead of unsqueeze(1) for CNN
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
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

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm