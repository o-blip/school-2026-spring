# %%
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import periodogram
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import load_hw2_test_set
from data_loading import load_training_data
from features import (
    add_bands,
    add_bias,
    extract_psd_bands,
    extract_mfcc_flat,
    extract_logmel,
    extract_mfcc_seq,
)

# HW4: BPNN, CNN, and RNN Models to Detect Debonding (Unhealthy) Cells
# Piero Risi Mortola
# Claude used in writing the code

# %% =============================================================================
# CONSTANTS
# =============================================================================
SR = 48000
NOISE_SAMPLE_DURATION = 0.5  # seconds — initial silence used as noise profile
EXPECTED_HITS = {"g": 10, "b": 24}  # expected taps per label

ML_RECORDINGS_DIR = "ML recordings"
WAV_DIR = "wav"
IMAGES_DIR = "Images"
CACHE_DIR = "cache"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


#  =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================

# %% ── 1.1 + 1.2  LOAD, DENOISE, SPLICE, NORMALIZE (with caching) ───────────────

TRAIN_CACHE = os.path.join(CACHE_DIR, "train_splices.npz")
HW2_CACHE = os.path.join(CACHE_DIR, "hw2_splices.npz")

healthy_hits, unhealthy_hits, _ = load_training_data(
    ML_RECORDINGS_DIR, WAV_DIR, SR, NOISE_SAMPLE_DURATION, EXPECTED_HITS,
    TRAIN_CACHE, IMAGES_DIR,
)

# %% ── 1.3  TRAIN / VALIDATION / TEST SPLIT (70/15/15) ──────────────────────────

X_all = np.concatenate([healthy_hits, unhealthy_hits], axis=0)
y_all = np.array(
    [0] * len(healthy_hits) + [1] * len(unhealthy_hits)
)  # 0=healthy, 1=unhealthy

# First split: hold out 15% as ML test set
X_trainval, X_test_ml, y_trainval, y_test_ml = train_test_split(
    X_all, y_all, test_size=0.15, shuffle=True, random_state=42, stratify=y_all
)
# Second split: 15/85 ≈ 17.6% of remaining → final 70/15/15
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=15/85, shuffle=True, random_state=42, stratify=y_trainval
)

print(
    f"Train — {len(X_train):4d} splices  "
    f"({(y_train == 0).sum()} healthy, {(y_train == 1).sum()} unhealthy)"
)
print(
    f"Val   — {len(X_val):4d} splices  "
    f"({(y_val == 0).sum()} healthy, {(y_val == 1).sum()} unhealthy)"
)
print(
    f"Test  — {len(X_test_ml):4d} splices  "
    f"({(y_test_ml == 0).sum()} healthy, {(y_test_ml == 1).sum()} unhealthy)"
)
print("Indep — HW2 recordings (loaded below)")


# %% ── 1.4  FEATURE EXTRACTION ───────────────────────────────────────────────────
# Classical flat features (BPNN / KNN / DT / LR / SVM)

# -- PSD bands --
X_train_psd = extract_psd_bands(X_train, SR)
X_val_psd   = extract_psd_bands(X_val, SR)
X_test_ml_psd = extract_psd_bands(X_test_ml, SR)
psd_mean, psd_std = X_train_psd.mean(axis=0), X_train_psd.std(axis=0)
X_train_psd   = (X_train_psd   - psd_mean) / psd_std
X_val_psd     = (X_val_psd     - psd_mean) / psd_std
X_test_ml_psd = (X_test_ml_psd - psd_mean) / psd_std

# -- MFCC flat --
X_train_mfcc = extract_mfcc_flat(X_train, SR)
X_val_mfcc   = extract_mfcc_flat(X_val, SR)
X_test_ml_mfcc = extract_mfcc_flat(X_test_ml, SR)
mfcc_mean, mfcc_std = X_train_mfcc.mean(axis=0), X_train_mfcc.std(axis=0)
X_train_mfcc   = (X_train_mfcc   - mfcc_mean) / mfcc_std
X_val_mfcc     = (X_val_mfcc     - mfcc_mean) / mfcc_std
X_test_ml_mfcc = (X_test_ml_mfcc - mfcc_mean) / mfcc_std

# -- Combined PSD + MFCC (BPNN) --
X_train_combined   = np.concatenate([X_train_psd, X_train_mfcc], axis=1)
X_val_combined     = np.concatenate([X_val_psd, X_val_mfcc], axis=1)
X_test_ml_combined = np.concatenate([X_test_ml_psd, X_test_ml_mfcc], axis=1)

# -- Bias-augmented (logistic regression) --
X_train_psd_b,    X_val_psd_b    = add_bias(X_train_psd),  add_bias(X_val_psd)
X_train_mfcc_b,   X_val_mfcc_b   = add_bias(X_train_mfcc), add_bias(X_val_mfcc)
X_test_ml_psd_b  = add_bias(X_test_ml_psd)
X_test_ml_mfcc_b = add_bias(X_test_ml_mfcc)

# -- Log-mel spectrogram (CNN) --
X_train_logmel = extract_logmel(X_train, SR)
X_val_logmel   = extract_logmel(X_val, SR)
logmel_mean = X_train_logmel.mean()
logmel_std  = X_train_logmel.std()
X_train_logmel = (X_train_logmel - logmel_mean) / logmel_std
X_val_logmel   = (X_val_logmel   - logmel_mean) / logmel_std

# -- MFCC sequence (RNN / CNN) --
X_train_seq = extract_mfcc_seq(X_train, SR)
X_val_seq   = extract_mfcc_seq(X_val, SR)
X_test_ml_seq = extract_mfcc_seq(X_test_ml, SR)
seq_mean = X_train_seq.mean()
seq_std  = X_train_seq.std()
X_train_seq   = (X_train_seq   - seq_mean) / seq_std
X_val_seq     = (X_val_seq     - seq_mean) / seq_std
X_test_ml_seq = (X_test_ml_seq - seq_mean) / seq_std

print("Feature shapes (train):")
print(f"  PSD bands : {X_train_psd.shape}")
print(f"  MFCC flat : {X_train_mfcc.shape}")
print(f"  Log-mel   : {X_train_logmel.shape}")
print(f"  MFCC seq  : {X_train_seq.shape}")

# Visualize average PSD (training set only)
train_healthy_waveforms = X_train[y_train == 0]
train_unhealthy_waveforms = X_train[y_train == 1]

f_h, _ = periodogram(train_healthy_waveforms[0], fs=SR)
avg_psd_h = np.mean([periodogram(s, fs=SR)[1] for s in train_healthy_waveforms], axis=0)
avg_psd_u = np.mean(
    [periodogram(s, fs=SR)[1] for s in train_unhealthy_waveforms], axis=0
)

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(f_h / 1000, avg_psd_h)
ax[0].set_title(f"Average PSD: Healthy (n={len(train_healthy_waveforms)}, train only)")
ax[0].set_ylabel("PSD")
ax[0].grid()
ax[0].set_xlim(0, 8)
add_bands(ax[0])
ax[1].plot(f_h / 1000, avg_psd_u, color="tab:orange")
ax[1].set_title(
    f"Average PSD: Unhealthy (n={len(train_unhealthy_waveforms)}, train only)"
)
ax[1].set_ylabel("PSD")
ax[1].set_xlabel("Frequency (kHz)")
ax[1].grid()
ax[1].set_xlim(0, 8)
add_bands(ax[1])
ax[1].xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax[0].legend(
    handles=[Patch(facecolor="gray", alpha=0.3, label="Feature bands")],
    fontsize=8,
    loc="upper right",
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "02_average_psd_bands.png"), dpi=150, bbox_inches="tight"
)
plt.show()


# =============================================================================
# PART 2: DEEP LEARNING MODELS  (BPNN, CNN, RNN)
# =============================================================================

# TODO (remaining): CNN, RNN
# Input shapes:
#   CNN  — X_train_logmel  (N, n_mels=64, time_frames)
#   RNN  — X_train_seq     (N, time_frames, n_mfcc=13)

# %% ── 2.1  BPNN ─────────────────────────────────────────────────────────────────

n_inputs = X_train_combined.shape[1]  # 35 features (combined MFCC and PSD)
# 5 layer BPNN
bpnn_net = nn.Sequential(
    nn.Linear(n_inputs, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

# ── Convert to tensors ────────────────────────────────────────────────────────
X_tr_t = torch.tensor(X_train_combined, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N,) → (N, 1)
X_val_t = torch.tensor(X_val_combined, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# ── DataLoader ────────────────────────────────────────────────────────────────
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

# ── Loss and optimizer ────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(bpnn_net.parameters(), lr=1e-3)

# ── Training loop ─────────────────────────────────────────────────────────────
N_EPOCHS = 15
train_losses, val_accs = [], []

for epoch in range(N_EPOCHS):
    bpnn_net.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(bpnn_net(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)  # accumulate total loss
    train_losses.append(epoch_loss / len(X_tr_t))  # mean loss over all samples

    bpnn_net.eval()
    with torch.no_grad():
        val_preds = (torch.sigmoid(bpnn_net(X_val_t)) >= 0.5).long().squeeze()
        val_accs.append(accuracy_score(y_val, val_preds.numpy()))

print(f"BPNN — best val acc: {max(val_accs):.2%} (epoch {val_accs.index(max(val_accs)) + 1})")

# ── Learning curve ────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(train_losses, color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(val_accs, color="tab:orange")
ax2.set_ylabel("Val Accuracy", color="tab:orange")
ax2.set_ylim(0, 1)
ax1.set_title("BPNN Learning Curve")
fig.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "07_bpnn_learning_curve.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# ── Predict wrapper ───────────────────────────────────────────────────────────
def bpnn_predict(X_np):
    bpnn_net.eval()
    with torch.no_grad():
        logits = bpnn_net(torch.tensor(X_np, dtype=torch.float32))
        return (torch.sigmoid(logits) >= 0.5).long().squeeze().numpy()


# %% ── 2.2  CNN ──────────────────────────────────────────────────────────────────

# ── Model definition ─────────────────────────────────────────────────────────
dummy = torch.zeros(1, 1, *X_train_seq.shape[1:])
flat_size = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.MaxPool2d(2),
    nn.Flatten(),
).forward(dummy).shape[1]

cnn_net = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(flat_size, 32), nn.ReLU(),
    nn.Linear(32, 1),
)

# ── Convert to tensors and build DataLoader ───────────────────────────────────
X_tr_cnn = torch.tensor(X_train_seq, dtype=torch.float32).unsqueeze(1)  # (N, 1, time_frames, 13)
X_val_cnn = torch.tensor(X_val_seq, dtype=torch.float32).unsqueeze(1)

cnn_loader = DataLoader(TensorDataset(X_tr_cnn, y_tr_t), batch_size=32, shuffle=True)

# ── Loss and optimizer ────────────────────────────────────────────────────────
cnn_criterion = nn.BCEWithLogitsLoss()
cnn_optimizer = torch.optim.Adam(cnn_net.parameters(), lr=1e-3)

# ── Training loop ─────────────────────────────────────────────────────────────
CNN_EPOCHS = 15
cnn_train_losses, cnn_val_accs = [], []

for epoch in range(CNN_EPOCHS):
    cnn_net.train()
    epoch_loss = 0.0
    for X_batch, y_batch in cnn_loader:
        cnn_optimizer.zero_grad()
        loss = cnn_criterion(cnn_net(X_batch), y_batch)
        loss.backward()
        cnn_optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
    cnn_train_losses.append(epoch_loss / len(X_tr_cnn))

    cnn_net.eval()
    with torch.no_grad():
        val_preds = (torch.sigmoid(cnn_net(X_val_cnn)) >= 0.5).long().squeeze()
        cnn_val_accs.append(accuracy_score(y_val, val_preds.numpy()))
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}/{CNN_EPOCHS}  loss={cnn_train_losses[-1]:.4f}  val={cnn_val_accs[-1]:.2%}")

print(f"CNN  — best val acc: {max(cnn_val_accs):.2%} (epoch {cnn_val_accs.index(max(cnn_val_accs)) + 1})")

# ── Learning curve ────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(cnn_train_losses, color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(cnn_val_accs, color="tab:orange")
ax2.set_ylabel("Val Accuracy", color="tab:orange")
ax2.set_ylim(0, 1)
ax1.set_title("CNN Learning Curve")
fig.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "09_cnn_learning_curve.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# ── Predict wrapper ───────────────────────────────────────────────────────────
def cnn_predict(X_np):
    cnn_net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
        return (torch.sigmoid(cnn_net(X_t)) >= 0.5).long().squeeze().numpy()


# %% ── 2.3  RNN (LSTM) ───────────────────────────────────────────────────────────

rnn_lstm = nn.LSTM(input_size=13, hidden_size=32, batch_first=True)
rnn_fc   = nn.Linear(32, 1)

X_tr_rnn  = torch.tensor(X_train_seq, dtype=torch.float32)
X_val_rnn = torch.tensor(X_val_seq,   dtype=torch.float32)

rnn_loader   = DataLoader(TensorDataset(X_tr_rnn, y_tr_t), batch_size=32, shuffle=True)
rnn_params   = list(rnn_lstm.parameters()) + list(rnn_fc.parameters())
rnn_criterion = nn.BCEWithLogitsLoss()
rnn_optimizer = torch.optim.Adam(rnn_params, lr=1e-3, weight_decay=0.01)

RNN_EPOCHS = 15
rnn_train_losses, rnn_val_accs = [], []

for epoch in range(RNN_EPOCHS):
    rnn_lstm.train()
    rnn_fc.train()
    epoch_loss = 0.0
    for X_batch, y_batch in rnn_loader:
        rnn_optimizer.zero_grad()
        _, (h_n, _) = rnn_lstm(X_batch)
        loss = rnn_criterion(rnn_fc(h_n.squeeze(0)), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn_params, max_norm=1.0)
        rnn_optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
    rnn_train_losses.append(epoch_loss / len(X_tr_rnn))

    rnn_lstm.eval()
    rnn_fc.eval()
    with torch.no_grad():
        _, (h_n, _) = rnn_lstm(X_val_rnn)
        val_preds = (torch.sigmoid(rnn_fc(h_n.squeeze(0))) >= 0.5).long().squeeze()
        rnn_val_accs.append(accuracy_score(y_val, val_preds.numpy()))
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}/{RNN_EPOCHS}  loss={rnn_train_losses[-1]:.4f}  val={rnn_val_accs[-1]:.2%}")

print(f"RNN  — best val acc: {max(rnn_val_accs):.2%} (epoch {rnn_val_accs.index(max(rnn_val_accs)) + 1})")

# ── Learning curve ────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(rnn_train_losses, color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(rnn_val_accs, color="tab:orange")
ax2.set_ylabel("Val Accuracy", color="tab:orange")
ax2.set_ylim(0, 1)
ax1.set_title("RNN (LSTM) Learning Curve")
fig.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "10_rnn_learning_curve.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# ── Predict wrapper ───────────────────────────────────────────────────────────
def rnn_predict(X_np):
    rnn_lstm.eval()
    rnn_fc.eval()
    with torch.no_grad():
        _, (h_n, _) = rnn_lstm(torch.tensor(X_np, dtype=torch.float32))
        return (torch.sigmoid(rnn_fc(h_n.squeeze(0))) >= 0.5).long().squeeze().numpy()

# =============================================================================
# PART 3: MODEL ROBUSTNESS EVALUATION
# =============================================================================

# %% ── 3.1  LOAD & PREPROCESS HW2 TEST SET (with caching) ───────────────────────

HW2_DIR = "hw2_recordings"

if os.path.exists(HW2_CACHE):
    print("Loading HW2 test splices from cache…")
    data = np.load(HW2_CACHE)
    hw2_healthy = data["healthy"]
    hw2_unhealthy = data["unhealthy"]
    print(
        f"  Healthy:   {hw2_healthy.shape[0]}  |  Unhealthy: {hw2_unhealthy.shape[0]}"
    )
else:
    print("Processing HW2 recordings…")
    hw2_healthy, hw2_unhealthy = load_hw2_test_set(
        HW2_DIR, SR, NOISE_SAMPLE_DURATION, EXPECTED_HITS
    )
    np.savez(HW2_CACHE, healthy=hw2_healthy, unhealthy=hw2_unhealthy)
    print(f"Saved HW2 test splices to {HW2_CACHE}")

X_test_hw2 = np.concatenate([hw2_healthy, hw2_unhealthy], axis=0)
y_test_hw2 = np.array([0] * len(hw2_healthy) + [1] * len(hw2_unhealthy))


# %% ── 3.2  FEATURE EXTRACTION FOR HW2 INDEPENDENT TEST SET ─────────────────────

# -- Classical --
X_test_hw2_psd      = (extract_psd_bands(X_test_hw2, SR) - psd_mean) / psd_std
X_test_hw2_mfcc     = (extract_mfcc_flat(X_test_hw2, SR) - mfcc_mean) / mfcc_std
X_test_hw2_psd_b    = add_bias(X_test_hw2_psd)
X_test_hw2_mfcc_b   = add_bias(X_test_hw2_mfcc)
X_test_hw2_combined = np.concatenate([X_test_hw2_psd, X_test_hw2_mfcc], axis=1)

# -- Deep learning --
X_test_hw2_seq = (extract_mfcc_seq(X_test_hw2, SR) - seq_mean) / seq_std

print(
    f"HW2 test set: {len(hw2_healthy)} healthy, {len(hw2_unhealthy)} unhealthy splices"
)


# %% ── 3.3  KNN ─────────────────────────────────────────────────────────────────

k_values = [1, 3, 5, 10]
feature_sets = [
    ("PSD Bands", X_train_psd, X_val_psd),
    ("MFCC", X_train_mfcc, X_val_mfcc),
]

fig, axes = plt.subplots(len(feature_sets), len(k_values), figsize=(14, 6))
for row, (feat_name, X_tr, X_v) in enumerate(feature_sets):
    for col, k in enumerate(k_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_train)
        y_pred = knn.predict(X_v)
        acc = accuracy_score(y_val, y_pred)
        ConfusionMatrixDisplay(
            confusion_matrix(y_val, y_pred), display_labels=["H", "U"]
        ).plot(ax=axes[row, col], cmap="Blues", colorbar=False)
        axes[row, col].set_title(f"k={k}\n{acc:.0%}", fontsize=9)
        if col > 0:
            axes[row, col].set_ylabel("")
        if row == 0:
            axes[row, col].set_xlabel("")
    axes[row, 0].set_ylabel(feat_name, fontsize=10, fontweight="bold")
plt.suptitle("KNN — Validation Confusion Matrices", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "03_knn_confusion.png"), dpi=150, bbox_inches="tight"
)
plt.show()

KNN_K = 5
knn_psd_final = KNeighborsClassifier(n_neighbors=KNN_K).fit(X_train_psd, y_train)
knn_mfcc_final = KNeighborsClassifier(n_neighbors=KNN_K).fit(X_train_mfcc, y_train)
print(
    f"KNN (k={KNN_K})  PSD val: {accuracy_score(y_val, knn_psd_final.predict(X_val_psd)):.2%}  "
    f"test: {accuracy_score(y_test_ml, knn_psd_final.predict(X_test_ml_psd)):.2%}"
)


# %% ── 3.4  DECISION TREE ────────────────────────────────────────────────────────

dt_mfcc = DecisionTreeClassifier(random_state=42, max_depth=5).fit(
    X_train_mfcc, y_train
)
dt_psd = DecisionTreeClassifier(random_state=42, max_depth=6).fit(X_train_psd, y_train)

dt_models = [
    ("MFCC (depth=5)", dt_mfcc, X_train_mfcc, X_val_mfcc),
    ("PSD Bands (depth=6)", dt_psd, X_train_psd, X_val_psd),
]
fig, axes = plt.subplots(2, len(dt_models), figsize=(10, 7))
for col, (label, dt, X_tr, X_v) in enumerate(dt_models):
    for row, (X_eval, y_eval, split_name) in enumerate(
        [(X_tr, y_train, "Train"), (X_v, y_val, "Validation")]
    ):
        y_pred = dt.predict(X_eval)
        ConfusionMatrixDisplay(
            confusion_matrix(y_eval, y_pred), display_labels=["Healthy", "Unhealthy"]
        ).plot(ax=axes[row, col], cmap="Greens", colorbar=False)
        axes[row, col].set_title(
            f"{label} | {split_name}\nAcc: {accuracy_score(y_eval, y_pred):.2%}"
        )
plt.suptitle(
    "Decision Tree: MFCC depth=5 vs PSD depth=6", fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "04_decision_tree_confusion.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# %% ── 3.5  LOGISTIC REGRESSION ─────────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -1 / m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))


alpha = 1
n_iter = 10000
m = len(y_train)

theta_psd = np.zeros(X_train_psd_b.shape[1])
cost_hist_psd = []
for _ in range(n_iter):
    h = sigmoid(X_train_psd_b @ theta_psd)
    theta_psd = theta_psd - (alpha / m) * X_train_psd_b.T @ (h - y_train)
    cost_hist_psd.append(compute_cost(X_train_psd_b, y_train, theta_psd))
print(
    f"LR PSD  — val: {accuracy_score(y_val, (sigmoid(X_val_psd_b @ theta_psd) >= 0.5).astype(int)):.2%}  "
    f"test: {accuracy_score(y_test_ml, (sigmoid(X_test_ml_psd_b @ theta_psd) >= 0.5).astype(int)):.2%}"
)

theta_mfcc = np.zeros(X_train_mfcc_b.shape[1])
cost_hist_mfcc = []
for _ in range(n_iter):
    h = sigmoid(X_train_mfcc_b @ theta_mfcc)
    theta_mfcc = theta_mfcc - (alpha / m) * X_train_mfcc_b.T @ (h - y_train)
    cost_hist_mfcc.append(compute_cost(X_train_mfcc_b, y_train, theta_mfcc))
print(
    f"LR MFCC — val: {accuracy_score(y_val, (sigmoid(X_val_mfcc_b @ theta_mfcc) >= 0.5).astype(int)):.2%}  "
    f"test: {accuracy_score(y_test_ml, (sigmoid(X_test_ml_mfcc_b @ theta_mfcc) >= 0.5).astype(int)):.2%}"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(cost_hist_psd)
axes[0].set_title("LR Cost — PSD")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("J(θ)")
axes[0].grid()
axes[1].plot(cost_hist_mfcc)
axes[1].set_title("LR Cost — MFCC")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("J(θ)")
axes[1].grid()
plt.tight_layout()
plt.show()


# %% ── 3.6  SVM ─────────────────────────────────────────────────────────────────

C_values = [0.01, 0.1, 1.0, 5.0, 10.0, 100.0, 1000.0, 10000.0]
gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0]
feature_sets_svm = [
    ("PSD Bands", X_train_psd, X_val_psd),
    ("MFCC", X_train_mfcc, X_val_mfcc),
]

best_params = {}
fig, axes = plt.subplots(1, len(feature_sets_svm), figsize=(14, 5))
for ax, (name, X_tr, X_v) in zip(axes, feature_sets_svm):
    grid = np.zeros((len(gamma_values), len(C_values)))
    for i, gamma in enumerate(gamma_values):
        for j, C in enumerate(C_values):
            svm = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
            svm.fit(X_tr, y_train)
            grid[i, j] = accuracy_score(y_val, svm.predict(X_v))
    best_idx = np.unravel_index(np.argmax(grid), grid.shape)
    best_params[name] = {"C": C_values[best_idx[1]], "gamma": gamma_values[best_idx[0]]}
    print(
        f"Best SVM ({name}): C={best_params[name]['C']}, "
        f"gamma={best_params[name]['gamma']}, val_acc={grid[best_idx]:.2%}"
    )
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(C_values)))
    ax.set_xticklabels(C_values, rotation=45)
    ax.set_yticks(range(len(gamma_values)))
    ax.set_yticklabels(gamma_values)
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
    ax.set_title(f"SVM Val Accuracy — {name}")
    for i in range(len(gamma_values)):
        for j in range(len(C_values)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.add_patch(
        plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
plt.colorbar(im, ax=axes[-1], label="Validation Accuracy")
plt.suptitle(
    "SVM: C vs Gamma (rbf kernel, tuned on validation)", fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "05_svm_C_gamma_grid.png"), dpi=150, bbox_inches="tight"
)
plt.show()

svm_psd = SVC(
    kernel="rbf",
    C=best_params["PSD Bands"]["C"],
    gamma=best_params["PSD Bands"]["gamma"],
    random_state=42,
).fit(X_train_psd, y_train)
svm_mfcc = SVC(
    kernel="rbf",
    C=best_params["MFCC"]["C"],
    gamma=best_params["MFCC"]["gamma"],
    random_state=42,
).fit(X_train_mfcc, y_train)
for name, model, X_te in [
    ("PSD Bands", svm_psd, X_test_ml_psd),
    ("MFCC", svm_mfcc, X_test_ml_mfcc),
]:
    print(f"SVM ({name})  test: {accuracy_score(y_test_ml, model.predict(X_te)):.2%}")


# %% ── 3.7  DEEP LEARNING MODELS — ALL SPLITS ───────────────────────────────────

dl_splits = [
    ("Train",      X_train_combined, X_train_seq, y_train),
    ("Val",        X_val_combined,   X_val_seq,   y_val),
    ("Test (ML)",  X_test_ml_combined, X_test_ml_seq, y_test_ml),
    ("Test (HW2)", X_test_hw2_combined, X_test_hw2_seq, y_test_hw2),
]

dl_models = [
    ("BPNN", bpnn_predict, 0),  # index 0 = combined features
    ("CNN",  cnn_predict,  1),  # index 1 = seq features
    ("RNN",  rnn_predict,  1),
]

dl_splits_trainvaltest = [
    ("Train",     X_train_combined, X_train_seq, y_train),
    ("Val",       X_val_combined,   X_val_seq,   y_val),
    ("Test (ML)", X_test_ml_combined, X_test_ml_seq, y_test_ml),
]

dl_test_ml_preds = {}  # saved for reuse in 3.10
print("\n--- Deep Learning: Train / Val / Test (ML) ---")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for row, (model_name, predict_fn, feat_idx) in enumerate(dl_models):
    for col, (split_name, X_combined, X_seq, y_true) in enumerate(dl_splits_trainvaltest):
        X_feat = X_combined if feat_idx == 0 else X_seq
        preds = predict_fn(X_feat)
        if split_name == "Test (ML)":
            dl_test_ml_preds[model_name] = preds
        acc = accuracy_score(y_true, preds)
        print(f"  {model_name:6s} {split_name:12s}  acc: {acc:.2%}")
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, preds), display_labels=["H", "U"]
        ).plot(ax=axes[row, col], cmap="Purples", colorbar=False)
        axes[row, col].set_title(f"{model_name} — {split_name}\n{acc:.2%}", fontsize=9)
        if col > 0:
            axes[row, col].set_ylabel("")
plt.suptitle("Deep Learning Models — Train / Val / Test", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "08_dl_trainvaltest_confusion.png"), dpi=150, bbox_inches="tight"
)
plt.show()


# %% ── 3.8  ROBUSTNESS SUMMARY — CONFUSION MATRICES (HW2 test set) ──────────────

lr_psd_pred  = (sigmoid(X_test_hw2_psd_b  @ theta_psd)  >= 0.5).astype(int)
lr_mfcc_pred = (sigmoid(X_test_hw2_mfcc_b @ theta_mfcc) >= 0.5).astype(int)

hw3_models = [
    (f"KNN PSD (k={KNN_K})", knn_psd_final,  X_test_hw2_psd),
    (f"KNN MFCC (k={KNN_K})", knn_mfcc_final, X_test_hw2_mfcc),
    ("DT  PSD (depth=6)",  dt_psd,  X_test_hw2_psd),
    ("DT  MFCC (depth=5)", dt_mfcc, X_test_hw2_mfcc),
    ("SVM PSD",  svm_psd,  X_test_hw2_psd),
    ("SVM MFCC", svm_mfcc, X_test_hw2_mfcc),
]

print("\n--- Robustness Evaluation (HW2 independent test set) ---")
for name, model, X_feat in hw3_models:
    print(f"  {name:25s}  acc: {accuracy_score(y_test_hw2, model.predict(X_feat)):.2%}")
print(f"  {'LR  PSD':25s}  acc: {accuracy_score(y_test_hw2, lr_psd_pred):.2%}")
print(f"  {'LR  MFCC':25s}  acc: {accuracy_score(y_test_hw2, lr_mfcc_pred):.2%}")

bpnn_hw2_pred = bpnn_predict(X_test_hw2_combined)
cnn_hw2_pred  = cnn_predict(X_test_hw2_seq)
rnn_hw2_pred  = rnn_predict(X_test_hw2_seq)

all_hw2_models = [
    (f"KNN PSD (k={KNN_K})", hw3_models[0][1].predict(X_test_hw2_psd),  "classical"),
    (f"KNN MFCC (k={KNN_K})", hw3_models[1][1].predict(X_test_hw2_mfcc), "classical"),
    ("DT  PSD (depth=6)",  hw3_models[2][1].predict(X_test_hw2_psd),  "classical"),
    ("DT  MFCC (depth=5)", hw3_models[3][1].predict(X_test_hw2_mfcc), "classical"),
    ("SVM PSD",  hw3_models[4][1].predict(X_test_hw2_psd),  "classical"),
    ("SVM MFCC", hw3_models[5][1].predict(X_test_hw2_mfcc), "classical"),
    ("LR  PSD",  lr_psd_pred,  "classical"),
    ("LR  MFCC", lr_mfcc_pred, "classical"),
    ("BPNN", bpnn_hw2_pred, "deep"),
    ("CNN",  cnn_hw2_pred,  "deep"),
    ("RNN",  rnn_hw2_pred,  "deep"),
]

print("\n--- Robustness Evaluation (HW2 independent test set) ---")
for name, preds, _ in all_hw2_models:
    print(f"  {name:25s}  acc: {accuracy_score(y_test_hw2, preds):.2%}")

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes_flat = axes.flatten()
for ax, (name, preds, kind) in zip(axes_flat, all_hw2_models):
    cmap = "Oranges" if kind == "deep" else "Blues"
    ConfusionMatrixDisplay(
        confusion_matrix(y_test_hw2, preds), display_labels=["H", "U"]
    ).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"{name}\n{accuracy_score(y_test_hw2, preds):.2%}", fontsize=9)
axes_flat[-1].set_visible(False)
plt.suptitle(
    "All Models — HW2 Independent Test Set", fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "06_robustness_hw2_allmodels.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# %% ── 3.9  ACCURACY COMPARISON BAR CHART (HW2 independent test set) ─────────

names  = [n for n, _, _ in all_hw2_models]
accs   = [accuracy_score(y_test_hw2, p) for _, p, _ in all_hw2_models]
colors = ["#4878CF" if k == "classical" else "#E87722" for _, _, k in all_hw2_models]

kinds  = [k for _, _, k in all_hw2_models]
best_classical = max((a for a, k in zip(accs, kinds) if k == "classical"))
best_deep      = max((a for a, k in zip(accs, kinds) if k == "deep"))
bold_names     = {n for n, a, k in zip(names, accs, kinds) if (k == "classical" and a == best_classical) or (k == "deep" and a == best_deep)}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(names[::-1], [a * 100 for a in accs[::-1]], color=colors[::-1])
ax.axvline(50, color="gray", linewidth=0.8, linestyle="--", label="50% baseline")
for bar, acc in zip(bars, accs[::-1]):
    ax.text(
        bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{acc:.1%}", va="center", fontsize=8,
    )
ax.set_xlim(0, 110)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Model Accuracy — HW2 Independent Test Set")
for label in ax.get_yticklabels():
    if label.get_text() in bold_names:
        label.set_fontweight("bold")
ax.legend(
    handles=[
        Patch(facecolor="#4878CF", label="Classical"),
        Patch(facecolor="#E87722", label="Deep Learning"),
    ],
    loc="lower right", fontsize=8,
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "07_accuracy_comparison_hw2.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# %% ── 3.10  ACCURACY COMPARISON BAR CHART (ML test set) ──────────────────────

lr_psd_ml_pred  = (sigmoid(X_test_ml_psd_b  @ theta_psd)  >= 0.5).astype(int)
lr_mfcc_ml_pred = (sigmoid(X_test_ml_mfcc_b @ theta_mfcc) >= 0.5).astype(int)

all_ml_models = [
    (f"KNN PSD (k={KNN_K})", knn_psd_final.predict(X_test_ml_psd),  "classical"),
    (f"KNN MFCC (k={KNN_K})", knn_mfcc_final.predict(X_test_ml_mfcc), "classical"),
    ("DT  PSD (depth=6)",  dt_psd.predict(X_test_ml_psd),  "classical"),
    ("DT  MFCC (depth=5)", dt_mfcc.predict(X_test_ml_mfcc), "classical"),
    ("SVM PSD",  svm_psd.predict(X_test_ml_psd),  "classical"),
    ("SVM MFCC", svm_mfcc.predict(X_test_ml_mfcc), "classical"),
    ("LR  PSD",  lr_psd_ml_pred,  "classical"),
    ("LR  MFCC", lr_mfcc_ml_pred, "classical"),
    ("BPNN", dl_test_ml_preds["BPNN"], "deep"),
    ("CNN",  dl_test_ml_preds["CNN"],  "deep"),
    ("RNN",  dl_test_ml_preds["RNN"],  "deep"),
]

names_ml  = [n for n, _, _ in all_ml_models]
accs_ml   = [accuracy_score(y_test_ml, p) for _, p, _ in all_ml_models]
colors_ml = ["#4878CF" if k == "classical" else "#E87722" for _, _, k in all_ml_models]
kinds_ml  = [k for _, _, k in all_ml_models]

best_classical_ml = max(a for a, k in zip(accs_ml, kinds_ml) if k == "classical")
best_deep_ml      = max(a for a, k in zip(accs_ml, kinds_ml) if k == "deep")
bold_names_ml     = {n for n, a, k in zip(names_ml, accs_ml, kinds_ml) if (k == "classical" and a == best_classical_ml) or (k == "deep" and a == best_deep_ml)}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(names_ml[::-1], [a * 100 for a in accs_ml[::-1]], color=colors_ml[::-1])
ax.axvline(50, color="gray", linewidth=0.8, linestyle="--", label="50% baseline")
for bar, acc in zip(bars, accs_ml[::-1]):
    ax.text(
        bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{acc:.1%}", va="center", fontsize=8,
    )
ax.set_xlim(0, 110)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Model Accuracy — ML Test Set (15% held-out)")
for label in ax.get_yticklabels():
    if label.get_text() in bold_names_ml:
        label.set_fontweight("bold")
ax.legend(
    handles=[
        Patch(facecolor="#4878CF", label="Classical"),
        Patch(facecolor="#E87722", label="Deep Learning"),
    ],
    loc="lower right", fontsize=8,
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "10_accuracy_comparison_ml.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


# %% ── 3.11  FULL ACCURACY SUMMARY TABLE ─────────────────────────────────────

def _acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

summary_models = [
    # name, train_preds, val_preds, test_ml_preds, hw2_preds
    (
        f"KNN PSD (k={KNN_K})",
        knn_psd_final.predict(X_train_psd),
        knn_psd_final.predict(X_val_psd),
        knn_psd_final.predict(X_test_ml_psd),
        knn_psd_final.predict(X_test_hw2_psd),
    ),
    (
        f"KNN MFCC (k={KNN_K})",
        knn_mfcc_final.predict(X_train_mfcc),
        knn_mfcc_final.predict(X_val_mfcc),
        knn_mfcc_final.predict(X_test_ml_mfcc),
        knn_mfcc_final.predict(X_test_hw2_mfcc),
    ),
    (
        "DT  PSD (depth=6)",
        dt_psd.predict(X_train_psd),
        dt_psd.predict(X_val_psd),
        dt_psd.predict(X_test_ml_psd),
        dt_psd.predict(X_test_hw2_psd),
    ),
    (
        "DT  MFCC (depth=5)",
        dt_mfcc.predict(X_train_mfcc),
        dt_mfcc.predict(X_val_mfcc),
        dt_mfcc.predict(X_test_ml_mfcc),
        dt_mfcc.predict(X_test_hw2_mfcc),
    ),
    (
        "SVM PSD",
        svm_psd.predict(X_train_psd),
        svm_psd.predict(X_val_psd),
        svm_psd.predict(X_test_ml_psd),
        svm_psd.predict(X_test_hw2_psd),
    ),
    (
        "SVM MFCC",
        svm_mfcc.predict(X_train_mfcc),
        svm_mfcc.predict(X_val_mfcc),
        svm_mfcc.predict(X_test_ml_mfcc),
        svm_mfcc.predict(X_test_hw2_mfcc),
    ),
    (
        "LR  PSD",
        (sigmoid(X_train_psd_b  @ theta_psd)  >= 0.5).astype(int),
        (sigmoid(X_val_psd_b    @ theta_psd)  >= 0.5).astype(int),
        lr_psd_ml_pred,
        lr_psd_pred,
    ),
    (
        "LR  MFCC",
        (sigmoid(X_train_mfcc_b @ theta_mfcc) >= 0.5).astype(int),
        (sigmoid(X_val_mfcc_b   @ theta_mfcc) >= 0.5).astype(int),
        lr_mfcc_ml_pred,
        lr_mfcc_pred,
    ),
    (
        "BPNN",
        bpnn_predict(X_train_combined),
        bpnn_predict(X_val_combined),
        dl_test_ml_preds["BPNN"],
        bpnn_hw2_pred,
    ),
    (
        "CNN",
        cnn_predict(X_train_seq),
        cnn_predict(X_val_seq),
        dl_test_ml_preds["CNN"],
        cnn_hw2_pred,
    ),
    (
        "RNN",
        rnn_predict(X_train_seq),
        rnn_predict(X_val_seq),
        dl_test_ml_preds["RNN"],
        rnn_hw2_pred,
    ),
]

col_w = 10
header = f"{'Model':<22} {'Train':>{col_w}} {'Val':>{col_w}} {'Test ML':>{col_w}} {'HW2 Indep':>{col_w}}"
sep    = "-" * len(header)
print(f"\n{sep}")
print(header)
print(sep)
for name, tr, vl, te, hw in summary_models:
    row = (
        f"{name:<22} "
        f"{_acc(y_train, tr):>{col_w}.2%} "
        f"{_acc(y_val,   vl):>{col_w}.2%} "
        f"{_acc(y_test_ml, te):>{col_w}.2%} "
        f"{_acc(y_test_hw2, hw):>{col_w}.2%}"
    )
    print(row)
print(sep)


# %%
