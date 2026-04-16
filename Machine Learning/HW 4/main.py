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

from preprocessing import load_datasets, load_hw2_test_set
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
# Cache avoids re-running the slow noise-reduction + splicing pipeline on every
# kernel restart.  Delete cache/*.npz to force a full reprocess.

TRAIN_CACHE = os.path.join(CACHE_DIR, "train_splices.npz")
HW2_CACHE = os.path.join(CACHE_DIR, "hw2_splices.npz")

if os.path.exists(TRAIN_CACHE):
    print("Loading training splices from cache…")
    data = np.load(TRAIN_CACHE)
    healthy_hits = data["healthy"]
    unhealthy_hits = data["unhealthy"]
    noise_plot_data = None  # not available from cache; skip noise-comparison plot
    print(
        f"  Healthy:   {healthy_hits.shape[0]}  |  Unhealthy: {unhealthy_hits.shape[0]}"
    )
else:
    print("Processing training recordings (first run — this may take a while)…")
    # Preprocessing: splicing, normalizing, noise reduction, etc.
    healthy_hits, unhealthy_hits, noise_plot_data = load_datasets(
        ML_RECORDINGS_DIR, WAV_DIR, SR, NOISE_SAMPLE_DURATION, EXPECTED_HITS
    )
    np.savez(TRAIN_CACHE, healthy=healthy_hits, unhealthy=unhealthy_hits)
    print(f"Saved training splices to {TRAIN_CACHE}")

# ── Noise-reduction comparison plot (only on first run) ──────────────────────
if noise_plot_data is not None:
    raw = noise_plot_data["raw"]
    clean = noise_plot_data["clean"]
    t = np.arange(len(raw)) / SR
    nd = noise_plot_data["noise_sample_duration"]
    ds = noise_plot_data["ds_num"]
    snum = noise_plot_data["num"]
    lbl = noise_plot_data["label"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(t, raw, linewidth=0.5)
    ax[0].axvspan(0, nd, alpha=0.2, color="red", label=f"Noise sample ({nd}s)")
    ax[0].set_title(f"Raw — Dataset {ds}, s_{snum} ({lbl})")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid()
    ax[1].plot(t, clean, linewidth=0.5, color="tab:orange")
    ax[1].axvspan(0, nd, alpha=0.2, color="red", label=f"Noise sample ({nd}s)")
    ax[1].set_title("After spectral gating")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid()
    ax[1].set_xlim(0, t[-1])
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGES_DIR, "01_noise_reduction_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %% ── 1.3  TRAIN / VALIDATION SPLIT ────────────────────────────────────────────

X_all = np.concatenate([healthy_hits, unhealthy_hits], axis=0)
y_all = np.array(
    [0] * len(healthy_hits) + [1] * len(unhealthy_hits)
)  # 0=healthy, 1=unhealthy

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.20, shuffle=True, random_state=42, stratify=y_all
)

print(
    f"Train — {len(X_train):4d} splices  "
    f"({(y_train == 0).sum()} healthy, {(y_train == 1).sum()} unhealthy)"
)
print(
    f"Val   — {len(X_val):4d} splices  "
    f"({(y_val == 0).sum()} healthy, {(y_val == 1).sum()} unhealthy)"
)
print("Test  — HW2 recordings (loaded below)")


# %% ── 1.4  FEATURE EXTRACTION ───────────────────────────────────────────────────
# Classical flat features (BPNN / KNN / DT / LR / SVM)

# -- PSD bands --
X_train_psd = extract_psd_bands(X_train, SR)
X_val_psd = extract_psd_bands(X_val, SR)
psd_mean, psd_std = X_train_psd.mean(axis=0), X_train_psd.std(axis=0)
X_train_psd = (X_train_psd - psd_mean) / psd_std
X_val_psd = (X_val_psd - psd_mean) / psd_std

# -- MFCC flat --
X_train_mfcc = extract_mfcc_flat(X_train, SR)
X_val_mfcc = extract_mfcc_flat(X_val, SR)
mfcc_mean, mfcc_std = X_train_mfcc.mean(axis=0), X_train_mfcc.std(axis=0)
X_train_mfcc = (X_train_mfcc - mfcc_mean) / mfcc_std
X_val_mfcc = (X_val_mfcc - mfcc_mean) / mfcc_std

# -- Bias-augmented (logistic regression) --
X_train_psd_b, X_val_psd_b = add_bias(X_train_psd), add_bias(X_val_psd)
X_train_mfcc_b, X_val_mfcc_b = add_bias(X_train_mfcc), add_bias(X_val_mfcc)

# -- Log-mel spectrogram (CNN) --
X_train_logmel = extract_logmel(X_train, SR)  # (N, n_mels, time_frames)
X_val_logmel = extract_logmel(X_val, SR)

# -- MFCC sequence (RNN) --
X_train_seq = extract_mfcc_seq(X_train, SR)  # (N, time_frames, n_mfcc)
X_val_seq = extract_mfcc_seq(X_val, SR)

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

# TODO: implement and train BPNN, CNN, and RNN models.
#
# Input shapes:
#   BPNN — X_train_psd  or  X_train_mfcc   (N, n_features)
#   CNN  — X_train_logmel                   (N, n_mels=64, time_frames)
#   RNN  — X_train_seq                      (N, time_frames, n_mfcc=13)
#
# After training, expose these variables for Part 3:
#   bpnn_model, cnn_model, rnn_model
#   — each should support a .predict(X) → y_pred interface


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

X_test = np.concatenate([hw2_healthy, hw2_unhealthy], axis=0)
y_test = np.array([0] * len(hw2_healthy) + [1] * len(hw2_unhealthy))


# %% ── 3.2  FEATURE EXTRACTION FOR TEST SET ────────────────────────────────────

# -- Classical --
X_test_psd = (extract_psd_bands(X_test, SR) - psd_mean) / psd_std
X_test_mfcc = (extract_mfcc_flat(X_test, SR) - mfcc_mean) / mfcc_std
X_test_psd_b = add_bias(X_test_psd)
X_test_mfcc_b = add_bias(X_test_mfcc)

# -- Deep learning --
X_test_logmel = extract_logmel(X_test, SR)
X_test_seq = extract_mfcc_seq(X_test, SR)

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
    f"KNN (k={KNN_K})  PSD: {accuracy_score(y_test, knn_psd_final.predict(X_test_psd)):.2%}  "
    f"MFCC: {accuracy_score(y_test, knn_mfcc_final.predict(X_test_mfcc)):.2%}"
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
    f"test: {accuracy_score(y_test, (sigmoid(X_test_psd_b @ theta_psd) >= 0.5).astype(int)):.2%}"
)

theta_mfcc = np.zeros(X_train_mfcc_b.shape[1])
cost_hist_mfcc = []
for _ in range(n_iter):
    h = sigmoid(X_train_mfcc_b @ theta_mfcc)
    theta_mfcc = theta_mfcc - (alpha / m) * X_train_mfcc_b.T @ (h - y_train)
    cost_hist_mfcc.append(compute_cost(X_train_mfcc_b, y_train, theta_mfcc))
print(
    f"LR MFCC — val: {accuracy_score(y_val, (sigmoid(X_val_mfcc_b @ theta_mfcc) >= 0.5).astype(int)):.2%}  "
    f"test: {accuracy_score(y_test, (sigmoid(X_test_mfcc_b @ theta_mfcc) >= 0.5).astype(int)):.2%}"
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
    ("PSD Bands", svm_psd, X_test_psd),
    ("MFCC", svm_mfcc, X_test_mfcc),
]:
    print(f"SVM ({name})  test: {accuracy_score(y_test, model.predict(X_te)):.2%}")


# %% ── 3.7  HW4 DEEP LEARNING MODELS ON TEST SET ────────────────────────────────

# TODO: once BPNN, CNN, and RNN are trained in Part 2, evaluate them here.
#
#   bpnn_test_pred = bpnn_model.predict(X_test_psd)   # or X_test_mfcc
#   cnn_test_pred  = cnn_model.predict(X_test_logmel)
#   rnn_test_pred  = rnn_model.predict(X_test_seq)


# %% ── 3.8  ROBUSTNESS SUMMARY — CONFUSION MATRICES (HW2 test set) ──────────────

lr_psd_pred = (sigmoid(X_test_psd_b @ theta_psd) >= 0.5).astype(int)
lr_mfcc_pred = (sigmoid(X_test_mfcc_b @ theta_mfcc) >= 0.5).astype(int)

hw3_models = [
    (f"KNN PSD (k={KNN_K})", knn_psd_final, X_test_psd),
    (f"KNN MFCC (k={KNN_K})", knn_mfcc_final, X_test_mfcc),
    ("DT  PSD (depth=6)", dt_psd, X_test_psd),
    ("DT  MFCC (depth=5)", dt_mfcc, X_test_mfcc),
    ("SVM PSD", svm_psd, X_test_psd),
    ("SVM MFCC", svm_mfcc, X_test_mfcc),
]

print("\n--- Robustness Evaluation (HW2 test set) ---")
for name, model, X_feat in hw3_models:
    print(f"  {name:25s}  acc: {accuracy_score(y_test, model.predict(X_feat)):.2%}")
print(f"  {'LR  PSD':25s}  acc: {accuracy_score(y_test, lr_psd_pred):.2%}")
print(f"  {'LR  MFCC':25s}  acc: {accuracy_score(y_test, lr_mfcc_pred):.2%}")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes_flat = axes.flatten()
for ax, (name, model, X_feat) in zip(axes_flat, hw3_models):
    y_pred = model.predict(X_feat)
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=["H", "U"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\n{accuracy_score(y_test, y_pred):.2%}", fontsize=9)
for ax, (preds, title) in zip(
    axes_flat[6:],
    [
        (lr_psd_pred, f"LR PSD\n{accuracy_score(y_test, lr_psd_pred):.2%}"),
        (lr_mfcc_pred, f"LR MFCC\n{accuracy_score(y_test, lr_mfcc_pred):.2%}"),
    ],
):
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds), display_labels=["H", "U"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=9)
plt.suptitle(
    "Model Robustness — HW2 Test Set (HW3 models)", fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(IMAGES_DIR, "06_robustness_hw2_hw3models.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()


