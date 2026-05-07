# %%
# =============================================================================
# Machine Learning Final Project — Bolt Tightening Condition Classification
# Task 1: Dependent test   — random 70/30 split pooled across all flanges
# Task 2: Independent test — leave-one-flange-out (train on 3, test on 1)
# Task 3: Experimental test - evaluate models on unlabelled collected data 
# =============================================================================
import preprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from features import extract_mfcc_sequence, build_features
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from models import knn_eval, svm_eval, cnn_eval, lstm_eval, cnn_train, lstm_train

# %% Load in datasets for tasks 1-3
# preprocess() returns a PreprocessedData dataclass with train/test splits and zscore-normalized splices.
# independent_split() builds the leave-one-flange-out splits for task 2.
data = preprocess.preprocess()

# Build task 2 splits: train on 3 flanges, test on the held-out flange.
# Evaluates whether the classifier generalizes to a bolt it never saw during training.
independent_splits: dict = {}
for flange in range(1, 5):
    independent_splits[flange] = preprocess.independent_split(data.datasets, flange)

# %% Shallow learner: KNN and SVM classifiers
# KNN and SVM are sensitive to feature scale, so features are z-score normalized
# before fitting. The scaler is fit on training data only to prevent data leakage.
# Three feature sets are compared: DWT only (35-D), MFCC only (52-D), and both (87-D).

best_knn_params  = {}  # saved for Task 3: { feature_name: (k, weights, metric) }
best_svm_kernels = {}  # saved for Task 3: { feature_name: kernel }

knn_cm_dep, svm_cm_dep = {}, {}
knn_cm_ind = {n: {} for n in ['DWT', 'MFCC', 'Both']}
svm_cm_ind = {n: {} for n in ['DWT', 'MFCC', 'Both']}

for name in ['DWT', 'MFCC', 'Both']:
    print(f"\n=== {name} Features ===")

    # --- Task 1: Dependent test (random 70/30 split) ---
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(build_features(data.X_train, name))
    X_te = scaler.transform(build_features(data.X_test, name))

    knn_res, best_knn_params[name], knn_cm_dep[name] = knn_eval(X_tr, X_te, data.y_train, data.y_test)
    svm_res, best_svm_kernels[name], svm_cm_dep[name] = svm_eval(X_tr, X_te, data.y_train, data.y_test)
    print(f"  Dependent Test — KNN={knn_res[0][3]:.4f}  SVM={svm_res[0][1]:.4f}")

    # --- Task 2: Independent test (leave-one-flange-out) ---
    for flange in range(1, 5):
        X_train_f, X_test_f, y_train_f, y_test_f = independent_splits[flange]
        scaler = StandardScaler()
        X_tr_ind = scaler.fit_transform(build_features(X_train_f, name))
        X_te_ind = scaler.transform(build_features(X_test_f, name))
        knn_res, _, knn_cm_ind[name][flange] = knn_eval(X_tr_ind, X_te_ind, y_train_f, y_test_f)
        svm_res, _, svm_cm_ind[name][flange] = svm_eval(X_tr_ind, X_te_ind, y_train_f, y_test_f)
        print(f"  Flange {flange} held out — KNN={knn_res[0][3]:.4f}  SVM={svm_res[0][1]:.4f}")


# %% 1D-CNN Model
# 1D-CNN model - Operates on raw z-score normalized waveforms (splice_len = 2450 samples).
# Three conv blocks with max-pooling downsample the signal; AdaptiveAvgPool
# collapses the time dimension to a fixed 64-D feature vector.

# Task 1: Dependent test
acc, cnn_cm_dep, cnn_loss_history_dependent = cnn_eval(data.X_train, data.X_test, data.y_train, data.y_test)
print(f"=== Dependent Test ===\n  Accuracy: {acc:.4f}\n{cnn_cm_dep}")

# Task 2: Independent (leave-one-flange-out)
cnn_cm_ind = {}
cnn_loss_history_independent = {}
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits.items():
    acc, cnn_cm_ind[flange], cnn_loss_history = cnn_eval(X_tr, X_te, y_tr, y_te)
    cnn_loss_history_independent[flange] = cnn_loss_history
    print(f"\n=== Independent Test — Flange {flange} held out ===\n  Accuracy: {acc:.4f}\n{cnn_cm_ind[flange]}")


# %% LSTM Model
# LSTM processes MFCC sequences: each splice is converted to (time_frames, 13),
# where the time dimension captures how spectral content evolves through the tap impulse.
# The final hidden state of the last LSTM layer is passed to the classifier layer.

# Task 1: Dependent test
X_train_mfcc = extract_mfcc_sequence(data.X_train)
X_test_mfcc  = extract_mfcc_sequence(data.X_test)

acc, lstm_cm_dep, lstm_loss_history_dependent = lstm_eval(X_train_mfcc, X_test_mfcc, data.y_train, data.y_test)
print(f"=== Dependent Test ===\n  Accuracy: {acc:.4f}\n{lstm_cm_dep}")

# Task 2: Independent (leave-one-flange-out)
lstm_cm_ind = {}
lstm_loss_history_independent = {}
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits.items():
    X_tr_mfcc = extract_mfcc_sequence(X_tr)
    X_te_mfcc = extract_mfcc_sequence(X_te)
    acc, lstm_cm_ind[flange], lstm_loss_history = lstm_eval(X_tr_mfcc, X_te_mfcc, y_tr, y_te)
    lstm_loss_history_independent[flange] = lstm_loss_history
    print(f"\n=== Independent Test — Flange {flange} held out ===\n  Accuracy: {acc:.4f}\n{lstm_cm_ind[flange]}")


# %% 2-Class Classification (0 ft-lb vs 25+50 ft-lb)
# Evaluate the same models but with binary labels: 0 ft-lb (class 0) vs 25+50 ft-lb (class 1). 
# Tests whether the models can at least distinguish between "no torque" and "some torque"
print("=" * 60)
print("2-CLASS CLASSIFICATION (0 ft-lb vs 25+50 ft-lb)")
print("=" * 60)

# --- Shallow learners (2-class) ---
knn_cm_dep_2, svm_cm_dep_2 = {}, {}
knn_cm_ind_2 = {n: {} for n in ['DWT', 'MFCC', 'Both']}
svm_cm_ind_2 = {n: {} for n in ['DWT', 'MFCC', 'Both']}

for name in ['DWT', 'MFCC', 'Both']:
    print(f"\n=== {name} Features (2-class) ===")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(build_features(data.X_train, name))
    X_te = scaler.transform(build_features(data.X_test, name))
    y_tr_bin = (data.y_train != 0).astype(int)
    y_te_bin = (data.y_test != 0).astype(int)
    knn_res, _, knn_cm_dep_2[name] = knn_eval(X_tr, X_te, y_tr_bin, y_te_bin)
    svm_res, _, svm_cm_dep_2[name] = svm_eval(X_tr, X_te, y_tr_bin, y_te_bin)
    print(f"  Dependent Test — KNN={knn_res[0][3]:.4f}  SVM={svm_res[0][1]:.4f}")

    for flange in range(1, 5):
        X_train_f, X_test_f, y_train_f, y_test_f = independent_splits[flange]
        scaler = StandardScaler()
        X_tr_ind = scaler.fit_transform(build_features(X_train_f, name))
        X_te_ind = scaler.transform(build_features(X_test_f, name))
        y_tr_bin = (y_train_f != 0).astype(int)
        y_te_bin = (y_test_f != 0).astype(int)
        knn_res, _, knn_cm_ind_2[name][flange] = knn_eval(X_tr_ind, X_te_ind, y_tr_bin, y_te_bin)
        svm_res, _, svm_cm_ind_2[name][flange] = svm_eval(X_tr_ind, X_te_ind, y_tr_bin, y_te_bin)
        print(f"  Flange {flange} held out — KNN={knn_res[0][3]:.4f}  SVM={svm_res[0][1]:.4f}")

# --- 1D-CNN (2-class) ---
print("\n=== 1D-CNN (2-class) ===")

y_tr_bin = (data.y_train != 0).astype(int)
y_te_bin = (data.y_test != 0).astype(int)
acc, cnn_cm_dep_2, cnn_loss_history_dependent_two_class = cnn_eval(data.X_train, data.X_test, y_tr_bin, y_te_bin, num_classes=2)
print(f"Dependent Test — Accuracy: {acc:.4f}\n{cnn_cm_dep_2}")
cnn_cm_ind_2 = {}
cnn_loss_history_independent_two_class = {}
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits.items():
    y_tr_bin = (y_tr != 0).astype(int)
    y_te_bin = (y_te != 0).astype(int)
    acc, cnn_cm_ind_2[flange], cnn_loss_history = cnn_eval(X_tr, X_te, y_tr_bin, y_te_bin, num_classes=2)
    cnn_loss_history_independent_two_class[flange] = cnn_loss_history
    print(f"\nFlange {flange} held out — Accuracy: {acc:.4f}\n{cnn_cm_ind_2[flange]}")

# --- LSTM (2-class) ---
print("\n=== LSTM (2-class) ===")

X_train_mfcc = extract_mfcc_sequence(data.X_train)
X_test_mfcc  = extract_mfcc_sequence(data.X_test)

y_tr_bin = (data.y_train != 0).astype(int)
y_te_bin = (data.y_test != 0).astype(int)
acc, lstm_cm_dep_2, lstm_loss_history_dependent_two_class = lstm_eval(X_train_mfcc, X_test_mfcc, y_tr_bin, y_te_bin, num_classes=2)
print(f"Dependent Test — Accuracy: {acc:.4f}\n{lstm_cm_dep_2}")

lstm_cm_ind_2 = {}
lstm_loss_history_independent_two_class = {}
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits.items():
    X_tr_mfcc = extract_mfcc_sequence(X_tr)
    X_te_mfcc = extract_mfcc_sequence(X_te)
    y_tr_bin = (y_tr != 0).astype(int)
    y_te_bin = (y_te != 0).astype(int)
    acc, lstm_cm_ind_2[flange], lstm_loss_history = lstm_eval(X_tr_mfcc, X_te_mfcc, y_tr_bin, y_te_bin, num_classes=2)
    lstm_loss_history_independent_two_class[flange] = lstm_loss_history
    print(f"\nFlange {flange} held out — Accuracy: {acc:.4f}\n{lstm_cm_ind_2[flange]}")
    
# %% Task 3: Experimental test — predict loading condition on unlabelled experimental data
# Models are retrained on the full labeled dataset (train + test) before predicting,
# since Task 2 already validated generalization and no held-out split is needed here.

LABEL_MAP = {0: "0ftlb", 1: "25ftlb", 2: "50ftlb"}

X_all = np.concatenate([data.X_train, data.X_test], axis=0)
y_all = np.concatenate([data.y_train, data.y_test], axis=0)

def print_flange_preds(model_name, preds, flange_ids):
    print(f"\n--- {model_name} ---")
    for flange in sorted(np.unique(flange_ids)):
        mask   = flange_ids == flange
        counts = {LABEL_MAP[c]: int(np.sum(preds[mask] == c)) for c in [0, 1, 2]}
        print(f"  Flange {flange}: {counts}")

# --- Shallow models (KNN, SVM) ---
# Best hyperparams already found in Task 1; retrain those configs on the full labeled dataset.
for name in ['DWT', 'MFCC', 'Both']:
    print(f"\n=== Task 3 — {name} Features ===")

    best_k, best_w, best_m = best_knn_params[name]
    best_kernel             = best_svm_kernels[name]

    scaler_all = StandardScaler()
    X_all_feat = scaler_all.fit_transform(build_features(X_all, name))
    X_exp_feat = scaler_all.transform(build_features(data.X_experimental, name))

    knn_full = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, metric=best_m)
    knn_full.fit(X_all_feat, y_all)

    svm_full = SVC(kernel=best_kernel)
    svm_full.fit(X_all_feat, y_all)

    print_flange_preds(f"KNN ({name})", knn_full.predict(X_exp_feat), data.y_experimental_flange)
    print_flange_preds(f"SVM ({name})", svm_full.predict(X_exp_feat), data.y_experimental_flange)

# --- CNN ---
print("\n=== Task 3 — CNN ===")
cnn_model, _ = cnn_train(X_all, y_all)
X_exp_cnn = torch.tensor(data.X_experimental, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    cnn_preds = cnn_model(X_exp_cnn).argmax(dim=1).numpy()
print_flange_preds("CNN", cnn_preds, data.y_experimental_flange)

# --- LSTM ---
print("\n=== Task 3 — LSTM ===")
X_all_mfcc = extract_mfcc_sequence(X_all)
X_exp_mfcc = extract_mfcc_sequence(data.X_experimental)
lstm_model, _  = lstm_train(X_all_mfcc, y_all)
X_exp_lstm  = torch.tensor(X_exp_mfcc, dtype=torch.float32)
with torch.no_grad():
    lstm_preds = lstm_model(X_exp_lstm).argmax(dim=1).numpy()
print_flange_preds("LSTM", lstm_preds, data.y_experimental_flange)


# %% Confusion Matrices of best models across tasks

LABELS_3 = ["0 ft·lb", "25 ft·lb", "50 ft·lb"]
LABELS_2 = ["0 ft·lb", "25/50 ft·lb"]
FEAT_NAMES = ['DWT', 'MFCC', 'Both']


def _sum_cms(cm_dict: dict) -> np.ndarray:
    """Sum confusion matrices across all flanges into one combined matrix."""
    return sum(cm_dict.values())


def plot_confusion_matrices(cms, titles, labels, suptitle):
    """Plot 8 confusion matrices in a 2×4 grid with row-normalized values."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, cm, title in zip(axes, cms, titles):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')

        for r in range(len(labels)):
            for c in range(len(labels)):
                val = cm_norm[r, c]
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if val > 0.55 else 'black')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Layout: row 0 = KNN (DWT, MFCC, Both) + CNN | row 1 = SVM (DWT, MFCC, Both) + LSTM
_titles = [f'KNN ({n})' for n in FEAT_NAMES] + ['CNN'] + \
          [f'SVM ({n})' for n in FEAT_NAMES] + ['LSTM']

# 3-class dependent
plot_confusion_matrices(
    [knn_cm_dep[n] for n in FEAT_NAMES] + [cnn_cm_dep] +
    [svm_cm_dep[n] for n in FEAT_NAMES] + [lstm_cm_dep],
    _titles, LABELS_3,
    "3-Class — Dependent Test (row-normalized confusion matrices)",
)

# 3-class independent (CMs summed across all 4 held-out flanges)
plot_confusion_matrices(
    [_sum_cms(knn_cm_ind[n]) for n in FEAT_NAMES] + [_sum_cms(cnn_cm_ind)] +
    [_sum_cms(svm_cm_ind[n]) for n in FEAT_NAMES] + [_sum_cms(lstm_cm_ind)],
    _titles, LABELS_3,
    "3-Class — Independent Test, all flanges combined (row-normalized confusion matrices)",
)

# 2-class dependent
plot_confusion_matrices(
    [knn_cm_dep_2[n] for n in FEAT_NAMES] + [cnn_cm_dep_2] +
    [svm_cm_dep_2[n] for n in FEAT_NAMES] + [lstm_cm_dep_2],
    _titles, LABELS_2,
    "2-Class — Dependent Test (row-normalized confusion matrices)",
)

# 2-class independent (CMs summed across all 4 held-out flanges)
plot_confusion_matrices(
    [_sum_cms(knn_cm_ind_2[n]) for n in FEAT_NAMES] + [_sum_cms(cnn_cm_ind_2)] +
    [_sum_cms(svm_cm_ind_2[n]) for n in FEAT_NAMES] + [_sum_cms(lstm_cm_ind_2)],
    _titles, LABELS_2,
    "2-Class — Independent Test, all flanges combined (row-normalized confusion matrices)",
)

# %% 3-class: selected models only (KNN MFCC, SVM MFCC, CNN, LSTM) — 2×4 grid

_sel_cms_dep = [knn_cm_dep['MFCC'], svm_cm_dep['MFCC'], cnn_cm_dep, lstm_cm_dep]
_sel_cms_ind = [_sum_cms(knn_cm_ind['MFCC']), _sum_cms(svm_cm_ind['MFCC']),
                _sum_cms(cnn_cm_ind), _sum_cms(lstm_cm_ind)]
_sel_titles  = ['KNN (MFCC)', 'SVM (MFCC)', 'CNN', 'LSTM']

fig, axes = plt.subplots(2, 4, figsize=(16, 8),gridspec_kw={'hspace': 0.15})
for col, (cm_dep, cm_ind, title) in enumerate(zip(_sel_cms_dep, _sel_cms_ind, _sel_titles)):
    for row, cm in enumerate([cm_dep, cm_ind]):
        ax = axes[row, col]
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS_3)))
        ax.set_yticks(range(len(LABELS_3)))
        ax.tick_params(bottom=(row == 1), left=(col == 0))

        ax.set_xticklabels(LABELS_3, rotation=30, ha='right', fontsize=12) if row == 1 else ax.set_xticklabels([])
        ax.set_yticklabels(LABELS_3, fontsize=12) if col == 0 else ax.set_yticklabels([])
        ax.set_xlabel('Predicted', fontsize=12) if row == 1 else ax.set_xlabel('')
        ax.set_ylabel('True', fontsize=12) if col == 0 else ax.set_ylabel('')
        if row == 0:
            ax.set_title(title, fontsize=16, fontweight='bold')
        for r in range(len(LABELS_3)):
            for c in range(len(LABELS_3)):
                val = cm_norm[r, c]
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=16, color='white' if val > 0.55 else 'black')

axes[0, 3].annotate('Dependent', xy=(1.05, 0.5), xycoords='axes fraction',
                    fontsize=16, fontweight='bold', va='center', rotation=270)
axes[1, 3].annotate('Independent', xy=(1.05, 0.5), xycoords='axes fraction',
                    fontsize=16, fontweight='bold', va='center', rotation=270)

fig.suptitle('3-Class Confusion Matrices',
             fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

# %% Loss curves for CNN and LSTM — dependent and independent tests

def _plot_loss_grid(dep_cnn, ind_cnn, dep_lstm, ind_lstm, suptitle):
    """2×2 grid: rows=model, cols=dep/ind. Independent plots show all 4 flanges + mean."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    flange_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    pairs = [
        (axes[0, 0], dep_cnn,  "CNN — Dependent",   False),
        (axes[0, 1], ind_cnn,  "CNN — Independent",  True),
        (axes[1, 0], dep_lstm, "LSTM — Dependent",  False),
        (axes[1, 1], ind_lstm, "LSTM — Independent", True),
    ]

    for ax, history, title, is_ind in pairs:
        if not is_ind:
            ax.plot(history, color="steelblue", linewidth=1.8)
        else:
            curves = [np.array(history[f]) for f in sorted(history)]
            max_len = max(len(c) for c in curves)
            padded = np.array([np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in curves])
            for i, (f, curve) in enumerate(zip(sorted(history), curves)):
                ax.plot(curve, color=flange_colors[i], linewidth=1, alpha=0.6, label=f"Flange {f}")
            mean_curve = padded.mean(axis=0)
            ax.plot(mean_curve, color="black", linewidth=2, linestyle="--", label="Mean")
            ax.legend(fontsize=7, loc="upper right")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Cross-entropy loss", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


_plot_loss_grid(
    cnn_loss_history_dependent,
    cnn_loss_history_independent,
    lstm_loss_history_dependent,
    lstm_loss_history_independent,
    "Training Loss — 3-Class Classification",
)

_plot_loss_grid(
    cnn_loss_history_dependent_two_class,
    cnn_loss_history_independent_two_class,
    lstm_loss_history_dependent_two_class,
    lstm_loss_history_independent_two_class,
    "Training Loss — 2-Class Classification",
)
# %% Waveform visualization in different domains (time, frequency, MFCC) for one example splice from each loading condition
# This is for qualitative analysis and presentation purposes, not part of the main evaluation pipeline.

SR = 48000
CONDITION_LABELS = {0: "0 ft·lb", 1: "25 ft·lb", 2: "50 ft·lb"}
CONDITION_COLORS = {0: '#4C72B0', 1: '#DD8452', 2: '#55A868'}

# One representative splice per loading condition (first occurrence in training set)
example_splices = {
    label: data.X_train[np.where(data.y_train == label)[0][0]]
    for label in CONDITION_LABELS
}

fig, axes = plt.subplots(2, 3, figsize=(16, 6))

for col, (label, splice) in enumerate(example_splices.items()):
    condition = CONDITION_LABELS[label]
    color     = CONDITION_COLORS[label]
    t_ms      = np.arange(len(splice)) / SR * 1000

    # --- Time domain (row 0) ---
    ax = axes[0, col]
    ax.plot(t_ms, splice, linewidth=1.5, color=color)
    ax.set_xlabel('')
    ax.set_title(f'{condition}', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel('Amplitude (norm)', fontsize=16)

    # --- MFCC spectrogram (row 1) ---
    ax = axes[1, col]
    mfcc = extract_mfcc_sequence(splice[np.newaxis])[0].T  # (13, time_frames)
    im = ax.imshow(
        mfcc, aspect='auto', origin='lower', cmap='coolwarm',
        extent=[0, t_ms[-1], 0.5, 13.5],
    )
    ax.set_xlabel('Time (ms)', fontsize=16)
    ax.set_yticks([1, 4, 7, 10, 13])
    ax.tick_params(labelsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if col == 0:
        ax.set_ylabel('MFCC index', fontsize=16)

# Row labels on the left
axes[0, 0].annotate('Time', xy=(-0.35, 0.5), xycoords='axes fraction',
                    fontsize=16, fontweight='bold', va='center', rotation=90)
axes[1, 0].annotate('Time-Frequency', xy=(-0.35, 0.5), xycoords='axes fraction',
                    fontsize=16, fontweight='bold', va='center', rotation=90)

fig.suptitle('Waveform Analysis by Loading Condition', fontsize=20, fontweight='bold')
plt.tight_layout()
fig.savefig('waveform_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %% Summary accuracies — all models, dependent and independent tests

_FEAT = ['DWT', 'MFCC', 'Both']

def _acc(cm): return cm.trace() / cm.sum() * 100

rows = (
    [('KNN', n, _acc(knn_cm_dep[n]), _acc(sum(knn_cm_ind[n].values()))) for n in _FEAT] +
    [('SVM', n, _acc(svm_cm_dep[n]), _acc(sum(svm_cm_ind[n].values()))) for n in _FEAT] +
    [('CNN',  '-', _acc(cnn_cm_dep),  _acc(sum(cnn_cm_ind.values()))),
     ('LSTM', '-', _acc(lstm_cm_dep), _acc(sum(lstm_cm_ind.values())))]
)

print(f"  {'Model':<6} {'Features':<8} {'Dependent':>10} {'Independent':>13}")
print("  " + "-" * 40)
for model, feat, dep, ind in rows:
    print(f"  {model:<6} {feat:<8} {dep:>9.2f}%  {ind:>10.2f}%")

# %%
