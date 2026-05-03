# %%
# =============================================================================
# Machine Learning Final Project — Bolt Tightening Condition Classification
# Task 1: Dependent test   — random 70/30 split pooled across all flanges
# Task 2: Independent test — leave-one-flange-out (train on 3, test on 1)
# =============================================================================
import preprocess
from features import extract_mfcc_sequence, build_features
from sklearn.preprocessing import StandardScaler
from models import knn_sweep, svm_eval, cnn_eval, lstm_eval

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

for name in ['DWT', 'MFCC', 'Both']:
    print(f"\n=== {name} Features ===")

    # --- Task 1: Dependent test (random 70/30 split) ---
    X_tr = build_features(data.X_train, name)
    X_te = build_features(data.X_test, name)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)  # fit only on training data
    X_te = scaler.transform(X_te)      # apply same transform to test (no leakage)

    knn_res = knn_sweep(X_tr, X_te, data.y_train, data.y_test)
    svm_res = svm_eval(X_tr, X_te, data.y_train, data.y_test)
    print("  Dependent Test:")
    print(f"    KNN best: acc={knn_res[0][3]:.4f}")
    print(f"    SVM best: acc={svm_res[0][1]:.4f}")

    # --- Task 2: Independent test (leave-one-flange-out) ---
    for flange in range(1, 5):
        X_train_f, X_test_f, y_train_f, y_test_f = independent_splits[flange]
        X_tr_ind = build_features(X_train_f, name)
        X_te_ind = build_features(X_test_f, name)
        scaler = StandardScaler()
        X_tr_ind = scaler.fit_transform(X_tr_ind)
        X_te_ind = scaler.transform(X_te_ind)

        knn_res = knn_sweep(X_tr_ind, X_te_ind, y_train_f, y_test_f)
        svm_res = svm_eval(X_tr_ind, X_te_ind, y_train_f, y_test_f)
        print(f"  Flange {flange} held out: KNN={knn_res[0][3]:.4f}  SVM={svm_res[0][1]:.4f}")


# %% 1D-CNN Model
# Operates on raw z-score normalized waveforms (splice_len = 2450 samples).
# Three conv blocks with max-pooling downsample the signal; AdaptiveAvgPool
# collapses the time dimension to a fixed 64-D feature vector.

# Task 1: Dependent test
acc, cm = cnn_eval(data.X_train, data.X_test, data.y_train, data.y_test)
print(f"=== Dependent Test ===\n  Accuracy: {acc:.4f}\n{cm}")

# Task 2: Independent (leave-one-flange-out)
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits.items():
    acc, cm = cnn_eval(X_tr, X_te, y_tr, y_te)
    print(f"\n=== Independent Test — Flange {flange} held out ===\n  Accuracy: {acc:.4f}\n{cm}")


# %% LSTM Model
# LSTM processes MFCC sequences: each splice is converted to (time_frames, 13),
# where the time dimension captures how spectral content evolves through the tap impulse.
# The final hidden state of the last LSTM layer is passed to the classifier layer.

# Task 1: Dependent test
X_train_mfcc = extract_mfcc_sequence(data.X_train)
X_test_mfcc  = extract_mfcc_sequence(data.X_test)

acc, cm = lstm_eval(X_train_mfcc, X_test_mfcc, data.y_train, data.y_test)
print(f"=== Dependent Test ===\n  Accuracy: {acc:.4f}\n{cm}")

# Task 2: Independent (leave-one-flange-out)
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits.items():
    X_tr_mfcc = extract_mfcc_sequence(X_tr)
    X_te_mfcc = extract_mfcc_sequence(X_te)
    acc, cm = lstm_eval(X_tr_mfcc, X_te_mfcc, y_tr, y_te)
    print(f"\n=== Independent Test — Flange {flange} held out ===\n  Accuracy: {acc:.4f}\n{cm}")


# %%