# %%
from preprocess import preprocess, independent_split
from features import extract_dwt_features, extract_mfcc_sequence, extract_mfcc_features
from sklearn.preprocessing import StandardScaler
from models import knn_sweep, svm_eval, cnn_eval, lstm_eval


# %% Load in datasets for tasks 1-3
data = preprocess() # preprocessed data in a dataclass
data_zscore = preprocess(normalization='zscore')

# independent testing: train on 3 datasets, test on remaining 1 dataset
independent_splits = {}
for flange in range(1, 5):
    independent_splits[flange] = independent_split(data.datasets, flange)

independent_splits_zscore = {}
for flange in range(1, 5):
    independent_splits_zscore[flange] = independent_split(data_zscore.datasets, flange)

    
# %% Shallow learner: KNN classifer
# first normalize the features since KNN is sensative to larger features
# Extract features: MFCC and DWT features, as well as a combined vector

def build_features(X, feature_type):
    if feature_type == 'DWT':
        return extract_dwt_features(X)
    elif feature_type == 'MFCC':
        return extract_mfcc_features(X)
    elif feature_type == 'Both':
        return np.hstack([extract_dwt_features(X), extract_mfcc_features(X)])

for name in ['DWT', 'MFCC', 'Both']:
    print(f"\n=== {name} Features ===")
    
    # Task 1
    X_tr = build_features(data.X_train, name)
    X_te = build_features(data.X_test, name)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    knn_res = knn_sweep(X_tr, X_te, data.y_train, data.y_test)
    svm_res = svm_eval(X_tr, X_te, data.y_train, data.y_test)
    print(f"  Dependent Test:")
    print(f"    KNN best: acc={knn_res[0][3]:.4f}")
    print(f"    SVM best: acc={svm_res[0][1]:.4f}")
    
    # Task 2
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
# Task 1
acc, cm = cnn_eval(data_zscore.X_train, data_zscore.X_test, data_zscore.y_train, data_zscore.y_test)
print(f"=== Dependent Test ===\n  Accuracy: {acc:.4f}\n{cm}")

# Task 2
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits_zscore.items():
    acc, cm = cnn_eval(X_tr, X_te, y_tr, y_te)
    print(f"\n=== Independent Test — Flange {flange} held out ===\n  Accuracy: {acc:.4f}\n{cm}")


# %% LSTM Model
# extract MFCC sequences
X_train_mfcc = extract_mfcc_sequence(data_zscore.X_train)
X_test_mfcc  = extract_mfcc_sequence(data_zscore.X_test)

# Task 1
acc, cm = lstm_eval(X_train_mfcc, X_test_mfcc, data_zscore.y_train, data_zscore.y_test)
print(f"=== Dependent Test ===\n  Accuracy: {acc:.4f}\n{cm}")

# Task 2
for flange, (X_tr, X_te, y_tr, y_te) in independent_splits_zscore.items():
    X_tr_mfcc = extract_mfcc_sequence(X_tr)
    X_te_mfcc = extract_mfcc_sequence(X_te)
    acc, cm = lstm_eval(X_tr_mfcc, X_te_mfcc, y_tr, y_te)
    print(f"\n=== Independent Test — Flange {flange} held out ===\n  Accuracy: {acc:.4f}\n{cm}")
    
    
