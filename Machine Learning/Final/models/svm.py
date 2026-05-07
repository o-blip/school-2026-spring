import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Sequence 

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
    best_kernel, _ = results[0]
    best_svm = SVC(kernel=best_kernel)
    best_svm.fit(X_train, y_train)
    return results, best_kernel, confusion_matrix(y_test, best_svm.predict(X_test))

