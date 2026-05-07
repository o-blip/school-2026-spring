import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import product
from typing import Sequence 
    

def knn_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_range: Sequence[int] = range(1, 11),
) -> list[tuple]:
    """Evaluate KNN with different hyperparameters.
    Weights: 'uniform' vs 'distance'; Distance metrics: 'euclidean' vs 'manhattan'.
    k_range: range of k values to evaluate (default 1 to 10).
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
    best_k, best_w, best_m, _ = results[0]
    best_knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, metric=best_m)
    best_knn.fit(X_train, y_train)
    return results, (best_k, best_w, best_m), confusion_matrix(y_test, best_knn.predict(X_test))

