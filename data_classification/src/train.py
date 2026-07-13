from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def find_best_k(X_train, y_train, max_k=20):
    """
    Tests k values from 1 to max_k using cross-validation
    and returns the k with the lowest error rate (the 'elbow').
    """
    error_rates = []
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        error_rates.append(1 - scores.mean())

    best_k = np.argmin(error_rates) + 1
    return best_k, error_rates

def train_model(X_train, y_train, n_neighbors=5):
    """
    Instantiate -> Fit: trains a KNN classifier on the training data.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model