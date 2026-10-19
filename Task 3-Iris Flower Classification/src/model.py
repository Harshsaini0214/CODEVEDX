# model module
"""
Model definitions, training wrappers, and hyperparameter grids for Iris classification.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# ─────────────────────── Model Registry ───────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(random_state=42),
    "SVM":                  SVC(probability=True, random_state=42),
    "K-Nearest Neighbors":  KNeighborsClassifier(),
    "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
}

PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Decision Tree":       {"max_depth": [None, 3, 5, 10], "min_samples_split": [2, 5]},
    "Random Forest":       {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "SVM":                 {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 11]},
    "Gradient Boosting":   {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
}


# ─────────────────────── Training Utilities ───────────────────────────────────

def train_model(name: str, X_train, y_train, tune: bool = False):
    """
    Train a single model by name.

    Parameters
    ----------
    name    : str   – key from MODELS dict
    X_train : array
    y_train : array
    tune    : bool  – if True, run GridSearchCV

    Returns
    -------
    Fitted estimator
    """
    model = MODELS[name]
    if tune and name in PARAM_GRIDS:
        gs = GridSearchCV(
            model, PARAM_GRIDS[name],
            cv=5, scoring="accuracy", n_jobs=-1
        )
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, tune: bool = False) -> dict:
    """Train every registered model and return a {name: fitted_model} dict."""
    results = {}
    for name in MODELS:
        results[name] = train_model(name, X_train, y_train, tune=tune)
    return results


def get_best_model(trained_models: dict, X_test, y_test) -> tuple:
    """
    Find the model with highest test accuracy.

    Returns
    -------
    (best_name, best_model)
    """
    from sklearn.metrics import accuracy_score
    best_name, best_model, best_acc = None, None, -1
    for name, model in trained_models.items():
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model
    return best_name, best_model
