# preprocessing module
"""
Data loading, cleaning, encoding, and splitting utilities for the Iris dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# ─────────────────────────── Constants ────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "IRIS.csv")
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
RANDOM_STATE = 42


# ─────────────────────────── Functions ────────────────────────────────────────

def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the Iris CSV dataset and return a clean DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df[TARGET_COLUMN].values
    return X, y


def encode_labels(y):
    """Encode string labels to integers. Returns encoded y and the fitted encoder."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def split_data(X, y, test_size: float = 0.2):
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )


def scale_features(X_train, X_test):
    """Fit StandardScaler on train set and transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(path: str = DATASET_PATH, test_size: float = 0.2):
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    scaler                            : fitted StandardScaler
    label_encoder                     : fitted LabelEncoder
    df                                : raw DataFrame
    """
    df = load_data(path)
    X, y = get_feature_matrix(df)
    y_enc, le = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X, y_enc, test_size)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, le, df
