"""
utils.py
--------
Shared helper functions used by train_model.py, predict.py, and app.py.

Includes:
- Dataset loading & cleaning
- Column normalization
- Model evaluation metrics
- Saving / loading model artifacts
- Prediction history logging
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config


# ------------------------------------------------------------------
# DATA LOADING & CLEANING
# ------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names using config.COLUMN_ALIASES so the pipeline
    works even if the dataset has slightly different column naming
    (e.g. 'TV Ad Budget ($)' instead of 'TV').
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in config.COLUMN_ALIASES:
            rename_map[col] = config.COLUMN_ALIASES[key]

    df = df.rename(columns=rename_map)

    # Drop stray index columns often found in Kaggle CSVs (e.g. "Unnamed: 0")
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    return df


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Loads the advertising dataset from disk, normalizes columns,
    and returns a cleaned DataFrame.
    """
    path = path or config.DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Please place your 'advertising.csv' file inside the "
            f"'dataset/' folder."
        )

    df = pd.read_csv(path)
    df = normalize_columns(df)

    required_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected columns: {required_cols}. Found: {list(df.columns)}"
        )

    return df[required_cols]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset:
    - Removes duplicate rows
    - Handles missing values (drops rows with NaNs in required columns)
    - Ensures numeric types
    """
    df = df.copy()

    # Ensure numeric types, coercing invalid entries to NaN
    for col in config.FEATURE_COLUMNS + [config.TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows with missing values
    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET_COLUMN])

    after_rows = len(df)
    removed = before_rows - after_rows

    return df, removed


# ------------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------------
def evaluate_model(y_true, y_pred) -> dict:
    """
    Computes standard regression evaluation metrics.
    Returns a dictionary of MAE, MSE, RMSE, and R2 Score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4),
    }


# ------------------------------------------------------------------
# MODEL PERSISTENCE
# ------------------------------------------------------------------
def save_artifact(obj, path: str):
    """Saves any Python object (model, scaler, etc.) to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    """Loads a joblib-saved artifact from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found at '{path}'. Please run 'train_model.py' first."
        )
    return joblib.load(path)


# ------------------------------------------------------------------
# INPUT VALIDATION
# ------------------------------------------------------------------
def validate_inputs(tv: float, radio: float, newspaper: float) -> list:
    """
    Validates prediction inputs against sane ranges.
    Returns a list of error message strings (empty list means valid).
    """
    errors = []
    for name, value in [("TV", tv), ("Radio", radio), ("Newspaper", newspaper)]:
        if value is None:
            errors.append(f"{name} budget is required.")
            continue
        if value < config.INPUT_MIN:
            errors.append(f"{name} budget cannot be negative.")
        if value > config.INPUT_MAX:
            errors.append(
                f"{name} budget seems unrealistically high (> {config.INPUT_MAX})."
            )
    return errors


# ------------------------------------------------------------------
# PREDICTION HISTORY LOGGING
# ------------------------------------------------------------------
def log_prediction(tv, radio, newspaper, predicted_sales, model_name):
    """
    Appends a prediction record to reports/prediction_history.csv.
    Creates the file with headers if it doesn't already exist.
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    record = pd.DataFrame(
        [
            {
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "TV": tv,
                "Radio": radio,
                "Newspaper": newspaper,
                "Predicted_Sales": round(float(predicted_sales), 4),
                "Model_Used": model_name,
            }
        ]
    )

    if os.path.exists(config.PREDICTION_HISTORY_PATH):
        record.to_csv(
            config.PREDICTION_HISTORY_PATH, mode="a", header=False, index=False
        )
    else:
        record.to_csv(config.PREDICTION_HISTORY_PATH, mode="w", header=True, index=False)


def load_prediction_history() -> pd.DataFrame:
    """Loads prediction history CSV if it exists, else returns an empty DataFrame."""
    if os.path.exists(config.PREDICTION_HISTORY_PATH):
        return pd.read_csv(config.PREDICTION_HISTORY_PATH)
    return pd.DataFrame(
        columns=["Timestamp", "TV", "Radio", "Newspaper", "Predicted_Sales", "Model_Used"]
    )
