"""
config.py
----------
Central configuration file for the Sales Prediction project.
Stores paths, constants, and settings used across the project.
"""

import os

# ------------------------------------------------------------------
# BASE DIRECTORY
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# FOLDER PATHS
# ------------------------------------------------------------------
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# ------------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------------
DATASET_PATH = os.path.join(DATASET_DIR, "advertising.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "sales_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_REPORT_PATH = os.path.join(REPORTS_DIR, "model_report.txt")
PREDICTION_HISTORY_PATH = os.path.join(REPORTS_DIR, "prediction_history.csv")

# ------------------------------------------------------------------
# DATASET COLUMNS
# ------------------------------------------------------------------
FEATURE_COLUMNS = ["TV", "Radio", "Newspaper"]
TARGET_COLUMN = "Sales"

# Alternate column names that might appear in slightly different datasets.
# utils.normalize_columns() uses this map to standardize column names.
COLUMN_ALIASES = {
    "tv": "TV",
    "tv ad budget ($)": "TV",
    "radio": "Radio",
    "radio ad budget ($)": "Radio",
    "newspaper": "Newspaper",
    "newspaper ad budget ($)": "Newspaper",
    "sales": "Sales",
    "sales ($)": "Sales",
}

# ------------------------------------------------------------------
# TRAIN / TEST SPLIT SETTINGS
# ------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ------------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------------
MODELS_TO_TRAIN = [
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
]

# ------------------------------------------------------------------
# STREAMLIT APP SETTINGS
# ------------------------------------------------------------------
APP_TITLE = "📈 Sales Prediction Using Machine Learning"
APP_ICON = "📈"
PAGE_LAYOUT = "wide"

# Input validation ranges (used to sanity-check user input on the dashboard)
INPUT_MIN = 0.0
INPUT_MAX = 1000.0
