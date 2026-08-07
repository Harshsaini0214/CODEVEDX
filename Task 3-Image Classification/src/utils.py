# utils module
"""
Shared utilities: model persistence, logging, and helper functions.
"""

import os
import joblib
import logging
from datetime import datetime
import json

# ─────────────────────────── Paths ────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IrisApp")


# ─────────────────────────── Persistence ──────────────────────────────────────

def save_model(model, filename: str = "best_model.pkl") -> str:
    """Serialize a model to the models/ directory."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")
    return path


def load_model(filename: str = "best_model.pkl"):
    """Load a serialized model from models/."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def save_artifact(obj, filename: str) -> str:
    """Generic joblib serialization for scalers, encoders, etc."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    return path


def load_artifact(filename: str):
    """Load a generic artifact from models/."""
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)


def save_metrics_report(metrics_df, best_model_name: str) -> str:
    """Save a JSON metrics report to reports/."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    path = os.path.join(REPORTS_DIR, "metrics_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved → {path}")
    return path


def model_exists(filename: str = "best_model.pkl") -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, filename))


# ─────────────────────────── Misc Helpers ─────────────────────────────────────

def species_info() -> dict:
    """Return descriptive info for each Iris species."""
    return {
        "Iris-setosa": {
            "description": "Iris setosa is a small species found in arctic and subarctic regions. "
                           "It has very short petals and is the easiest to classify.",
            "color": "#6C63FF",
            "emoji": "🌸",
        },
        "Iris-versicolor": {
            "description": "Iris versicolor, the blue flag iris, is native to eastern North America. "
                           "It has medium-sized petals and overlaps with virginica.",
            "color": "#FF6584",
            "emoji": "🌺",
        },
        "Iris-virginica": {
            "description": "Iris virginica is native to eastern North America and has the largest petals "
                           "of the three species. It closely resembles versicolor.",
            "color": "#43E97B",
            "emoji": "🌿",
        },
    }


def format_confidence(proba: float) -> str:
    """Return a human-readable confidence string with color tier."""
    if proba >= 0.90:
        return f"✅ Very High ({proba:.1%})"
    elif proba >= 0.70:
        return f"🟡 High ({proba:.1%})"
    elif proba >= 0.50:
        return f"🟠 Moderate ({proba:.1%})"
    else:
        return f"🔴 Low ({proba:.1%})"
