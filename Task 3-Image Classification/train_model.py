"""
train_model.py
──────────────
Standalone script to train all models, pick the best one, and persist
the model + scaler + label_encoder to models/.

Usage:
    python train_model.py              # train all, no hyperparameter tuning
    python train_model.py --tune       # train with GridSearchCV tuning
"""

import argparse
import sys
import os

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_pipeline
from src.model import train_all_models, get_best_model
from src.evaluation import all_model_metrics
from src.utils import save_model, save_artifact, save_metrics_report, logger


def main(tune: bool = False):
    logger.info("=" * 55)
    logger.info("  Iris Flower Classification — Training Pipeline")
    logger.info("=" * 55)

    # ── 1. Preprocess ──────────────────────────────────────────
    logger.info("Loading and preprocessing data …")
    X_train, X_test, y_train, y_test, scaler, le, df = preprocess_pipeline()
    logger.info(f"  Train samples : {len(X_train)}")
    logger.info(f"  Test  samples : {len(X_test)}")
    logger.info(f"  Classes       : {list(le.classes_)}")

    # ── 2. Train ───────────────────────────────────────────────
    logger.info(f"Training all models (tune={tune}) …")
    trained_models = train_all_models(X_train, y_train, tune=tune)

    # ── 3. Evaluate ────────────────────────────────────────────
    logger.info("Evaluating models …")
    metrics_df = all_model_metrics(trained_models, X_test, y_test, le)
    print("\n" + metrics_df.to_string(index=False))

    # ── 4. Best model ──────────────────────────────────────────
    best_name, best_model = get_best_model(trained_models, X_test, y_test)
    best_acc = metrics_df.loc[metrics_df["Model"] == best_name, "Accuracy"].values[0]
    logger.info(f"\n🏆 Best model: {best_name}  (Accuracy={best_acc:.4f})")

    # ── 5. Persist ─────────────────────────────────────────────
    logger.info("Saving artefacts …")
    save_model(best_model, "best_model.pkl")
    save_artifact(scaler,   "scaler.pkl")
    save_artifact(le,       "label_encoder.pkl")

    # Save all trained models
    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        save_model(model, filename)

    # Save metrics report
    save_metrics_report(metrics_df, best_name)

    logger.info("Training complete! ✅")
    return trained_models, best_name, metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris classification models")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV hyperparameter tuning")
    args = parser.parse_args()
    main(tune=args.tune)
