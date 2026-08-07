"""
train_model.py
---------------
Trains multiple regression models (Linear Regression, Decision Tree,
Random Forest) on the advertising dataset, evaluates them, selects the
best-performing model, and saves the trained model + scaler + report.

Run:
    python train_model.py
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import config
import utils


def get_models():
    """Returns a dictionary of model name -> untrained model instance."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=config.RANDOM_STATE
        ),
    }


def main():
    print("=" * 60)
    print(" SALES PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------
    print("\n[1/6] Loading dataset...")
    try:
        df = utils.load_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    print(f"    Loaded {len(df)} rows.")

    # ----------------------------------------------------------
    # 2. Clean dataset
    # ----------------------------------------------------------
    print("\n[2/6] Cleaning dataset (duplicates & missing values)...")
    df, removed = utils.clean_dataset(df)
    print(f"    Removed {removed} row(s). {len(df)} rows remain.")

    if len(df) < 10:
        print("\n❌ ERROR: Not enough data to train a reliable model "
              "(need at least 10 rows).")
        sys.exit(1)

    # ----------------------------------------------------------
    # 3. Train / test split
    # ----------------------------------------------------------
    print("\n[3/6] Splitting into train/test sets...")
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ----------------------------------------------------------
    # 4. Feature scaling
    # ----------------------------------------------------------
    print("\n[4/6] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------------------------
    # 5. Train & evaluate multiple models
    # ----------------------------------------------------------
    print("\n[5/6] Training & evaluating models...\n")
    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = utils.evaluate_model(y_test, y_pred)
        results[name] = metrics
        trained_models[name] = model

        print(f"  🔹 {name}")
        for metric_name, value in metrics.items():
            print(f"       {metric_name:10s}: {value}")
        print()

    # ----------------------------------------------------------
    # 6. Select best model (highest R2 Score) and save artifacts
    # ----------------------------------------------------------
    best_model_name = max(results, key=lambda k: results[k]["R2 Score"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    print("=" * 60)
    print(f" 🏆 BEST MODEL: {best_model_name}")
    print(f"    R2 Score: {best_metrics['R2 Score']}")
    print("=" * 60)

    print("\n[6/6] Saving model, scaler, and report...")
    utils.save_artifact(best_model, config.MODEL_PATH)
    utils.save_artifact(scaler, config.SCALER_PATH)

    # Also save which model type won, so predict.py / app.py can display it
    utils.save_artifact(best_model_name, config.MODEL_PATH.replace(".pkl", "_name.pkl"))

    write_report(results, best_model_name, len(df), removed)

    print(f"    ✅ Model saved to:  {config.MODEL_PATH}")
    print(f"    ✅ Scaler saved to: {config.SCALER_PATH}")
    print(f"    ✅ Report saved to: {config.MODEL_REPORT_PATH}")
    print("\nDone! You can now run the app with:  streamlit run app.py\n")


def write_report(results: dict, best_model_name: str, n_rows: int, n_removed: int):
    """Writes a plain-text model comparison report to reports/model_report.txt."""
    import os

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("SALES PREDICTION - MODEL TRAINING REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset rows used for training: {n_rows}")
    lines.append(f"Rows removed during cleaning: {n_removed}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("MODEL COMPARISON")
    lines.append("-" * 60)

    for name, metrics in results.items():
        marker = " <-- BEST MODEL" if name == best_model_name else ""
        lines.append(f"\n{name}{marker}")
        for metric_name, value in metrics.items():
            lines.append(f"   {metric_name:10s}: {value}")

    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Selected Model: {best_model_name}")
    lines.append(f"Selection Criteria: Highest R2 Score")
    lines.append("-" * 60)

    with open(config.MODEL_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
