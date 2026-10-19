"""
predict.py
----------
Command-line script to make sales predictions using the trained model.

Usage:
    python predict.py --tv 200 --radio 35 --newspaper 50

If run without arguments, it will prompt for input interactively.
"""

import argparse
import sys
import pandas as pd

import config
import utils


def load_model_and_scaler():
    """Loads the trained model and scaler from disk."""
    try:
        model = utils.load_artifact(config.MODEL_PATH)
        scaler = utils.load_artifact(config.SCALER_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    model_name = "Trained Model"
    try:
        model_name = utils.load_artifact(config.MODEL_PATH.replace(".pkl", "_name.pkl"))
    except FileNotFoundError:
        pass

    return model, scaler, model_name


def predict_sales(model, scaler, tv: float, radio: float, newspaper: float) -> float:
    """Runs a single prediction given TV, Radio, and Newspaper budgets."""
    X = pd.DataFrame([[tv, radio, newspaper]], columns=config.FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return prediction


def interactive_mode(model, scaler, model_name):
    """Prompts the user for input values interactively."""
    print("\n📈 Sales Prediction - Interactive Mode")
    print("-" * 40)
    try:
        tv = float(input("Enter TV advertising budget: "))
        radio = float(input("Enter Radio advertising budget: "))
        newspaper = float(input("Enter Newspaper advertising budget: "))
    except ValueError:
        print("\n❌ ERROR: Please enter valid numeric values.")
        sys.exit(1)

    run_prediction(model, scaler, model_name, tv, radio, newspaper)


def run_prediction(model, scaler, model_name, tv, radio, newspaper):
    """Validates input, runs prediction, prints result, and logs history."""
    errors = utils.validate_inputs(tv, radio, newspaper)
    if errors:
        print("\n❌ Invalid input:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    prediction = predict_sales(model, scaler, tv, radio, newspaper)

    print("\n" + "=" * 40)
    print(" PREDICTION RESULT")
    print("=" * 40)
    print(f" Model Used     : {model_name}")
    print(f" TV Budget      : {tv}")
    print(f" Radio Budget   : {radio}")
    print(f" Newspaper      : {newspaper}")
    print("-" * 40)
    print(f" Predicted Sales: {prediction:.2f} Units")
    print("=" * 40)

    utils.log_prediction(tv, radio, newspaper, prediction, model_name)
    print(f"\n(Logged to {config.PREDICTION_HISTORY_PATH})")


def main():
    parser = argparse.ArgumentParser(
        description="Predict product sales from advertising budgets."
    )
    parser.add_argument("--tv", type=float, help="TV advertising budget")
    parser.add_argument("--radio", type=float, help="Radio advertising budget")
    parser.add_argument("--newspaper", type=float, help="Newspaper advertising budget")

    args = parser.parse_args()

    model, scaler, model_name = load_model_and_scaler()

    if args.tv is None or args.radio is None or args.newspaper is None:
        interactive_mode(model, scaler, model_name)
    else:
        run_prediction(model, scaler, model_name, args.tv, args.radio, args.newspaper)


if __name__ == "__main__":
    main()
