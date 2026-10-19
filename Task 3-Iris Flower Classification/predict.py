"""
predict.py
──────────
Standalone CLI prediction script.

Usage:
    python predict.py --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
    python predict.py --interactive
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_model, load_artifact, format_confidence, logger
from src.preprocessing import FEATURE_COLUMNS


def predict_single(sepal_length: float, sepal_width: float,
                   petal_length: float, petal_width: float) -> dict:
    """
    Run inference on a single sample.

    Returns
    -------
    dict with keys: species, confidence, probabilities
    """
    model = load_model("best_model.pkl")
    scaler = load_artifact("scaler.pkl")
    le     = load_artifact("label_encoder.pkl")

    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X)

    pred_idx = model.predict(X_scaled)[0]
    species  = le.inverse_transform([pred_idx])[0]

    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
    confidence = float(proba[pred_idx]) if proba is not None else None

    return {
        "species":       species,
        "confidence":    confidence,
        "probabilities": dict(zip(le.classes_, proba.tolist())) if proba is not None else {},
    }


def interactive_mode():
    print("\n🌸 Iris Flower Classifier — Interactive Mode")
    print("─" * 45)
    print("Enter measurements in centimetres (press Ctrl+C to quit).\n")
    while True:
        try:
            sl = float(input("  Sepal Length (cm): "))
            sw = float(input("  Sepal Width  (cm): "))
            pl = float(input("  Petal Length (cm): "))
            pw = float(input("  Petal Width  (cm): "))
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except ValueError:
            print("  ⚠ Please enter a valid number.\n")
            continue

        result = predict_single(sl, sw, pl, pw)
        print(f"\n  Predicted Species : {result['species']}")
        if result["confidence"] is not None:
            print(f"  Confidence        : {format_confidence(result['confidence'])}")
            print("\n  Class Probabilities:")
            for cls, prob in result["probabilities"].items():
                bar = "█" * int(prob * 30)
                print(f"    {cls:<22} {prob:.3f}  {bar}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Iris Flower Predictor")
    parser.add_argument("--sl", type=float, help="Sepal Length (cm)")
    parser.add_argument("--sw", type=float, help="Sepal Width  (cm)")
    parser.add_argument("--pl", type=float, help="Petal Length (cm)")
    parser.add_argument("--pw", type=float, help="Petal Width  (cm)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive prompt")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if None in (args.sl, args.sw, args.pl, args.pw):
        parser.error("Provide --sl, --sw, --pl, --pw (or use --interactive).")

    result = predict_single(args.sl, args.sw, args.pl, args.pw)
    print(f"\nPredicted Species : {result['species']}")
    if result["confidence"] is not None:
        print(f"Confidence        : {format_confidence(result['confidence'])}")
        print("\nClass Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"  {cls:<22} {prob:.4f}")


if __name__ == "__main__":
    main()
