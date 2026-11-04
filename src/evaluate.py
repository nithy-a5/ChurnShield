
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data import load_data
from src.preprocessing import clean_data, split_features_target


def evaluate_saved_model():
    """Evaluate the saved model on the held-out test split."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )

    print("Loading dataset...")
    df = clean_data(load_data())

    X, y = split_features_target(df)

    # Recreate the same 60/20/20 split used by train.py.
    _, X_temp, _, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    _, X_test, _, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("Loading saved model...")
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]

    # Keep the same feature order used during training.
    X_test = X_test[feature_columns]

    # Generate predictions and probabilities.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Calculate metrics.
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    print("\nSaved model evaluation")
    print(f"Model: {artifact['selected_model']}")
    print(f"Test samples: {len(X_test)}")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Save a separate evaluation report.
    report_path = MODEL_PATH.parent / "evaluation_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_saved_model()
