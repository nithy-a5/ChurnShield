from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH


class ChurnPredictor:
    """Load the trained churn model and predict customer churn risk."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. "
                "Run 'python -m src.train' first."
            )

        # The saved artifact contains the pipeline and its metadata.
        artifact = joblib.load(MODEL_PATH)

        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.feature_columns = artifact["feature_columns"]
        self.selected_model = artifact["selected_model"]

    def predict(self, customer: dict) -> dict:
        """Return churn probability, prediction, and risk level."""

        # Build a DataFrame using the exact columns used during training.
        df = pd.DataFrame([customer])
        df = df.reindex(columns=self.feature_columns)

        # Calculate the probability of churn (class 1).
        probability = float(self.model.predict_proba(df)[0][1])

        # Use the threshold selected during model training.
        predicted_churn = int(probability >= self.threshold)

        if probability >= 0.70:
            risk = "HIGH"
        elif probability >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "churn_probability": round(probability, 4),
            "predicted_churn": predicted_churn,
            "risk_level": risk,
            "decision_threshold": round(self.threshold, 4),
            "model": self.selected_model,
        }


if __name__ == "__main__":
    predictor = ChurnPredictor()

    # Use the exact feature names from the training dataset.
    customer = {
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 5,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.50,
        "Total Charges": 427.50,
        "CLTV": 4000,
    }

    print(predictor.predict(customer))
