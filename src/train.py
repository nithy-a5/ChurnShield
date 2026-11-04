from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Your Excel dataset is currently in the project root.
DATA_PATH = ROOT / "data" / "Telco_customer_churn.xlsx"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "Churn Value"
RANDOM_STATE = 42


# Columns that should not be used as model features.
DROP_COLUMNS = [
    "CustomerID",
    "Churn Label",
    "Churn Reason",
    "Churn Score",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]


# --------------------------------------------------
# 2. Load and clean the dataset
# --------------------------------------------------

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Excel dataset and perform basic cleaning."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Check the file location and filename."
        )

    df = pd.read_excel(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert blank or invalid Total Charges values to NaN.
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"].astype(str).str.strip(),
            errors="coerce",
        )

    # The target must contain valid binary labels.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            f"'{TARGET_COLUMN}' must contain only 0 and 1."
        )

    return df


# --------------------------------------------------
# 3. Build the preprocessing pipeline
# --------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing steps for numeric and categorical columns."""

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor


# --------------------------------------------------
# 4. Create complete ML pipelines
# --------------------------------------------------

def build_models(X_train: pd.DataFrame) -> dict:
    """Create models with preprocessing included in each pipeline."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    pipelines = {}

    for name, model in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", model),
            ]
        )

    return pipelines


# --------------------------------------------------
# 5. Evaluate a model at a chosen threshold
# --------------------------------------------------

def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Calculate classification metrics and confusion matrix."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions
        ).tolist(),
    }


# --------------------------------------------------
# 6. Train, select, and test models
# --------------------------------------------------

def main():
    print("Loading dataset...")
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    # Separate features and target.
    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS,
        errors="ignore",
    )
    y = df[TARGET_COLUMN]

    if X.empty:
        raise ValueError("No feature columns remain after dropping columns.")

    # Split into 60% training, 20% validation, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"\nTraining rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows:       {len(X_test)}")

    # Train each candidate model using training data only.
    pipelines = build_models(X_train)
    validation_results = {}

    print("\nTraining and comparing models...")

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_val)[:, 1]

        validation_results[name] = {
            "model": pipeline,
            "roc_auc": roc_auc_score(y_val, probabilities),
            "pr_auc": average_precision_score(y_val, probabilities),
        }

        print(
            f"Validation ROC-AUC: "
            f"{validation_results[name]['roc_auc']:.4f}"
        )
        print(
            f"Validation PR-AUC: "
            f"{validation_results[name]['pr_auc']:.4f}"
        )

    # Select the model using validation ROC-AUC, not test results.
    best_name = max(
        validation_results,
        key=lambda name: validation_results[name]["roc_auc"],
    )

    best_model = validation_results[best_name]["model"]

    print(f"\nSelected model: {best_name}")

    # Choose a probability threshold using validation data only.
    val_probabilities = best_model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.20, 0.81, 0.01)

    threshold_scores = [
        f1_score(
            y_val,
            (val_probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_threshold = float(thresholds[int(np.argmax(threshold_scores))])

    print(f"Selected threshold: {best_threshold:.2f}")

    # Evaluate once on the held-out test set.
    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        threshold=best_threshold,
    )

    print("\nFinal test results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    # Save the fitted pipeline and its chosen threshold together.
    artifact = {
        "model": best_model,
        "threshold": best_threshold,
        "target_column": TARGET_COLUMN,
        "feature_columns": X.columns.tolist(),
        "selected_model": best_name,
    }

    joblib.dump(artifact, MODEL_PATH)

    # Save model comparison and final test metrics.
    metrics_output = {
        "selected_model": best_name,
        "validation_results": {
            name: {
                "roc_auc": float(result["roc_auc"]),
                "pr_auc": float(result["pr_auc"]),
            }
            for name, result in validation_results.items()
        },
        "test_metrics": test_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=4)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()