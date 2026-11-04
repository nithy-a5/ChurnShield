from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.train import load_data, DATA_PATH, TARGET_COLUMN, DROP_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"


# --------------------------------------------------
# Test 1: Dataset file exists
# --------------------------------------------------

def test_dataset_exists():
    assert DATA_PATH.exists(), (
        f"Dataset not found at {DATA_PATH}"
    )


# --------------------------------------------------
# Test 2: Dataset loads correctly
# --------------------------------------------------

def test_load_data():
    df = load_data()

    assert not df.empty
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].isin([0, 1]).all()


# --------------------------------------------------
# Test 3: Dataset has expected columns
# --------------------------------------------------

def test_required_columns_exist():
    df = load_data()

    required_columns = [
        "CustomerID",
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Churn Value",
    ]

    for column in required_columns:
        assert column in df.columns, (
            f"Missing required column: {column}"
        )


# --------------------------------------------------
# Test 4: Target contains both classes
# --------------------------------------------------

def test_target_contains_both_classes():
    df = load_data()

    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


# --------------------------------------------------
# Test 5: Saved model exists
# --------------------------------------------------

def test_saved_model_exists():
    assert MODEL_PATH.exists(), (
        "Trained model not found. Run "
        "'python -m src.train' first."
    )


# --------------------------------------------------
# Test 6: Model artifact has expected structure
# --------------------------------------------------

def test_saved_model_artifact():
    assert MODEL_PATH.exists(), (
        "Run 'python -m src.train' first."
    )

    artifact = joblib.load(MODEL_PATH)

    assert "model" in artifact
    assert "threshold" in artifact
    assert "feature_columns" in artifact
    assert "selected_model" in artifact

    assert 0.0 <= artifact["threshold"] <= 1.0


# --------------------------------------------------
# Test 7: Model makes valid predictions
# --------------------------------------------------

def test_model_predictions():
    assert MODEL_PATH.exists(), (
        "Run 'python -m src.train' first."
    )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = load_data()

    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS,
        errors="ignore",
    )

    # Use a small sample with the same features as training.
    sample = X[feature_columns].head(5)

    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)[:, 1]

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})
    assert len(probabilities) == 5
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


# --------------------------------------------------
# Test 8: Model handles an unseen customer category
# --------------------------------------------------

def test_model_handles_unseen_category():
    assert MODEL_PATH.exists(), (
        "Run 'python -m src.train' first."
    )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = load_data()
    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS,
        errors="ignore",
    )

    sample = X[feature_columns].head(1).copy()

    # Find a categorical feature and supply an unseen value.
    categorical_columns = sample.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    if len(categorical_columns) == 0:
        pytest.skip("No categorical features available.")

    column = categorical_columns[0]
    sample[column] = "__UNSEEN_TEST_CATEGORY__"

    probabilities = model.predict_proba(sample)[:, 1]

    assert len(probabilities) == 1
    assert 0.0 <= probabilities[0] <= 1.0