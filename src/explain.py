
from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.train import (
    load_data,
    TARGET_COLUMN,
    DROP_COLUMNS,
    RANDOM_STATE,
    MODEL_PATH,
)


def main():

    print("Loading dataset...")

    df = load_data()

    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS,
        errors="ignore"
    )

    y = df[TARGET_COLUMN]

    # Recreate the same validation split used during training.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("Loading trained model...")

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]

    print("Calculating permutation importance...")

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False
    )

    output_path = MODEL_PATH.parent / "feature_importance.csv"

    importance_df.to_csv(output_path, index=False)

    print("\nTop 10 Important Features:")

    print(importance_df.head(10).to_string(index=False))

    print(f"\nFeature importance saved to: {output_path}")


if __name__ == "__main__":
    main()
