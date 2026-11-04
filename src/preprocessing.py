import pandas as pd

from src.config import TARGET_COLUMN, DROP_COLUMNS


# --------------------------------------------------
# 1. Clean the dataset
# --------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw customer churn dataset."""

    if df.empty:
        raise ValueError("The dataset is empty.")

    df = df.copy()

    # Remove duplicate customer records.
    df = df.drop_duplicates()

    # Convert Total Charges to numeric.
    # Blank strings and invalid values become NaN.
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"].astype(str).str.strip(),
            errors="coerce",
        )

    # Validate the target column.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

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
# 2. Separate features and target
# --------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features (X) from the target (y)."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    columns_to_drop = [TARGET_COLUMN] + DROP_COLUMNS

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    y = df[TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError(
            "No feature columns remain after feature selection."
        )

    return X, y