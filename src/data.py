
import pandas as pd

from src.config import RAW_DATA_PATH


def load_data(file_path=RAW_DATA_PATH):
    """Load the raw customer churn dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    df = pd.read_excel(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df
