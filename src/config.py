from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Telco_customer_churn.xlsx"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

# Metrics output
METRICS_PATH = MODEL_DIR / "metrics.json"

# Target column
TARGET_COLUMN = "Churn Value"

# Columns excluded from model features
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

# Reproducibility
RANDOM_STATE = 42

# Test dataset proportion
TEST_SIZE = 0.2

# Validation dataset proportion
VALIDATION_SIZE = 0.2
