# Dataset Documentation

## Overview

ChurnShield uses a telecommunications customer churn dataset to build a binary classification model that predicts whether a customer is likely to leave a service provider.

The dataset contains customer demographics, subscribed services, account information, billing details, and churn outcomes.

## Dataset File

**Expected filename:** `Telco_customer_churn.xlsx`

**Expected location:** Project root directory, one level above the `data/` folder.

```text
churnshield/
├── data/
│   └── README.md
├── Telco_customer_churn.xlsx
└── src/
    └── config.py
```

The dataset is stored outside the `data/` folder because the current project configuration loads it from the project root.

## Dataset Source

Obtain the dataset from an authorized source and record the original source URL here:

- Source URL: Add the verified dataset URL.
- File format: Excel (`.xlsx`)
- Number of records: 7,043
- Number of columns: 33

Check the dataset's license and usage terms before redistributing it.

## Target Variable

The target column is `Churn Value`.

| Value | Meaning |
|---|---|
| `0` | Customer did not churn |
| `1` | Customer churned |

The model uses this column as the label it learns to predict.

## Data Preprocessing

The project applies the following preprocessing steps:

- Remove duplicate records.
- Convert `Total Charges` to a numeric data type.
- Validate and convert the target variable.
- Separate input features from the target.
- Exclude identifiers and selected columns that could introduce target leakage.
- Impute missing feature values.
- Scale numerical features.
- One-hot encode categorical features.

Preprocessing and feature transformation are implemented in `src/preprocessing.py` and the model pipeline in `src/train.py`.

## Data Handling

- The raw dataset is excluded from version control through `.gitignore`.
- Do not commit private, confidential, or personally identifiable customer information.
- Keep the original dataset unchanged; perform cleaning and transformations in code.
- Document any future changes to the dataset or its source.

## Reproducing the Project

1. Obtain the dataset from the documented source.
2. Place `Telco_customer_churn.xlsx` in the project root directory.
3. Install the dependencies listed in `requirements.txt`.
4. Run `python -m src.train` from the project root.

For the complete setup instructions, refer to the main project `README.md`.

