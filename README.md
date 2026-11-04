# ChurnShield: Customer Churn Prediction

## Overview

ChurnShield is a machine learning project that predicts whether a telecommunications customer is likely to leave a service provider.

The goal is to help businesses identify customers who may be at risk of churning so that they can take proactive retention measures, improve customer satisfaction, and reduce customer loss.

This project covers the end-to-end machine learning workflow, including data cleaning, exploratory analysis, feature preprocessing, model training, model comparison, evaluation, and automated testing.

## Problem Statement

Customer churn is a significant challenge for subscription-based businesses. Acquiring new customers can be more expensive than retaining existing ones.

By analyzing customer demographics, service subscriptions, contract details, tenure, and billing information, ChurnShield aims to identify customers who are likely to discontinue their services.

This project treats churn prediction as a **binary classification problem**:

- `0` — Customer did not churn.
- `1` — Customer churned.

## Objectives

- Analyze customer data to understand churn patterns.
- Clean and preprocess numerical and categorical features.
- Compare multiple machine learning classification algorithms.
- Evaluate model performance using appropriate classification metrics.
- Save the trained model for reuse.
- Build a modular, testable machine learning pipeline.
- Extend the project with a prediction API and web interface.

## Dataset

The project uses a telecommunications customer churn dataset containing customer demographics, account information, subscribed services, billing details, and churn outcomes.

**Target column:** `Churn Value`

The dataset includes features such as:

- Gender, senior-citizen status, partner, and dependents.
- Tenure in months.
- Monthly charges and total charges.
- Contract type and payment method.
- Internet and phone services.
- Churn outcome.

The original dataset contains 7,043 customer records and 33 columns.

The raw dataset is not included in this repository. Obtain the dataset from an authorized source and place `Telco_customer_churn.xlsx` in the project root directory.

## Machine Learning Workflow

1. **Data loading:** Load the Excel dataset using Pandas.
2. **Data cleaning:** Remove duplicate records, convert `Total Charges` to numeric values, and validate the target column.
3. **Feature selection:** Separate input features from the target and exclude identifiers and columns that could introduce target leakage.
4. **Preprocessing:** Impute missing numerical and categorical values, scale numerical features, and one-hot encode categorical features.
5. **Data splitting:** Divide the dataset into training, validation, and test sets using stratified sampling.
6. **Model training:** Train and compare multiple classification algorithms.
7. **Model selection:** Select the best model using validation ROC-AUC.
8. **Threshold selection:** Choose a classification threshold using validation-set F1-score.
9. **Evaluation:** Evaluate the selected model on the held-out test set.
10. **Model persistence:** Save the trained pipeline and evaluation metrics for reuse.

## Models Compared

The training pipeline compares the following algorithms:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Logistic Regression was selected in the current experiment based on validation ROC-AUC.

## Model Performance

The following results were obtained from the held-out test set for the selected Logistic Regression model.

| Metric | Result |
|---|---:|
| Accuracy | 77.86% |
| Precision | 56.60% |
| Recall | 71.12% |
| F1-score | 63.03% |
| ROC-AUC | 84.53% |
| PR-AUC (Average Precision) | 65.16% |

**Classification threshold:** 0.63

### Confusion Matrix

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 831 | 204 |
| Actual: Churn | 108 | 266 |

### Interpretation

- The model identified 266 of the 374 customers who actually churned.
- It detected approximately 71% of actual churners.
- Of the customers predicted to churn, approximately 57% actually churned.

Recall is particularly relevant when the business wants to identify as many potential churners as possible. However, the appropriate precision-recall trade-off depends on the cost of customer-retention campaigns.

These results are an initial baseline, not a guarantee of performance on future customer data.

## Project Structure

```text
churnshield/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_churn_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/
│   └── app.py
├── models/
│   └── .gitkeep
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

*Note: The prediction module, API, and Docker deployment should be described as completed features only after they have been implemented and tested.*

## Technologies Used

- **Programming language:** Python
- **Data manipulation:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Data visualization:** Matplotlib, Seaborn (for exploratory analysis, if used)
- **Testing:** Pytest
- **Development tools:** Jupyter Notebook and an IDE

Planned extensions include FastAPI, Docker-based deployment, and an interactive web interface.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd churnshield
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place `Telco_customer_churn.xlsx` in the project root directory, alongside `README.md`.

### 5. Train the models

```bash
python -m src.train
```

This command trains and compares the models, selects a model and threshold, and saves the model artifact and training metrics under `models/`.

### 6. Evaluate the saved model

```bash
python -m src.evaluate
```

This command loads the saved model, evaluates it on the reproducibly reconstructed test split, and writes `evaluation_report.json`.

### 7. Run automated tests

```bash
python -m pytest tests/ -v
```

Ensure the dataset and required model artifacts exist before running tests that depend on them.

## Generated Artifacts

After successful training and evaluation, the project generates:

- `models/churn_model.joblib` — trained model pipeline and associated prediction settings.
- `models/metrics.json` — model comparison results and final test metrics.
- `models/evaluation_report.json` — evaluation results from the saved model.

These files are generated by the training and evaluation scripts and may not be present in a fresh clone.

## Future Improvements

- [ ] Complete and test a reusable prediction module.
- [ ] Build a REST API using FastAPI.
- [ ] Develop an interactive customer churn prediction interface.
- [ ] Add model performance visualizations and a dashboard.
- [ ] Add API and model integration tests.
- [ ] Containerize the application with Docker.
- [ ] Add continuous integration for automated testing.
- [ ] Monitor model performance and data drift.
- [ ] Tune model hyperparameters and evaluate alternative classification thresholds.

## Limitations

- Model performance depends on the quality and representativeness of the available dataset.
- The current results are based on a single held-out test split.
- Classification thresholds should be chosen according to business costs and retention objectives.
- Model performance should be validated on newer customer data before real-world deployment.

## Author

Developed as a machine learning portfolio project to demonstrate practical skills in Python, data preprocessing, supervised learning, model evaluation, and software engineering.

