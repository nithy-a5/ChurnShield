
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

from src.train import (
    load_data,
    TARGET_COLUMN,
    DROP_COLUMNS,
    RANDOM_STATE,
    MODEL_PATH,
)

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"

st.set_page_config(
    page_title="ChurnShield Model Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ChurnShield - Model Evaluation Dashboard")
st.caption("Machine Learning Performance and Explainability")

# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    artifact = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    importance = pd.read_csv(IMPORTANCE_PATH)

    return artifact, metrics, importance


try:
    artifact, metrics, importance = load_artifacts()

    model = artifact["model"]
    threshold = artifact["threshold"]

except Exception as exc:
    st.error(f"Could not load project artifacts: {exc}")
    st.stop()


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Model", metrics["selected_model"])
col2.metric("Decision Threshold", f"{threshold:.2f}")
col3.metric("Important Features", len(importance))

# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

st.divider()
st.subheader("Model Performance")

test_metrics = metrics["test_metrics"]

metric_columns = st.columns(5)

display_metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1_score"),
    ("ROC-AUC", "roc_auc"),
]

for column, (label, key) in zip(metric_columns, display_metrics):
    column.metric(label, f"{test_metrics[key]:.2%}")


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("Model Comparison - Validation Results")

comparison = pd.DataFrame(
    metrics["validation_results"]
).T

st.dataframe(
    comparison.style.format("{:.4f}"),
    use_container_width=True,
)

st.bar_chart(comparison[["roc_auc", "pr_auc"]])


# --------------------------------------------------
# RECREATE TEST DATA
# --------------------------------------------------

df = load_data()

X = df.drop(
    columns=[TARGET_COLUMN] + DROP_COLUMNS,
    errors="ignore",
)

y = df[TARGET_COLUMN]

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

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.divider()
st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["No Churn", "Churn"],
    cmap="Blues",
    ax=ax,
)

ax.set_title("Customer Churn Confusion Matrix")
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

st.subheader("ROC Curve")

fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}",
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# PRECISION RECALL CURVE
# --------------------------------------------------

st.subheader("Precision-Recall Curve")

precision, recall, _ = precision_recall_curve(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(y_test, probabilities)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    recall,
    precision,
    label=f"PR-AUC = {pr_auc:.3f}",
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()
st.subheader("Feature Importance Analysis")

top_features = importance.head(10).sort_values(
    "importance_mean",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"],
)

ax.set_xlabel("Permutation Importance (ROC-AUC decrease)")
ax.set_title("Top 10 Important Features")
ax.grid(axis="x", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    importance,
    use_container_width=True,
)

st.caption(
    "Feature importance represents predictive contribution, "
    "not proof of causation."
)

st.success("Model evaluation dashboard loaded successfully.")
