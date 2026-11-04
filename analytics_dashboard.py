from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco_customer_churn.xlsx"

st.set_page_config(
    page_title="ChurnShield Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ChurnShield Analytics")
st.caption("Explore customer churn patterns and retention opportunities.")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    df = pd.read_excel(DATA_PATH)

    required_columns = [
        "Churn Value",
        "Contract",
        "Tenure Months",
        "Monthly Charges",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Churn Value"] = pd.to_numeric(
        df["Churn Value"], errors="coerce"
    )
    df["Tenure Months"] = pd.to_numeric(
        df["Tenure Months"], errors="coerce"
    )
    df["Monthly Charges"] = pd.to_numeric(
        df["Monthly Charges"], errors="coerce"
    )

    df = df.dropna(subset=["Churn Value"])
    df = df[df["Churn Value"].isin([0, 1])]

    return df


try:
    df = load_data()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# Sidebar filters
st.sidebar.header("Dashboard Filters")

if "Contract" in df.columns:
    contracts = sorted(df["Contract"].dropna().unique().tolist())
    selected_contracts = st.sidebar.multiselect(
        "Contract type",
        options=contracts,
        default=contracts,
    )
    filtered_df = df[df["Contract"].isin(selected_contracts)].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()


# Summary metrics
total_customers = len(filtered_df)
churned_customers = int(
    (filtered_df["Churn Value"] == 1).sum()
)
retained_customers = int(
    (filtered_df["Churn Value"] == 0).sum()
)
churn_rate = (
    churned_customers / total_customers * 100
    if total_customers
    else 0
)

st.subheader("Customer Overview")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Total Customers", f"{total_customers:,}")
metric2.metric("Churned Customers", f"{churned_customers:,}")
metric3.metric("Retained Customers", f"{retained_customers:,}")
metric4.metric("Churn Rate", f"{churn_rate:.1f}%")


# Churn distribution
st.divider()
st.subheader("Churn Distribution")

distribution = (
    filtered_df["Churn Value"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)
distribution.index = ["Retained", "Churned"]
distribution_df = distribution.rename("Customers").reset_index()
distribution_df.columns = ["Customer Status", "Customers"]

st.bar_chart(
    distribution_df,
    x="Customer Status",
    y="Customers",
)


# Churn by contract
st.divider()
st.subheader("Churn Rate by Contract Type")

if "Contract" in filtered_df.columns:
    contract_summary = (
        filtered_df.groupby("Contract", dropna=False)["Churn Value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    contract_summary["Churn Rate (%)"] = contract_summary["mean"] * 100
    contract_summary = contract_summary.rename(
        columns={"count": "Customers"}
    )

    st.bar_chart(
        contract_summary,
        x="Contract",
        y="Churn Rate (%)",
    )

    st.dataframe(
        contract_summary[
            ["Contract", "Customers", "Churn Rate (%)"]
        ].assign(
            **{"Churn Rate (%)": lambda x: x["Churn Rate (%)"].round(1)}
        ),
        use_container_width=True,
        hide_index=True,
    )


# Churn by tenure
st.divider()
st.subheader("Churn Rate by Customer Tenure")

tenure_df = filtered_df.dropna(subset=["Tenure Months"]).copy()

if not tenure_df.empty:
    tenure_df["Tenure Group"] = pd.cut(
        tenure_df["Tenure Months"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49–60 months",
            "60+ months",
        ],
    )

    tenure_summary = (
        tenure_df.groupby("Tenure Group", observed=False)["Churn Value"]
        .mean()
        .mul(100)
        .rename("Churn Rate (%)")
        .reset_index()
    )

    st.bar_chart(
        tenure_summary,
        x="Tenure Group",
        y="Churn Rate (%)",
    )


# Churn by monthly charges
st.divider()
st.subheader("Churn by Monthly Charges")

charges_df = filtered_df.dropna(subset=["Monthly Charges"]).copy()

if not charges_df.empty:
    charges_df["Monthly Charge Band"] = pd.cut(
        charges_df["Monthly Charges"],
        bins=[-0.01, 30, 60, 90, 120, float("inf")],
        labels=[
            "Up to 30",
            "30–60",
            "60–90",
            "90–120",
            "120+",
        ],
    )

    charges_summary = (
        charges_df.groupby(
            "Monthly Charge Band", observed=False
        )["Churn Value"]
        .mean()
        .mul(100)
        .rename("Churn Rate (%)")
        .reset_index()
    )

    st.bar_chart(
        charges_summary,
        x="Monthly Charge Band",
        y="Churn Rate (%)",
    )


# Data preview
with st.expander("View customer data"):
    st.dataframe(filtered_df.head(100), use_container_width=True)

st.caption(
    "Exploratory analysis of the supplied dataset. "
    "Associations shown in these charts do not establish causation."
)
