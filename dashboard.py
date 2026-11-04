import os
import requests
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)
HISTORY_PATH = Path(
    os.getenv(
        "HISTORY_PATH",
        "models/prediction_history.csv"
    )
)


HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_prediction_history(result):

    record = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "churn_probability": result["churn_probability"],
        "predicted_churn": result["predicted_churn"],
        "risk_level": result["risk_level"],
        "model": result["model"]
    }])

    record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not HISTORY_PATH.exists(),
        index=False
    )

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnShield | Customer Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #64748B;
        margin-top: 5px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 10px;
        background-color: #FEE2E2;
        color: #991B1B;
    }

    .risk-medium {
        padding: 18px;
        border-radius: 10px;
        background-color: #FEF3C7;
        color: #92400E;
    }

    .risk-low {
        padding: 18px;
        border-radius: 10px;
        background-color: #DCFCE7;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🛡️ ChurnShield</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">AI-powered Customer Churn Prediction Platform</p>',
    unsafe_allow_html=True,
)

st.write(
    "Analyze customer information and identify customers "
    "who may be at risk of leaving your business."
)

st.divider()

# --------------------------------------------------
# API CONNECTION STATUS
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=3,
    )

    if health_response.ok:
        st.sidebar.success("Prediction API: Connected")
    else:
        st.sidebar.warning("Prediction API: Unavailable")

except requests.exceptions.RequestException:
    st.sidebar.error("Prediction API: Disconnected")

st.sidebar.title("About ChurnShield")

st.sidebar.write(
    """
    ChurnShield uses a trained Machine Learning model
    to estimate customer churn probability.

    **Model:** Logistic Regression

    **Backend:** FastAPI

    **Frontend:** Streamlit
    """
)

# --------------------------------------------------
# CUSTOMER FORM
# --------------------------------------------------

st.markdown(
    '<p class="section-title">Customer Information</p>',
    unsafe_allow_html=True,
)

with st.form("customer_form"):

    st.subheader("Personal Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col3:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=5,
        )

    st.divider()

    st.subheader("Services Information")

    col1, col2 = st.columns(2)

    with col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
        )

    st.divider()

    st.subheader("Contract and Billing Information")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col2:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=85.50,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=427.50,
            step=10.0,
        )

        cltv = st.number_input(
            "Customer Lifetime Value (CLTV)",
            min_value=0.0,
            value=4000.0,
            step=100.0,
        )

    st.divider()

    submitted = st.form_submit_button(
        "🛡️ Predict Customer Churn",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    customer = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv,
    }

    try:

        with st.spinner("Analyzing customer information..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=customer,
                timeout=15,
            )

        if response.ok:

            result = response.json()
            save_prediction_history(result)

            probability = result["churn_probability"]
            predicted_churn = result["predicted_churn"]
            risk = result["risk_level"]

            st.divider()

            st.header("Prediction Results")

            # Metrics
            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Churn Probability",
                f"{probability:.2%}",
            )

            metric2.metric(
                "Predicted Churn",
                "Yes" if predicted_churn == 1 else "No",
            )

            metric3.metric(
                "Customer Risk",
                risk,
            )

            # Probability visualization
            st.subheader("Churn Probability Analysis")

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"Estimated churn probability: {probability:.2%}",
            )

            # Risk explanation
            st.subheader("Risk Assessment")

            if risk == "HIGH":

                st.error(
                    """
                    HIGH RISK

                    This customer has a high estimated probability
                    of leaving the company.

                    Recommended action: Review customer experience,
                    service issues and retention opportunities.
                    """
                )

            elif risk == "MEDIUM":

                st.warning(
                    """
                    MEDIUM RISK

                    This customer has a moderate estimated churn risk.

                    Recommended action: Monitor customer satisfaction
                    and consider proactive engagement.
                    """
                )

            else:

                st.success(
                    """
                    LOW RISK

                    This customer has a relatively low estimated
                    probability of leaving.

                    Recommended action: Continue providing
                    consistent customer support.
                    """
                )

            # Model information
            st.divider()

            st.subheader("Model Information")

            st.write(f"**Selected Model:** {result['model']}")

            st.write(
                f"**Decision Threshold:** "
                f"{result['decision_threshold']:.0%}"
            )

            st.caption(
                "Predictions are estimates based on historical data "
                "and should not be treated as guarantees of future behavior."
            )

        else:

            st.error(
                f"API Error: HTTP {response.status_code}"
            )

            st.code(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please start the backend server."
        )

    except requests.exceptions.Timeout:

        st.error("API request timed out. Please try again.")

    except (ValueError, KeyError) as exc:

        st.error(f"Invalid prediction response: {exc}")

    except requests.exceptions.RequestException as exc:

        st.error(f"Request failed: {exc}")

        # --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()

st.header("Customer Prediction History")

if HISTORY_PATH.exists():

    history = pd.read_csv(HISTORY_PATH)

    st.metric("Total Predictions", len(history))

    st.dataframe(
        history.sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True
    )

    csv_data = history.to_csv(index=False)

    st.download_button(
        "Download Prediction History",
        data=csv_data,
        file_name="churn_prediction_history.csv",
        mime="text/csv"
    )

else:
    st.info("No prediction history available yet.")