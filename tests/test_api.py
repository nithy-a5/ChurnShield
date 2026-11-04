    
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint():
    customer = {
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 5,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.50,
        "Total Charges": 427.50,
        "CLTV": 4000
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "predicted_churn" in result
    assert "risk_level" in result
    assert "decision_threshold" in result
    assert "model" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["predicted_churn"] in [0, 1]
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_prediction_rejects_missing_required_field():
    customer = {
        "Gender": "Female",
        "Senior Citizen": 0
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

