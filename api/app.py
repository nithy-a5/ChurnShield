from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.predict import ChurnPredictor


app = FastAPI(
    title="ChurnShield API",
    description="Customer churn prediction service",
    version="1.0.0",
)

predictor = ChurnPredictor()


class Customer(BaseModel):
    Gender: str
    Senior_Citizen: int = Field(alias="Senior Citizen", ge=0, le=1)
    Partner: str
    Dependents: str
    Tenure_Months: int = Field(alias="Tenure Months", ge=0)
    Phone_Service: str = Field(alias="Phone Service")
    Multiple_Lines: str = Field(alias="Multiple Lines")
    Internet_Service: str = Field(alias="Internet Service")
    Online_Security: str = Field(alias="Online Security")
    Online_Backup: str = Field(alias="Online Backup")
    Device_Protection: str = Field(alias="Device Protection")
    Tech_Support: str = Field(alias="Tech Support")
    Streaming_TV: str = Field(alias="Streaming TV")
    Streaming_Movies: str = Field(alias="Streaming Movies")
    Contract: str
    Paperless_Billing: str = Field(alias="Paperless Billing")
    Payment_Method: str = Field(alias="Payment Method")
    Monthly_Charges: float = Field(alias="Monthly Charges", ge=0)
    Total_Charges: float = Field(alias="Total Charges", ge=0)
    CLTV: float = Field(ge=0)

    model_config = {"populate_by_name": True}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: Customer):
    result = predictor.predict(customer.model_dump(by_alias=True))
    return result
