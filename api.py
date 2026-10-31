"""FastAPI backend for churn-risk scoring.

Start from the repository root:  uvicorn api:app --reload --port 8000
Swagger documentation:           http://127.0.0.1:8000/docs

The API only loads the saved model bundle and scores requests. It never retrains.
All ML logic lives in src/ so the Streamlit app can use the same functions directly.
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.explain import local_explanation, top_factor_text
from src.predict import load_bundle, model_info, prediction_records, validate_and_score
from src.preprocessing import NUMERIC_LIMITS

MODEL_STATE = {"bundle": None, "error": None}


@asynccontextmanager
async def lifespan(app):
    try:
        MODEL_STATE["bundle"] = load_bundle()
    except FileNotFoundError as exc:
        MODEL_STATE["error"] = str(exc)
    yield


app = FastAPI(
    title="HCLTech Customer Churn Risk API",
    description="Scores customers for 30-day churn risk using the saved model. "
                "Predictions support human retention review; they are not automatic decisions.",
    version="1.0.0",
    lifespan=lifespan,
)


def limits(name):
    low, high = NUMERIC_LIMITS[name]
    return {"ge": low, "le": high}


class CustomerFeatures(BaseModel):
    """The 24 engineered features the model was trained on (values at the snapshot date)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "customer_id": "CUST999001", "usage_change_pct_m3_to_m1": -35.0, "billing_risk": 2,
        "service_pain_score": 3, "inactive_flag": 0, "unpaid_amount_6m": 0.0, "stickiness": 1,
        "price_hike_flag": 1, "discount_pct": 0.0, "resolution_rate": 0.5, "days_since_login": 20,
        "bill_unpaid_flag": 0, "billing_disputes_last_6m": 1, "charge_change_pct_3m": 10.0,
        "renewal_due_60d": 0, "support_tickets_m1": 2, "net_charge": 650.0, "support_tickets_m3": 1,
        "last_csat_score": 2.0, "contract_type": "Month-to-month", "tenure_band": "7-12m",
        "region": "South", "paperless_billing": "Yes", "payment_method": "Cash",
        "acquisition_channel": "Online"}})

    customer_id: Optional[str] = Field(None, description="Optional identifier echoed back in the response")
    usage_change_pct_m3_to_m1: float = Field(..., **limits("usage_change_pct_m3_to_m1"))
    billing_risk: int = Field(..., **limits("billing_risk"), description="Late payments + billing disputes, last 6 months")
    service_pain_score: int = Field(..., **limits("service_pain_score"))
    inactive_flag: Literal[0, 1]
    unpaid_amount_6m: float = Field(..., **limits("unpaid_amount_6m"))
    stickiness: int = Field(..., **limits("stickiness"))
    price_hike_flag: Literal[0, 1]
    discount_pct: float = Field(..., **limits("discount_pct"))
    resolution_rate: float = Field(..., **limits("resolution_rate"))
    days_since_login: int = Field(..., **limits("days_since_login"))
    bill_unpaid_flag: Literal[0, 1]
    billing_disputes_last_6m: int = Field(..., **limits("billing_disputes_last_6m"))
    charge_change_pct_3m: float = Field(..., **limits("charge_change_pct_3m"))
    renewal_due_60d: Literal[0, 1]
    support_tickets_m1: int = Field(..., **limits("support_tickets_m1"))
    net_charge: float = Field(..., **limits("net_charge"))
    support_tickets_m3: int = Field(..., **limits("support_tickets_m3"))
    last_csat_score: Optional[float] = Field(None, ge=1, le=5, description="Optional; filled with the training median if missing")
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    tenure_band: Literal["0-6m", "7-12m", "13-24m", "25m+"]
    region: Literal["East", "North", "South", "West", "Unknown"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal["Autopay", "Credit card", "UPI", "Cash"]
    acquisition_channel: Literal["Online", "Retail store", "Call center", "Partner", "Unknown"]


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=5000)


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(description="Calibrated probability of churn in the next 30 days")
    raw_model_probability: float = Field(description="Model output before calibration")
    risk_category: str
    flagged_at_threshold: bool
    top_risk_factors: str


class Contribution(BaseModel):
    feature: str
    value: str
    shap_log_odds: float
    effect: str


class SinglePredictionResponse(Prediction):
    explanation_note: str
    contributions: list[Contribution]


class BatchPredictionResponse(BaseModel):
    count: int
    operating_threshold: float
    predictions: list[Prediction]


def get_bundle():
    if MODEL_STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=MODEL_STATE["error"] or "Model is not loaded")
    return MODEL_STATE["bundle"]


def score_customers(customers):
    bundle = get_bundle()
    df = pd.DataFrame([c.model_dump() for c in customers])
    scored, errors = validate_and_score(df, bundle)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    records = prediction_records(scored, bundle["threshold"])
    for record, factors in zip(records, top_factor_text(scored, bundle)):
        record["top_risk_factors"] = factors
    return bundle, scored, records


@app.get("/health")
def health():
    bundle = MODEL_STATE["bundle"]
    if bundle is None:
        return {"status": "model_not_loaded", "detail": MODEL_STATE["error"]}
    return {"status": "ok", **model_info(bundle)}


@app.post("/predict", response_model=SinglePredictionResponse)
def predict(customer: CustomerFeatures):
    bundle, scored, records = score_customers([customer])
    table, _ = local_explanation(scored.iloc[[0]], bundle)
    contributions = [
        {"feature": r.feature, "value": str(r.value), "shap_log_odds": round(float(r.shap_log_odds), 4), "effect": r.effect}
        for r in table.head(8).itertuples()
    ]
    return {**records[0], "contributions": contributions,
            "explanation_note": "SHAP contributions are in log-odds of the raw model, not percentage points of the "
                                "calibrated probability. They describe model behaviour, not causes."}


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchRequest):
    bundle, _, records = score_customers(request.customers)
    return {"count": len(records), "operating_threshold": bundle["threshold"], "predictions": records}
