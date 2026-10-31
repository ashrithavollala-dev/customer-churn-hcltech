"""Shared inference functions used by the API, the dashboard and the evaluation scripts.

Nothing here trains a model; it only loads the saved bundle and scores rows.
"""
from functools import lru_cache

import joblib
import pandas as pd

from src.calibration import apply_calibrator
from src.config import FEATURES, MODEL_BUNDLE_PATH
from src.preprocessing import validate_feature_frame
from src.prioritize import risk_category


@lru_cache(maxsize=1)
def load_bundle(path=MODEL_BUNDLE_PATH):
    """Load the saved model bundle once and reuse it (it is read-only)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run 'python -m src.train' from the repository root first.")
    return joblib.load(path)


def predict_raw_probability(model, df):
    """Uncalibrated probability from the fitted model (LogReg pipeline or CatBoost)."""
    return model.predict_proba(df[FEATURES])[:, 1]


def score_dataframe(df, bundle):
    """Add raw probability, final (calibrated) probability and risk category to a copy of df.

    `df` must already contain valid values for the 24 features.
    """
    scored = df.copy()
    raw = predict_raw_probability(bundle["model"], scored)
    scored["raw_model_probability"] = raw
    scored["churn_probability"] = apply_calibrator(bundle["calibrator"], raw)
    scored["risk_category"] = risk_category(scored["churn_probability"], bundle["threshold"],
                                            bundle["train_churn_rate"])
    return scored


def validate_and_score(df, bundle=None):
    """Validate user-supplied rows and score them. Returns (scored_df or None, errors)."""
    bundle = bundle or load_bundle()
    clean, errors = validate_feature_frame(df, defaults=bundle["training_defaults"])
    if errors:
        return None, errors
    return score_dataframe(clean, bundle), []


def prediction_records(scored, threshold, id_column="customer_id"):
    """Convert a scored dataframe to plain dicts for JSON responses."""
    records = []
    for i, row in scored.reset_index(drop=True).iterrows():
        records.append({
            "customer_id": str(row[id_column]) if id_column in scored.columns and pd.notna(row[id_column]) else f"row_{i}",
            "churn_probability": round(float(row["churn_probability"]), 4),
            "raw_model_probability": round(float(row["raw_model_probability"]), 4),
            "risk_category": str(row["risk_category"]),
            "flagged_at_threshold": bool(row["churn_probability"] >= threshold),
        })
    return records


def model_info(bundle):
    return {
        "model_name": bundle["model_name"],
        "calibration_applied": bundle["calibrator"] is not None,
        "operating_threshold": round(float(bundle["threshold"]), 4),
        "train_churn_rate": round(float(bundle["train_churn_rate"]), 4),
        "features": FEATURES,
        "trained_on": bundle["trained_on"],
    }


def expected_columns_example(bundle):
    """One example row (training medians / modes) to help users build a compatible CSV."""
    return {col: bundle["training_defaults"][col] for col in FEATURES}

