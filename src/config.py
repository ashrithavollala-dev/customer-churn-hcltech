"""Shared paths, column lists and settings used by every stage of the project."""
from pathlib import Path

# Everything is resolved relative to the repository root, so scripts work from any machine.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "customer_churn_raw_1050.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SPLIT_FILES = {
    "train": DATA_DIR / "train_final.csv",
    "validation": DATA_DIR / "validation_final.csv",
    "test": DATA_DIR / "test_final.csv",
}

MODEL_BUNDLE_PATH = MODELS_DIR / "churn_model_bundle.joblib"

ID_COL = "customer_id"
DATE_COL = "snapshot_date"
TARGET = "churn_next_30d"

# The 24 features chosen earlier with L1 logistic regression + TimeSeriesSplit on TRAIN only.
CATEGORICAL_FEATURES = [
    "contract_type",
    "tenure_band",
    "region",
    "paperless_billing",
    "payment_method",
    "acquisition_channel",
]
NUMERIC_FEATURES = [
    "usage_change_pct_m3_to_m1",
    "billing_risk",
    "service_pain_score",
    "inactive_flag",
    "unpaid_amount_6m",
    "stickiness",
    "price_hike_flag",
    "discount_pct",
    "resolution_rate",
    "days_since_login",
    "bill_unpaid_flag",
    "billing_disputes_last_6m",
    "charge_change_pct_3m",
    "renewal_due_60d",
    "support_tickets_m1",
    "net_charge",
    "support_tickets_m3",
    "last_csat_score",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed category values (taken from the training data).
CATEGORY_VALUES = {
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "tenure_band": ["0-6m", "7-12m", "13-24m", "25m+"],
    "region": ["East", "North", "South", "West", "Unknown"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": ["Autopay", "Credit card", "UPI", "Cash"],
    "acquisition_channel": ["Online", "Retail store", "Call center", "Partner", "Unknown"],
}

RANDOM_SEED = 42

# Default review capacity used for model selection and reporting (top 20% of customers).
DEFAULT_TOP_PCT = 0.20

# Segments used for error analysis. Billing segment is derived from billing_risk.
SEGMENT_COLUMNS = [
    "contract_type",
    "tenure_band",
    "region",
    "payment_method",
    "paperless_billing",
    "acquisition_channel",
    "billing_segment",
    "price_hike_flag",
    "renewal_due_60d",
]
MIN_SEGMENT_ROWS = 30
MIN_SEGMENT_CHURNERS = 5
