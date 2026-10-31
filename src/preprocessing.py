"""Loading the splits, separating features from the target, and validating new input rows."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    DATE_COL,
    FEATURES,
    NUMERIC_FEATURES,
    SPLIT_FILES,
    TARGET,
)

# Sensible physical limits for each numeric feature. Values outside them are rejected
# rather than silently clipped, so the user can see what is wrong with the input.
NUMERIC_LIMITS = {
    "usage_change_pct_m3_to_m1": (-100.0, 1000.0),
    "billing_risk": (0, 20),
    "service_pain_score": (0, 50),
    "inactive_flag": (0, 1),
    "unpaid_amount_6m": (0.0, 1e6),
    "stickiness": (0, 20),
    "price_hike_flag": (0, 1),
    "discount_pct": (0.0, 100.0),
    "resolution_rate": (0.0, 1.0),
    "days_since_login": (0, 3650),
    "bill_unpaid_flag": (0, 1),
    "billing_disputes_last_6m": (0, 50),
    "charge_change_pct_3m": (-100.0, 1000.0),
    "renewal_due_60d": (0, 1),
    "support_tickets_m1": (0, 100),
    "net_charge": (0.0, 1e6),
    "support_tickets_m3": (0, 100),
    "last_csat_score": (1.0, 5.0),
}


def load_split(name):
    """Load one of 'train', 'validation', 'test' exactly as provided (no reshuffling)."""
    df = pd.read_csv(SPLIT_FILES[name])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def load_all_splits():
    return {name: load_split(name) for name in SPLIT_FILES}


def split_features_target(df):
    """Return model inputs (only the 24 features) and the label. IDs and dates are excluded."""
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).to_numpy() if TARGET in df.columns else None
    return X, y


def make_linear_preprocessor():
    """Imputation + scaling for numbers, one-hot encoding for categories (used by Logistic Regression).

    Fitted on training rows only, inside the model pipeline, so validation/test never influence it.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def training_defaults(train_df):
    """Median of each numeric feature and mode of each categorical feature from TRAIN.

    Used for the default values in the dashboard form and to fill an optional missing CSAT score.
    """
    defaults = {col: float(train_df[col].median()) for col in NUMERIC_FEATURES}
    defaults.update({col: str(train_df[col].mode()[0]) for col in CATEGORICAL_FEATURES})
    return defaults


def add_billing_segment(df):
    """Group billing_risk (late payments + disputes) into readable bands for segment analysis."""
    df = df.copy()
    df["billing_segment"] = pd.cut(
        df["billing_risk"], bins=[-1, 0, 2, np.inf], labels=["No billing issues", "1-2 issues", "3+ issues"]
    ).astype(str)
    return df


def validate_feature_frame(df, defaults=None):
    """Check that uploaded/entered rows contain usable values for all 24 model features.

    Returns (clean_dataframe, list_of_error_messages). If the error list is not empty the rows
    must not be scored. Only last_csat_score may be missing; it is filled with the training median,
    which mirrors how missing CSAT was handled when the dataset was prepared.
    """
    errors = []
    missing_cols = [c for c in FEATURES if c not in df.columns and c != "last_csat_score"]
    if missing_cols:
        return None, [f"Missing required columns: {', '.join(missing_cols)}"]

    clean = df.copy()
    if "last_csat_score" not in clean.columns:
        clean["last_csat_score"] = np.nan
    if defaults is not None:
        clean["last_csat_score"] = pd.to_numeric(clean["last_csat_score"], errors="coerce").fillna(
            defaults["last_csat_score"]
        )

    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(clean[col], errors="coerce")
        bad_type = values.isna() & clean[col].notna()
        if bad_type.any():
            errors.append(f"{col}: non-numeric values in rows {_row_list(bad_type)}")
        if values.isna().any() and not bad_type.any():
            errors.append(f"{col}: missing values in rows {_row_list(values.isna())}")
        low, high = NUMERIC_LIMITS[col]
        out_of_range = values.notna() & ((values < low) | (values > high))
        if out_of_range.any():
            errors.append(f"{col}: values outside [{low}, {high}] in rows {_row_list(out_of_range)}")
        clean[col] = values

    for col in CATEGORICAL_FEATURES:
        values = clean[col].astype("string").str.strip()
        bad = ~values.isin(CATEGORY_VALUES[col]) | values.isna()
        if bad.any():
            errors.append(
                f"{col}: unexpected values {sorted(values[bad].fillna('<missing>').unique().tolist())}; "
                f"allowed: {CATEGORY_VALUES[col]}"
            )
        clean[col] = values.astype(object)

    extra_cols = [c for c in clean.columns if c not in FEATURES]
    return clean[FEATURES + extra_cols] if not errors else None, errors


def _row_list(mask, limit=10):
    rows = np.flatnonzero(np.asarray(mask)).tolist()
    text = ", ".join(str(r) for r in rows[:limit])
    return text + (" ..." if len(rows) > limit else "")
