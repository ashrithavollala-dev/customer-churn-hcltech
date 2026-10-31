import numpy as np
import pandas as pd

from src.config import DATE_COL, FEATURES, ID_COL, TARGET
from src.preprocessing import (
    load_all_splits, make_linear_preprocessor, split_features_target, training_defaults, validate_feature_frame,
)


def test_splits_are_chronological_and_disjoint():
    splits = load_all_splits()
    assert splits["train"][DATE_COL].max() < splits["validation"][DATE_COL].min()
    assert splits["validation"][DATE_COL].max() < splits["test"][DATE_COL].min()
    ids = [set(df[ID_COL]) for df in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_model_inputs_exclude_id_date_and_target():
    train = load_all_splits()["train"]
    X, y = split_features_target(train)
    assert list(X.columns) == FEATURES
    assert len(FEATURES) == 24
    for col in (ID_COL, DATE_COL, TARGET):
        assert col not in X.columns
    assert set(np.unique(y)) == {0, 1}


def test_preprocessor_is_fitted_on_train_only_and_handles_unseen_category():
    train = load_all_splits()["train"]
    X, _ = split_features_target(train)
    pre = make_linear_preprocessor().fit(X)
    scaled_means = pre.transform(X)[:, :18].mean(axis=0)
    assert np.allclose(scaled_means, 0, atol=1e-8)  # scaler statistics come from train
    odd = X.head(1).copy()
    odd["region"] = "Mars"  # unseen category -> all-zero one-hot, no crash
    assert pre.transform(odd).shape[1] == pre.transform(X.head(1)).shape[1]


def test_validate_feature_frame_accepts_good_rows_and_fills_csat():
    train = load_all_splits()["train"]
    defaults = training_defaults(train)
    rows = train[FEATURES].head(3).copy()
    rows["last_csat_score"] = [np.nan, 3.0, np.nan]
    clean, errors = validate_feature_frame(rows, defaults)
    assert errors == []
    assert clean["last_csat_score"].tolist() == [defaults["last_csat_score"], 3.0, defaults["last_csat_score"]]


def test_validate_feature_frame_reports_problems():
    train = load_all_splits()["train"]
    rows = train[FEATURES].head(2).copy()
    rows["contract_type"] = ["Weekly", "One year"]
    rows["discount_pct"] = [150.0, 5.0]
    rows["net_charge"] = ["abc", 100.0]
    clean, errors = validate_feature_frame(rows, training_defaults(train))
    assert clean is None
    text = " ".join(errors)
    assert "contract_type" in text and "discount_pct" in text and "net_charge" in text

    clean, errors = validate_feature_frame(pd.DataFrame({"billing_risk": [1]}))
    assert clean is None and "Missing required columns" in errors[0]
