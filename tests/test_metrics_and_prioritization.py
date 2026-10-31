import numpy as np
import pandas as pd

from src.calibration import apply_calibrator, calibration_table, fit_sigmoid_calibrator
from src.evaluate import classification_metrics
from src.prioritize import (
    capacity_table, gain_curve, risk_category, select_for_review, threshold_metrics, top_k_metrics,
)

# 10 customers, 3 churners. Scores rank two churners at the top.
Y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_top_k_metrics_hand_calculated():
    m = top_k_metrics(Y, P, 3)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (2, 1, 1)
    assert m["precision_at_k"] == 2 / 3
    assert m["recall_at_k"] == 2 / 3
    assert np.isclose(m["lift_at_k"], (2 / 3) / 0.3)


def test_threshold_metrics_hand_calculated():
    m = threshold_metrics(Y, P, 0.5)  # flags the top 5
    assert m["flagged"] == 5
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (3, 2, 0)
    assert m["precision"] == 0.6 and m["recall"] == 1.0
    assert np.isclose(m["f1"], 2 * 0.6 / 1.6)


def test_classification_metrics_counts_add_up():
    m = classification_metrics(Y, P, 0.5)
    assert m["true_positives"] + m["false_positives"] + m["false_negatives"] + m["true_negatives"] == len(Y)
    assert 0 <= m["pr_auc"] <= 1 and 0 <= m["roc_auc"] <= 1


def test_gain_curve_ends_at_full_capture():
    gain = gain_curve(Y, P)
    assert gain["churn_captured"].iloc[0] == 0 and gain["churn_captured"].iloc[-1] == 1
    assert gain["churn_captured"].is_monotonic_increasing


def test_capacity_table_shares():
    table = capacity_table(Y, P, shares=(0.1, 0.5))
    assert table["k"].tolist() == [1, 5]


def test_select_for_review_threshold_and_capacity():
    df = pd.DataFrame({"customer_id": list("abcdefghij"), "churn_probability": P})
    out = select_for_review(df, threshold=0.45, top_k=3)
    assert out["selected_for_review"].sum() == 3
    assert out.loc[out["selected_for_review"], "customer_id"].tolist() == ["a", "b", "c"]
    out = select_for_review(df, threshold=0.75)
    assert out["selected_for_review"].sum() == 2
    out = select_for_review(df, top_pct=0.2)
    assert out["selected_for_review"].sum() == 2
    assert out["risk_rank"].tolist() == list(range(1, 11))


def test_risk_category_bands():
    cats = risk_category([0.1, 0.3, 0.5], threshold=0.44, base_rate=0.23)
    assert cats.tolist() == ["Low", "Medium", "High"]


def test_sigmoid_calibration_preserves_order_and_fixes_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.6, 2000)
    y = rng.binomial(1, true_p)
    inflated = np.clip(true_p * 1.5, 0, 0.99)  # an over-confident model
    cal = fit_sigmoid_calibrator(inflated, y)
    calibrated = apply_calibrator(cal, inflated)
    assert np.all(np.diff(calibrated[np.argsort(inflated)]) >= -1e-12)  # ranking unchanged
    assert abs(calibrated.mean() - y.mean()) < abs(inflated.mean() - y.mean())
    table = calibration_table(y, calibrated)
    assert table["customers"].sum() == len(y)
