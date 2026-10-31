"""Sigmoid (Platt) calibration fitted on validation predictions of the frozen selected model.

Procedure (no test labels are used anywhere here):
1. The selected model is trained on TRAIN only and then frozen.
2. Its raw validation probabilities are converted to log-odds and a 1-feature logistic regression
   maps log-odds -> calibrated probability.
3. To judge whether calibration helps, the Brier score is estimated with stratified K-fold
   cross-validation inside the validation set: the calibrator is fitted on 4/5 of the validation
   rows and scored on the remaining 1/5, so improvement is never measured on rows it was fitted on.
4. Calibration is applied only if this cross-validated Brier score is lower than the raw Brier score.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid_calibrator(raw_probabilities, y_true):
    # Very weak regularisation so this is ordinary Platt scaling.
    calibrator = LogisticRegression(C=1e4)
    calibrator.fit(_logit(raw_probabilities), y_true)
    return calibrator


def apply_calibrator(calibrator, raw_probabilities):
    if calibrator is None:
        return np.asarray(raw_probabilities, dtype=float)
    return calibrator.predict_proba(_logit(raw_probabilities))[:, 1]


def cross_validated_brier(raw_probabilities, y_true, n_splits=5):
    """Brier score of out-of-fold calibrated predictions within the validation set."""
    raw_probabilities = np.asarray(raw_probabilities)
    y_true = np.asarray(y_true)
    out_of_fold = np.zeros(len(y_true))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for fit_idx, score_idx in folds.split(raw_probabilities.reshape(-1, 1), y_true):
        calibrator = fit_sigmoid_calibrator(raw_probabilities[fit_idx], y_true[fit_idx])
        out_of_fold[score_idx] = apply_calibrator(calibrator, raw_probabilities[score_idx])
    return brier_score_loss(y_true, out_of_fold), out_of_fold


def calibration_table(y_true, probabilities, n_bins=5):
    """Mean predicted vs observed churn rate in equal-size probability bins (quantile bins).

    Quantile bins are used because with ~170 rows fixed-width bins would leave some nearly empty.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probabilities)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    table = df.groupby("bin").agg(customers=("y", "size"), mean_predicted=("p", "mean"),
                                  observed_rate=("y", "mean"), churners=("y", "sum")).reset_index()
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table


def decide_calibration(raw_val_probabilities, y_val):
    """Return (calibrator or None, summary dict) following the procedure in the module docstring."""
    raw_brier = brier_score_loss(y_val, raw_val_probabilities)
    cv_brier, out_of_fold = cross_validated_brier(raw_val_probabilities, y_val)
    use_calibration = cv_brier < raw_brier
    calibrator = fit_sigmoid_calibrator(raw_val_probabilities, y_val) if use_calibration else None
    summary = {
        "validation_brier_raw": raw_brier,
        "validation_brier_calibrated_cv": cv_brier,
        "calibration_applied": bool(use_calibration),
        "validation_mean_predicted_raw": float(np.mean(raw_val_probabilities)),
        "validation_mean_predicted_calibrated_cv": float(np.mean(out_of_fold)),
        "validation_observed_churn_rate": float(np.mean(y_val)),
    }
    if calibrator is not None:
        summary["platt_slope"] = float(calibrator.coef_[0][0])
        summary["platt_intercept"] = float(calibrator.intercept_[0])
    return calibrator, summary, out_of_fold
