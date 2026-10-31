"""Stage 2: train, tune on validation, select, calibrate, choose threshold, save the model bundle.

Run from the repository root:  python -m src.train

Order of decisions (test data is never loaded in this file):
1. Fit Dummy, Logistic Regression and CatBoost candidates on TRAIN.
2. Pick hyperparameters and the model family using VALIDATION PR-AUC and top-20% churn capture.
3. Fit sigmoid calibration on validation predictions of the frozen selected model
   (applied only if a cross-validated Brier score inside validation improves).
4. Choose the operating threshold on validation (calibrated) probabilities.
5. Save everything to models/churn_model_bundle.joblib. Test evaluation happens in src/evaluate.py.
"""
import itertools
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.calibration import apply_calibrator, calibration_table, decide_calibration
from src.config import (
    CATEGORICAL_FEATURES, DATE_COL, DEFAULT_TOP_PCT, FEATURES, FIGURES_DIR, ID_COL, MODEL_BUNDLE_PATH,
    MODELS_DIR, RANDOM_SEED, REPORTS_DIR, TARGET,
)
from src.evaluate import (
    MODEL_LABELS, classification_metrics, plot_calibration, plot_confusion, plot_gain_lift, plot_pr_roc,
)
from src.predict import predict_raw_probability
from src.preprocessing import load_split, make_linear_preprocessor, split_features_target, training_defaults
from src.prioritize import capacity_table, threshold_table

# Small, deliberate grids: the validation set has only ~43 churners, so a big search would overfit it.
LOGREG_GRID = {"C": [0.01, 0.03, 0.1, 0.3, 1.0], "class_weight": [None, "balanced"]}
CATBOOST_GRID = {"depth": [3, 4, 5], "l2_leaf_reg": [3, 10], "iterations": [200, 500]}

# CatBoost must beat Logistic Regression by this much validation PR-AUC to justify the extra complexity.
COMPLEXITY_MARGIN = 0.02


def build_dummy():
    return DummyClassifier(strategy="prior")


def build_logistic_regression(C, class_weight):
    return Pipeline([
        ("preprocess", make_linear_preprocessor()),
        ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000,  # L2 penalty (default)
                                     random_state=RANDOM_SEED)),
    ])


def build_catboost(depth, l2_leaf_reg, iterations):
    # CatBoost handles the categorical columns natively; no scaling or one-hot encoding is needed.
    return CatBoostClassifier(
        depth=depth, l2_leaf_reg=l2_leaf_reg, iterations=iterations, learning_rate=0.03,
        cat_features=CATEGORICAL_FEATURES, random_seed=RANDOM_SEED, verbose=0, allow_writing_files=False,
    )


def evaluate_candidate(model, X_train, y_train, X_val, y_val):
    p_train = predict_raw_probability(model, X_train)
    p_val = predict_raw_probability(model, X_val)
    train_m = classification_metrics(y_train, p_train, 0.5)
    val_m = classification_metrics(y_val, p_val, 0.5)
    return {
        "train_pr_auc": train_m["pr_auc"], "val_pr_auc": val_m["pr_auc"],
        "train_roc_auc": train_m["roc_auc"], "val_roc_auc": val_m["roc_auc"],
        "val_recall_at_top20pct": val_m["recall_at_top20pct"], "val_brier": val_m["brier"],
    }


def tune(family, grid, builder, X_train, y_train, X_val, y_val):
    rows, models = [], {}
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = builder(**params).fit(X_train, y_train)
        result = evaluate_candidate(model, X_train, y_train, X_val, y_val)
        key = json.dumps(params)
        rows.append({"family": family, "params": key, **result})
        models[key] = model
    table = pd.DataFrame(rows)
    # Highest validation PR-AUC, ties broken by top-20% churn capture.
    best = table.sort_values(["val_pr_auc", "val_recall_at_top20pct"], ascending=False).iloc[0]
    return table, models[best["params"]], json.loads(best["params"])


def choose_model(lr_row, cb_row):
    """Prefer the simpler Logistic Regression unless CatBoost is clearly better on validation."""
    cb_clearly_better = (cb_row["val_pr_auc"] > lr_row["val_pr_auc"] + COMPLEXITY_MARGIN
                         and cb_row["val_recall_at_top20pct"] >= lr_row["val_recall_at_top20pct"])
    if cb_clearly_better:
        return "catboost", (f"CatBoost validation PR-AUC {cb_row['val_pr_auc']:.3f} exceeds Logistic Regression "
                            f"{lr_row['val_pr_auc']:.3f} by more than {COMPLEXITY_MARGIN} with equal or better top-20% capture.")
    return "logistic_regression", (
        f"Logistic Regression validation PR-AUC {lr_row['val_pr_auc']:.3f} vs CatBoost {cb_row['val_pr_auc']:.3f}; "
        f"top-20% capture {lr_row['val_recall_at_top20pct']:.3f} vs {cb_row['val_recall_at_top20pct']:.3f}. "
        f"CatBoost is not better by more than {COMPLEXITY_MARGIN} PR-AUC with equal or better capture, so the simpler, "
        "more transparent model is kept.")


def choose_threshold(y_val, p_val):
    """Threshold that maximises F1 on validation probabilities (ties -> lower threshold, higher recall)."""
    table = threshold_table(y_val, p_val, thresholds=np.round(np.arange(0.05, 0.91, 0.01), 2))
    best = table.sort_values(["f1", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def comparison_table(models, X_train, y_train, X_val, y_val):
    """Train vs validation metrics for each final candidate (raw probabilities, threshold 0.5)."""
    rows = []
    for name, model in models.items():
        for split, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
            m = classification_metrics(y, predict_raw_probability(model, X), 0.5)
            rows.append({"model": MODEL_LABELS[name], "split": split, **m})
    table = pd.DataFrame(rows)
    gaps = []
    for name in models:
        tr = table[(table.model == MODEL_LABELS[name]) & (table.split == "train")].iloc[0]
        va = table[(table.model == MODEL_LABELS[name]) & (table.split == "validation")].iloc[0]
        gaps.append({"model": MODEL_LABELS[name], "pr_auc_train_minus_val": tr.pr_auc - va.pr_auc,
                     "roc_auc_train_minus_val": tr.roc_auc - va.roc_auc})
    return table, pd.DataFrame(gaps)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("validation")
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(val)
    print(f"Train {X_train.shape}, churn rate {y_train.mean():.3f} | Validation {X_val.shape}, churn rate {y_val.mean():.3f}")

    # 1-2. Fit candidates on train, tune on validation.
    dummy = build_dummy().fit(X_train, y_train)
    lr_table, lr_model, lr_params = tune("logistic_regression", LOGREG_GRID, build_logistic_regression,
                                         X_train, y_train, X_val, y_val)
    cb_table, cb_model, cb_params = tune("catboost", CATBOOST_GRID, build_catboost, X_train, y_train, X_val, y_val)
    tuning = pd.concat([lr_table, cb_table], ignore_index=True)
    tuning.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    print(tuning.round(3).to_string(index=False))

    candidates = {"dummy": dummy, "logistic_regression": lr_model, "catboost": cb_model}
    comparison, gaps = comparison_table(candidates, X_train, y_train, X_val, y_val)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    gaps.to_csv(REPORTS_DIR / "overfitting_gaps.csv", index=False)

    lr_row = lr_table.sort_values(["val_pr_auc", "val_recall_at_top20pct"], ascending=False).iloc[0]
    cb_row = cb_table.sort_values(["val_pr_auc", "val_recall_at_top20pct"], ascending=False).iloc[0]
    selected_name, reason = choose_model(lr_row, cb_row)
    selected_model = candidates[selected_name]
    print(f"\nSelected: {selected_name}. {reason}")

    # 3. Calibration on validation predictions of the frozen selected model.
    p_val_raw = predict_raw_probability(selected_model, X_val)
    calibrator, cal_summary, p_val_oof = decide_calibration(p_val_raw, y_val)
    p_val_final = apply_calibrator(calibrator, p_val_raw)
    print("Calibration:", json.dumps(cal_summary, indent=2))

    # 4. Operating threshold on validation probabilities that the deployed model will output.
    threshold, val_thresholds = choose_threshold(y_val, p_val_final)
    print(f"Operating threshold (max validation F1): {threshold:.2f}")

    # Validation reports and figures for both models.
    val_probs = {name: predict_raw_probability(candidates[name], X_val) for name in ["logistic_regression", "catboost"]}
    plot_pr_roc(y_val, val_probs, "validation", "validation_pr_roc.png")
    k_threshold = {}
    for name, p in val_probs.items():
        # Confusion matrices at the default top-20% review capacity (same capacity for both models).
        cutoff = np.sort(p)[::-1][int(round(DEFAULT_TOP_PCT * len(p))) - 1]
        k_threshold[name] = cutoff
        plot_confusion(y_val, p, cutoff, f"{MODEL_LABELS[name]} - validation top 20%", f"validation_confusion_{name}.png")
    plot_calibration(y_val, {"raw": p_val_raw, "calibrated (out-of-fold)": p_val_oof},
                     f"Calibration - validation ({MODEL_LABELS[selected_name]})", "validation_calibration.png")
    plot_gain_lift(y_val, p_val_final, "validation", "validation_gain_lift.png")

    reports = []
    for name, p in val_probs.items():
        reports.append(f"== {MODEL_LABELS[name]} - validation, top 20% capacity (score >= {k_threshold[name]:.3f}) ==\n"
                       + classification_report(y_val, (p >= k_threshold[name]).astype(int), target_names=["stay", "churn"], digits=3))
    reports.append(f"== Selected {MODEL_LABELS[selected_name]} - validation, operating threshold {threshold:.2f} ==\n"
                   + classification_report(y_val, (p_val_final >= threshold).astype(int), target_names=["stay", "churn"], digits=3))
    (REPORTS_DIR / "validation_classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")

    pd.concat([
        calibration_table(y_val, p_val_raw).assign(probability="raw"),
        calibration_table(y_val, p_val_oof).assign(probability="calibrated_out_of_fold"),
    ]).to_csv(REPORTS_DIR / "validation_calibration_table.csv", index=False)
    val_thresholds.to_csv(REPORTS_DIR / "validation_threshold_tradeoff.csv", index=False)
    capacity_table(y_val, p_val_final).to_csv(REPORTS_DIR / "validation_capacity_lift.csv", index=False)

    val_out = val[[ID_COL, DATE_COL, TARGET]].copy()
    val_out["raw_model_probability"] = p_val_raw
    val_out["churn_probability"] = p_val_final
    val_out["calibrated_out_of_fold"] = p_val_oof
    for name, p in val_probs.items():
        val_out[f"raw_{name}"] = p
    val_out.to_csv(REPORTS_DIR / "predictions_validation.csv", index=False)

    # 5. Save the bundle. Test data has not been touched.
    selection = {
        "selected_model": selected_name, "reason": reason, "logistic_regression_params": lr_params,
        "catboost_params": cb_params, "operating_threshold": threshold,
        "threshold_rule": "maximise F1 on validation probabilities", **cal_summary,
    }
    bundle = {
        "model_name": selected_name,
        "model": selected_model,
        "calibrator": calibrator,
        "threshold": threshold,
        "features": FEATURES,
        "train_churn_rate": float(y_train.mean()),
        "training_defaults": training_defaults(train),
        "background_sample": X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_SEED),
        "candidate_models": {"logistic_regression": lr_model, "catboost": cb_model},
        "selection": selection,
        "trained_on": {"train_period": f"{train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}",
                       "validation_period": f"{val[DATE_COL].min().date()} to {val[DATE_COL].max().date()}",
                       "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")},
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    (REPORTS_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {MODEL_BUNDLE_PATH.relative_to(MODELS_DIR.parent)}")
    print(comparison[["model", "split", "pr_auc", "roc_auc", "brier", "recall_at_top20pct", "precision", "recall", "f1"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
