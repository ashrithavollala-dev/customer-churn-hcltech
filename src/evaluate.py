"""Metrics, plots and the one-time final test evaluation.

Run from the repository root AFTER training:  python -m src.evaluate
This is the only script that reads test labels. Model choice, calibration and threshold are
already frozen inside models/churn_model_bundle.joblib, so nothing here can change them.
"""
import json

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from src.calibration import calibration_table
from src.config import (
    DATE_COL, DEFAULT_TOP_PCT, FIGURES_DIR, ID_COL, MIN_SEGMENT_CHURNERS, MIN_SEGMENT_ROWS, RANDOM_SEED,
    REPORTS_DIR, SEGMENT_COLUMNS, TARGET,
)
from src.preprocessing import add_billing_segment
from src.prioritize import capacity_table, gain_curve, threshold_metrics, threshold_table, top_k_metrics

MODEL_LABELS = {"dummy": "Dummy (prior)", "logistic_regression": "Logistic Regression", "catboost": "CatBoost"}


def classification_metrics(y_true, probabilities, threshold):
    """Threshold-free (PR-AUC, ROC-AUC, Brier) and threshold-based metrics in one dict."""
    y_true = np.asarray(y_true)
    has_both_classes = len(np.unique(y_true)) == 2
    k = int(round(DEFAULT_TOP_PCT * len(y_true)))
    top = top_k_metrics(y_true, probabilities, k)
    metrics = {
        "rows": len(y_true),
        "churners": int(y_true.sum()),
        "pr_auc": average_precision_score(y_true, probabilities) if has_both_classes else np.nan,
        "roc_auc": roc_auc_score(y_true, probabilities) if has_both_classes else np.nan,
        "brier": brier_score_loss(y_true, probabilities),
        "mean_predicted": float(np.mean(probabilities)),
        "observed_rate": float(np.mean(y_true)),
        "precision_at_top20pct": top["precision_at_k"],
        "recall_at_top20pct": top["recall_at_k"],
        "lift_at_top20pct": top["lift_at_k"],
    }
    at_threshold = threshold_metrics(y_true, probabilities, threshold)
    metrics.update({key: at_threshold[key] for key in
                    ["threshold", "flagged", "precision", "recall", "f1",
                     "true_positives", "false_positives", "false_negatives"]})
    metrics["true_negatives"] = int(len(y_true) - metrics["flagged"] - metrics["false_negatives"])
    return metrics


def bootstrap_ci(y_true, probabilities, n_boot=1000, seed=RANDOM_SEED):
    """95% percentile bootstrap intervals for PR-AUC, ROC-AUC and top-20% recall/precision."""
    rng = np.random.default_rng(seed)
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    k = int(round(DEFAULT_TOP_PCT * len(y_true)))
    samples = {"pr_auc": [], "roc_auc": [], "recall_at_top20pct": [], "precision_at_top20pct": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() in (0, len(idx)):
            continue
        samples["pr_auc"].append(average_precision_score(y_true[idx], probabilities[idx]))
        samples["roc_auc"].append(roc_auc_score(y_true[idx], probabilities[idx]))
        top = top_k_metrics(y_true[idx], probabilities[idx], k)
        samples["recall_at_top20pct"].append(top["recall_at_k"])
        samples["precision_at_top20pct"].append(top["precision_at_k"])
    rows = []
    for metric, values in samples.items():
        rows.append({"metric": metric, "ci_low": np.percentile(values, 2.5),
                     "ci_high": np.percentile(values, 97.5), "bootstrap_samples": len(values)})
    return pd.DataFrame(rows)


def segment_performance(df, probability_col, threshold):
    """Per-segment counts, churn rate and errors at the operating threshold."""
    df = add_billing_segment(df)
    rows = []
    for col in SEGMENT_COLUMNS:
        for value, group in df.groupby(col):
            y = group[TARGET].to_numpy()
            m = threshold_metrics(y, group[probability_col].to_numpy(), threshold)
            small = len(group) < MIN_SEGMENT_ROWS or y.sum() < MIN_SEGMENT_CHURNERS
            rows.append({
                "segment": col, "segment_value": value, "customers": len(group), "churners": int(y.sum()),
                "actual_churn_rate": y.mean(), "mean_predicted": group[probability_col].mean(),
                "flagged": m["flagged"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "false_positives": m["false_positives"], "false_negatives": m["false_negatives"],
                "sample_note": "insufficient sample - interpret with caution" if small else "",
            })
    return pd.DataFrame(rows)


# ---------- plots (PNG files for the README / report; the dashboard redraws them with Plotly) ----------

def plot_pr_roc(y_true, prob_by_model, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in prob_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        axes[0].plot(recall, precision, label=f"{MODEL_LABELS.get(name, name)} (AP={average_precision_score(y_true, p):.3f})")
        fpr, tpr, _ = roc_curve(y_true, p)
        axes[1].plot(fpr, tpr, label=f"{MODEL_LABELS.get(name, name)} (AUC={roc_auc_score(y_true, p):.3f})")
    axes[0].axhline(np.mean(y_true), color="grey", ls="--", lw=1, label="No-skill (churn rate)")
    axes[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall - {title}")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {title}")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def plot_confusion(y_true, probabilities, threshold, title, filename):
    cm = confusion_matrix(y_true, (np.asarray(probabilities) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
           xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def plot_calibration(y_true, prob_by_label, title, filename, n_bins=5):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Perfect calibration")
    for label, p in prob_by_label.items():
        observed, predicted = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=f"{label} (Brier={brier_score_loss(y_true, p):.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def plot_gain_lift(y_true, probabilities, title, filename):
    gain = gain_curve(y_true, probabilities)
    caps = capacity_table(y_true, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(gain["share_reviewed"], gain["churn_captured"], label="Model")
    axes[0].plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Random")
    axes[0].set(xlabel="Share of customers reviewed", ylabel="Share of churners captured", title=f"Cumulative gain - {title}")
    axes[1].bar([f"{s:.0%}" for s in caps["share_reviewed"]], caps["lift_at_k"], color="#1f5fa8")
    axes[1].axhline(1, color="grey", ls="--", lw=1)
    axes[1].set(xlabel="Top share reviewed", ylabel="Lift", title=f"Lift at capacity - {title}")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def main():
    from src.predict import load_bundle, predict_raw_probability, score_dataframe
    from src.preprocessing import load_split

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    test = load_split("test")
    y_test = test[TARGET].to_numpy()
    threshold = bundle["threshold"]

    # Selected model, exactly as it will be used in production (calibrated if calibration was applied).
    scored = score_dataframe(test, bundle)
    p_final = scored["churn_probability"].to_numpy()
    p_raw = scored["raw_model_probability"].to_numpy()

    metrics = classification_metrics(y_test, p_final, threshold)
    metrics["model"] = bundle["model_name"]
    metrics["probability"] = "calibrated" if bundle["calibrator"] is not None else "raw"
    raw_metrics = classification_metrics(y_test, p_raw, threshold)
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "test_metrics.csv", index=False)

    ci = bootstrap_ci(y_test, p_final)
    point = {"pr_auc": metrics["pr_auc"], "roc_auc": metrics["roc_auc"],
             "recall_at_top20pct": metrics["recall_at_top20pct"], "precision_at_top20pct": metrics["precision_at_top20pct"]}
    ci.insert(1, "point_estimate", ci["metric"].map(point))
    ci.to_csv(REPORTS_DIR / "test_bootstrap_ci.csv", index=False)

    capacity_table(y_test, p_final).to_csv(REPORTS_DIR / "test_capacity_lift.csv", index=False)
    threshold_table(y_test, p_final).to_csv(REPORTS_DIR / "test_threshold_tradeoff.csv", index=False)
    calibration_table(y_test, p_final).to_csv(REPORTS_DIR / "test_calibration_table.csv", index=False)
    segment_performance(scored, "churn_probability", threshold).to_csv(
        REPORTS_DIR / "test_segment_performance.csv", index=False)

    # Both candidate models on test - reported for transparency only, AFTER selection was frozen.
    reference = {name: predict_raw_probability(model, test) for name, model in bundle["candidate_models"].items()}
    ref_rows = []
    for name, p in reference.items():
        m = classification_metrics(y_test, p, 0.5)
        m["model"] = name
        ref_rows.append(m)
    pd.DataFrame(ref_rows).to_csv(REPORTS_DIR / "test_reference_both_models_raw.csv", index=False)

    report_text = classification_report(y_test, (p_final >= threshold).astype(int), target_names=["stay", "churn"], digits=3)
    (REPORTS_DIR / "test_classification_report.txt").write_text(
        f"Selected model: {bundle['model_name']} | threshold {threshold:.2f} (chosen on validation)\n\n{report_text}",
        encoding="utf-8")

    # Predictions file for the dashboard (labels kept here because this is historical evaluation data).
    out = test[[ID_COL, DATE_COL, TARGET]].copy()
    out["raw_model_probability"] = p_raw
    out["churn_probability"] = p_final
    for name, p in reference.items():
        out[f"raw_{name}"] = p
    out.to_csv(REPORTS_DIR / "predictions_test.csv", index=False)

    label = MODEL_LABELS[bundle["model_name"]]
    plot_pr_roc(y_test, {bundle["model_name"]: p_final}, "test", "test_pr_roc.png")
    plot_confusion(y_test, p_final, threshold, f"{label} - test (t={threshold:.2f})", "test_confusion.png")
    plot_calibration(y_test, {"raw": p_raw, "calibrated": p_final} if bundle["calibrator"] is not None else {"raw": p_raw},
                     "Calibration - test", "test_calibration.png")
    plot_gain_lift(y_test, p_final, "test", "test_gain_lift.png")

    summary = {key: (round(v, 4) if isinstance(v, float) else v) for key, v in metrics.items()}
    summary["brier_raw_uncalibrated"] = round(raw_metrics["brier"], 4)
    (REPORTS_DIR / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(ci.to_string(index=False))
    print(report_text)


if __name__ == "__main__":
    main()
