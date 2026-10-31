"""Retention prioritization: ranking, review selection and capacity (top-K) metrics.

Thresholds and capacities are applied to already-computed probabilities, so they can be changed
in the dashboard without retraining.

Run from the repository root:  python -m src.prioritize
Builds reports/retention_queue.csv for the most recent period (test customers, Nov-Dec 2025).
"""
import numpy as np
import pandas as pd

from src.config import DATE_COL, DEFAULT_TOP_PCT, ID_COL, REPORTS_DIR


def top_k_metrics(y_true, scores, k):
    """Precision, recall and lift if the team reviews only the k highest-risk customers."""
    y_true = np.asarray(y_true)
    k = int(min(max(k, 0), len(y_true)))
    order = np.argsort(-np.asarray(scores), kind="stable")
    selected = np.zeros(len(y_true), dtype=bool)
    selected[order[:k]] = True
    tp = int((selected & (y_true == 1)).sum())
    fp = int((selected & (y_true == 0)).sum())
    fn = int((~selected & (y_true == 1)).sum())
    total_churners = int(y_true.sum())
    base_rate = y_true.mean() if len(y_true) else 0.0
    precision = tp / k if k else 0.0
    recall = tp / total_churners if total_churners else 0.0
    return {
        "k": k,
        "share_reviewed": k / len(y_true) if len(y_true) else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_at_k": precision,
        "recall_at_k": recall,  # also called churn capture
        "lift_at_k": precision / base_rate if base_rate else 0.0,
    }


def capacity_table(y_true, scores, shares=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)):
    """Top-K metrics for several review capacities expressed as a share of customers."""
    n = len(y_true)
    rows = [top_k_metrics(y_true, scores, int(round(share * n))) for share in shares]
    return pd.DataFrame(rows)


def threshold_metrics(y_true, probabilities, threshold):
    """Flagged volume and errors when every customer with probability >= threshold is flagged."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(probabilities) >= threshold
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    fn = int((~flagged & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    base_rate = y_true.mean() if len(y_true) else 0.0
    return {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift": precision / base_rate if base_rate else 0.0,
    }


def threshold_table(y_true, probabilities, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return pd.DataFrame([threshold_metrics(y_true, probabilities, t) for t in thresholds])


def gain_curve(y_true, scores):
    """Cumulative share of churners captured as more of the ranked list is reviewed."""
    y_sorted = np.asarray(y_true)[np.argsort(-np.asarray(scores), kind="stable")]
    n = len(y_sorted)
    captured = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    share = np.arange(1, n + 1) / n
    return pd.DataFrame({
        "share_reviewed": np.concatenate([[0.0], share]),
        "churn_captured": np.concatenate([[0.0], captured]),
    })


def risk_category(probabilities, threshold, base_rate):
    """High = at or above the operating threshold; Medium = above the historical churn rate; else Low."""
    p = np.asarray(probabilities)
    return np.where(p >= threshold, "High", np.where(p >= base_rate, "Medium", "Low"))


def select_for_review(scored, probability_col="churn_probability", threshold=None, top_k=None, top_pct=None):
    """Rank customers and mark who goes to the review list.

    A customer is selected when they pass the probability threshold (if given) AND fall inside the
    capacity (top_k rows or top_pct share, if given). With no rule set, nobody is selected.
    """
    ranked = scored.sort_values(probability_col, ascending=False, kind="stable").reset_index(drop=True)
    ranked["risk_rank"] = np.arange(1, len(ranked) + 1)
    selected = np.ones(len(ranked), dtype=bool)
    if threshold is None and top_k is None and top_pct is None:
        selected[:] = False
    if threshold is not None:
        selected &= ranked[probability_col].to_numpy() >= threshold
    capacity = None
    if top_k is not None:
        capacity = int(top_k)
    if top_pct is not None:
        pct_capacity = int(round(top_pct * len(ranked)))
        capacity = pct_capacity if capacity is None else min(capacity, pct_capacity)
    if capacity is not None:
        selected &= ranked["risk_rank"].to_numpy() <= capacity
    ranked["selected_for_review"] = selected
    return ranked


def customer_segment(df):
    """Short segment label shown in the queue (contract + tenure band)."""
    return df["contract_type"].astype(str) + " | " + df["tenure_band"].astype(str)


def build_retention_queue(features_df, threshold, base_rate, top_k=None, top_pct=None, top_factors=None):
    """Score table -> ranked retention queue with risk category, segment and top risk factors."""
    queue = features_df.copy()
    queue["risk_category"] = risk_category(queue["churn_probability"], threshold, base_rate)
    queue["segment"] = customer_segment(queue)
    if top_factors is not None:
        queue["top_risk_factors"] = top_factors
    queue = select_for_review(queue, threshold=threshold, top_k=top_k, top_pct=top_pct)
    first = [ID_COL, DATE_COL, "risk_rank", "churn_probability", "risk_category", "segment",
             "top_risk_factors", "selected_for_review"]
    first = [c for c in first if c in queue.columns]
    return queue[first + [c for c in queue.columns if c not in first]]


def main():
    # Imported here to avoid a circular import (predict/explain use helpers from this module).
    from src.explain import top_factor_text
    from src.predict import load_bundle, score_dataframe
    from src.preprocessing import load_split

    bundle = load_bundle()
    latest = load_split("test")
    scored = score_dataframe(latest, bundle)
    factors = top_factor_text(latest, bundle)
    queue = build_retention_queue(
        scored, threshold=bundle["threshold"], base_rate=bundle["train_churn_rate"],
        top_pct=DEFAULT_TOP_PCT, top_factors=factors,
    )
    # The queue is a forward-looking list: the churn label is removed on purpose.
    queue = queue.drop(columns=["churn_next_30d"], errors="ignore")
    queue.to_csv(REPORTS_DIR / "retention_queue.csv", index=False)
    print(f"Saved retention queue: {len(queue)} customers, {int(queue['selected_for_review'].sum())} selected "
          f"(threshold {bundle['threshold']:.2f} and top {DEFAULT_TOP_PCT:.0%} capacity).")
    print(queue.head(10)[[ID_COL, "risk_rank", "churn_probability", "risk_category", "segment",
                          "top_risk_factors"]].to_string(index=False))


if __name__ == "__main__":
    main()
