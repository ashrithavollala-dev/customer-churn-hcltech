"""Streamlit dashboard for churn-risk scoring and retention prioritization.

Start from the repository root:  streamlit run app.py

Every number shown comes from the saved model bundle (models/) or from files written by the
training/evaluation scripts (reports/). Predictions use the shared functions in src/predict.py;
if the environment variable CHURN_API_URL is set, Page 2 calls the FastAPI backend instead and
falls back to local scoring if the API cannot be reached.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.calibration import calibration_table
from src.config import (
    CATEGORICAL_FEATURES, CATEGORY_VALUES, FEATURES, FIGURES_DIR, ID_COL, NUMERIC_FEATURES, REPORTS_DIR, TARGET,
)
from src.explain import local_explanation, plain_language_summary, top_factor_text, waterfall_figure
from src.predict import load_bundle, score_dataframe, validate_and_score
from src.preprocessing import NUMERIC_LIMITS, add_billing_segment, load_split
from src.prioritize import (
    build_retention_queue, capacity_table, gain_curve, threshold_metrics, threshold_table,
)

st.set_page_config(page_title="Churn Risk & Retention", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
RISK_COLORS = {"Low": "#b7d3f2", "Medium": "#5b9be3", "High": "#1a4f8f"}
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "catboost": "CatBoost"}
BINARY_FEATURES = ["inactive_flag", "price_hike_flag", "bill_unpaid_flag", "renewal_due_60d"]
FLOAT_FEATURES = ["usage_change_pct_m3_to_m1", "unpaid_amount_6m", "discount_pct", "resolution_rate",
                  "charge_change_pct_3m", "net_charge", "last_csat_score"]


def style(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), legend=dict(orientation="h", y=-0.2))
    return fig


def get_api_url():
    url = os.environ.get("CHURN_API_URL", "")
    if not url:
        try:
            url = st.secrets.get("CHURN_API_URL", "")
        except Exception:  # no secrets.toml present
            url = ""
    return url.rstrip("/")


# ---------- cached data ----------

@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def read_report(name):
    path = REPORTS_DIR / name
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


@st.cache_data
def scoring_population():
    """Most recent period (test snapshots, Nov-Dec 2025) scored with the saved model.

    Labels are dropped here: this table represents the forward-looking view. Labelled evaluation
    uses reports/predictions_test.csv instead.
    """
    bundle = get_bundle()
    df = load_split("test").drop(columns=[TARGET])
    scored = score_dataframe(df, bundle)
    scored["top_risk_factors"] = top_factor_text(df, bundle)
    return add_billing_segment(scored)


def score_rows(df):
    """Score rows through the API when configured, otherwise with local shared functions."""
    bundle = get_bundle()
    api_url = get_api_url()
    if api_url:
        try:
            payload = {"customers": json.loads(df[[c for c in df.columns if c in FEATURES + [ID_COL]]]
                                               .to_json(orient="records"))}
            response = requests.post(f"{api_url}/predict-batch", json=payload, timeout=20)
            if response.status_code == 200:
                preds = pd.DataFrame(response.json()["predictions"])
                scored, errors = validate_and_score(df, bundle)  # needed locally for SHAP plots
                if errors:
                    return None, errors, "api"
                scored["churn_probability"] = preds["churn_probability"].to_numpy()
                scored["raw_model_probability"] = preds["raw_model_probability"].to_numpy()
                scored["risk_category"] = preds["risk_category"].to_numpy()
                return scored, [], "api"
            return None, [f"API returned {response.status_code}: {response.text[:500]}"], "api"
        except requests.RequestException as exc:
            st.info(f"API at {api_url} not reachable ({exc.__class__.__name__}); using local model instead.")
    scored, errors = validate_and_score(df, bundle)
    return scored, errors, "local"


def show_local_explanation(row_df, raw_prob, calibrated_prob):
    bundle = get_bundle()
    table, base_value = local_explanation(row_df, bundle)
    st.write(plain_language_summary(table, raw_prob, calibrated_prob))
    left, right = st.columns([3, 2])
    with left:
        fig = waterfall_figure(table, base_value)
        st.pyplot(fig, clear_figure=True)
        plt.close("all")
    with right:
        shown = table.head(10).copy()
        shown["value"] = shown["value"].astype(str)
        st.dataframe(shown.round(3), hide_index=True, width="stretch")
    st.caption("SHAP values are contributions to the log-odds of the raw (uncalibrated) model, measured against "
               "an average training customer. They are not percentage-point changes in the calibrated probability "
               "and they do not prove causation.")


# ---------- pages ----------

def page_overview():
    st.header("Overview")
    bundle = get_bundle()
    pop = scoring_population()
    st.caption(f"Scoring population: {len(pop)} customer snapshots from the most recent period "
               f"({pop['snapshot_date'].min():%b %Y} - {pop['snapshot_date'].max():%b %Y}). "
               f"Risk categories: High >= {bundle['threshold']:.2f} (operating threshold chosen on validation), "
               f"Medium >= {bundle['train_churn_rate']:.2f} (training churn rate), otherwise Low.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligible customers", len(pop))
    c2.metric("High-risk customers", int((pop["risk_category"] == "High").sum()))
    c3.metric("Average predicted risk", f"{pop['churn_probability'].mean():.1%}")
    c4.metric("Model", MODEL_LABELS[bundle["model_name"]])

    left, right = st.columns(2)
    with left:
        counts = pop["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        counts.columns = ["risk_category", "customers"]
        fig = px.bar(counts, x="risk_category", y="customers", color="risk_category",
                     color_discrete_map=RISK_COLORS, text="customers", title="Customers by risk category")
        fig.update_traces(textposition="outside", showlegend=False)
        st.plotly_chart(style(fig), width="stretch")
    with right:
        fig = px.histogram(pop, x="churn_probability", nbins=20, title="Distribution of predicted churn probability",
                           color_discrete_sequence=[BLUE])
        fig.add_vline(x=bundle["threshold"], line_dash="dash", line_color=GREY,
                      annotation_text=f"threshold {bundle['threshold']:.2f}")
        fig.update_layout(bargap=0.05, xaxis_title="Churn probability", yaxis_title="Customers")
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("Historical churn trend")
    monthly = read_report("eda_monthly_churn.csv")
    fig = px.bar(monthly, x="snapshot_date", y="churn_rate", color="split", text=monthly["churners"],
                 color_discrete_map={"train": "#b7d3f2", "validation": "#5b9be3", "test": "#1a4f8f"},
                 hover_data=["customers", "churners"], title="Observed 30-day churn rate by monthly snapshot")
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Snapshot (month end)", yaxis_title="Churn rate")
    st.plotly_chart(style(fig), width="stretch")
    st.caption("Labelled history across the train, validation and test periods (number of churners on each bar). "
               "Each month contains different customers, so this is not a cohort trend.")

    st.subheader("Segments in the scoring population")
    segment = st.selectbox("Segment", CATEGORICAL_FEATURES + ["billing_segment"], index=0)
    seg = pop.groupby(segment).agg(customers=(ID_COL, "size"), avg_risk=("churn_probability", "mean"),
                                   high_risk=("risk_category", lambda s: int((s == "High").sum()))).reset_index()
    left, right = st.columns(2)
    with left:
        fig = px.bar(seg, x=segment, y="customers", text="customers", title="Customers", color_discrete_sequence=[BLUE])
        st.plotly_chart(style(fig, 320), width="stretch")
    with right:
        fig = px.bar(seg, x=segment, y="avg_risk", text=seg["avg_risk"].map("{:.0%}".format),
                     title="Average predicted churn probability", color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(style(fig, 320), width="stretch")


def page_prediction():
    st.header("Churn prediction")
    bundle = get_bundle()
    api_url = get_api_url()
    st.caption(f"Prediction source: {'FastAPI at ' + api_url if api_url else 'local model (shared src/predict.py)'}.")
    st.info("The model uses 24 engineered features, not raw account fields. The form starts from the training "
            "median (numbers) or most common value (categories) for every feature; change the ones you know.")
    defaults = bundle["training_defaults"]

    manual, upload = st.tabs(["Enter one customer", "Upload CSV"])
    with manual:
        with st.form("manual_form"):
            values = {}
            cols = st.columns(3)
            for i, feat in enumerate(CATEGORICAL_FEATURES + NUMERIC_FEATURES):
                col = cols[i % 3]
                if feat in CATEGORICAL_FEATURES:
                    options = CATEGORY_VALUES[feat]
                    values[feat] = col.selectbox(feat, options, index=options.index(defaults[feat]))
                elif feat in BINARY_FEATURES:
                    values[feat] = col.selectbox(feat, [0, 1], index=int(defaults[feat]))
                else:
                    low, high = NUMERIC_LIMITS[feat]
                    if feat in FLOAT_FEATURES:
                        values[feat] = col.number_input(feat, float(low), float(high), float(defaults[feat]))
                    else:
                        values[feat] = col.number_input(feat, int(low), int(high), int(defaults[feat]), step=1)
            submitted = st.form_submit_button("Predict churn risk", type="primary")
        if submitted:
            row = pd.DataFrame([{ID_COL: "manual_entry", **values}])
            scored, errors, source = score_rows(row)
            if errors:
                st.error("\n".join(errors))
            else:
                r = scored.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Churn probability (30 days)", f"{r['churn_probability']:.1%}")
                c2.metric("Risk category", r["risk_category"])
                c3.metric("Flagged at threshold", "Yes" if r["churn_probability"] >= bundle["threshold"] else "No")
                st.caption(f"Scored by: {source}. Raw model output before calibration: {r['raw_model_probability']:.1%}.")
                show_local_explanation(scored.iloc[[0]], r["raw_model_probability"], r["churn_probability"])

    with upload:
        st.write(f"Upload a CSV with the 24 model feature columns (and optionally `{ID_COL}`). "
                 "`last_csat_score` may be blank; it is filled with the training median.")
        template = pd.DataFrame([{ID_COL: "CUST_EXAMPLE", **{f: defaults[f] for f in FEATURES}}])
        st.download_button("Download template CSV", template.to_csv(index=False), "churn_input_template.csv", "text/csv")
        file = st.file_uploader("Customer CSV", type="csv")
        if file is not None:
            try:
                uploaded = pd.read_csv(file)
            except Exception as exc:
                st.error(f"Could not read the file as CSV: {exc}")
                return
            scored, errors, source = score_rows(uploaded)
            if errors:
                st.error("The file cannot be scored:\n\n- " + "\n- ".join(errors))
                return
            if ID_COL not in scored.columns:
                scored[ID_COL] = [f"row_{i}" for i in range(len(scored))]
            scored["top_risk_factors"] = top_factor_text(scored, bundle)
            scored = scored.sort_values("churn_probability", ascending=False)
            st.success(f"Scored {len(scored)} customers ({source}). "
                       f"{int((scored['risk_category'] == 'High').sum())} are High risk.")
            show_cols = [ID_COL, "churn_probability", "risk_category", "top_risk_factors"]
            st.dataframe(scored[show_cols].round(4), hide_index=True, width="stretch")
            st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_customers.csv", "text/csv")
            chosen = st.selectbox("Explain customer", scored[ID_COL].astype(str).tolist())
            row = scored[scored[ID_COL].astype(str) == chosen].head(1)
            show_local_explanation(row, row["raw_model_probability"].iloc[0], row["churn_probability"].iloc[0])


def page_prioritization():
    st.header("Retention prioritization")
    bundle = get_bundle()
    pop = scoring_population()
    st.caption("Ranked review list for the most recent period. Changing the threshold or capacity re-ranks "
               "existing scores; the model is not retrained. Selection means 'send to a person for review', "
               "not an automatic retention action.")

    c1, c2, c3 = st.columns([2, 1, 1])
    threshold = c1.slider("Risk threshold (probability)", 0.0, 1.0, float(bundle["threshold"]), 0.01)
    mode = c2.radio("Review capacity", ["Top-K customers", "Top % of customers", "No capacity limit"])
    top_k = top_pct = None
    if mode == "Top-K customers":
        top_k = c3.number_input("K", 1, len(pop), min(30, len(pop)), step=1)
    elif mode == "Top % of customers":
        top_pct = c3.number_input("Percent", 1, 100, 20, step=1) / 100

    queue = build_retention_queue(pop, threshold, bundle["train_churn_rate"], top_k=top_k, top_pct=top_pct,
                                  top_factors=pop["top_risk_factors"].to_numpy())
    selected = queue[queue["selected_for_review"]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers flagged for review", len(selected))
    m2.metric("Above threshold", int((queue["churn_probability"] >= threshold).sum()))
    m3.metric("Average risk of flagged", f"{selected['churn_probability'].mean():.1%}" if len(selected) else "-")

    cols = [ID_COL, "risk_rank", "churn_probability", "risk_category", "segment", "top_risk_factors",
            "usage_change_pct_m3_to_m1", "days_since_login", "billing_risk", "service_pain_score", "selected_for_review"]
    only_selected = st.checkbox("Show only customers selected for review", value=True)
    shown = selected if only_selected else queue
    st.dataframe(shown[cols].round(3), hide_index=True, width="stretch", height=420,
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "churn_probability", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download prioritized list (CSV)", queue.to_csv(index=False), "retention_queue.csv", "text/csv")

    with st.expander("How would these settings have performed historically? (labelled test period)"):
        pred = read_report("predictions_test.csv")
        flagged_ids = set(selected[ID_COL])
        y = pred[TARGET].to_numpy()
        flagged = pred[ID_COL].isin(flagged_ids).to_numpy()
        tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        a, b, c, d = st.columns(4)
        a.metric("True positives", tp)
        b.metric("False positives", fp)
        c.metric("Missed churners", fn)
        d.metric("Churn captured", f"{tp / max(y.sum(), 1):.0%}")
        st.caption("These customers' real outcomes are known because this period is the held-out test set. "
                   "For genuinely new customers, outcomes are unknown and only the ranked list above applies.")


def page_performance():
    st.header("Model performance")
    bundle = get_bundle()
    selection = read_report("model_selection.json")
    st.write(f"**Selected model:** {MODEL_LABELS[bundle['model_name']]}. {selection['reason']}")

    st.subheader("Model comparison (train vs validation)")
    comparison = read_report("model_comparison.csv")
    cols = ["model", "split", "pr_auc", "roc_auc", "brier", "precision_at_top20pct", "recall_at_top20pct",
            "lift_at_top20pct", "precision", "recall", "f1"]
    st.dataframe(comparison[cols].round(3), hide_index=True, width="stretch")
    st.caption("Source: reports/model_comparison.csv. Raw (uncalibrated) probabilities; precision/recall/F1 at 0.5. "
               "Top-20% metrics use the 20% highest-scored customers. A large train-validation gap means overfitting: "
               "CatBoost's PR-AUC drops more from train to validation than Logistic Regression's.")
    with st.expander("All tuning runs (validation used for hyperparameter choice)"):
        st.dataframe(read_report("tuning_results.csv").round(3), hide_index=True, width="stretch")

    dataset = st.radio("Dataset for the charts below",
                       ["Validation (Sep-Oct 2025, used for selection, calibration and threshold)",
                        "Test (Nov-Dec 2025, evaluated once after all decisions were frozen)"], horizontal=True)
    is_test = dataset.startswith("Test")
    pred = read_report("predictions_test.csv" if is_test else "predictions_validation.csv")
    y = pred[TARGET].to_numpy()
    p_final = pred["churn_probability"].to_numpy()
    st.caption(f"{len(y)} customers, {int(y.sum())} churners. With so few churners every metric has wide uncertainty.")

    if is_test:
        ci = read_report("test_bootstrap_ci.csv")
        st.dataframe(ci.round(3), hide_index=True, width="stretch")
        st.caption("Bootstrap 95% confidence intervals (1,000 resamples of the test set), selected model.")

    left, right = st.columns(2)
    curves = {MODEL_LABELS[n]: pred[f"raw_{n}"].to_numpy() for n in ["logistic_regression", "catboost"]}
    pr_fig, roc_fig = go.Figure(), go.Figure()
    for (label, p), color in zip(curves.items(), [BLUE, ORANGE]):
        precision, recall, _ = precision_recall_curve(y, p)
        pr_fig.add_scatter(x=recall, y=precision, name=f"{label} (AP {average_precision_score(y, p):.3f})",
                           line=dict(color=color, width=2))
        fpr, tpr, _ = roc_curve(y, p)
        roc_fig.add_scatter(x=fpr, y=tpr, name=f"{label} (AUC {roc_auc_score(y, p):.3f})", line=dict(color=color, width=2))
    pr_fig.add_hline(y=y.mean(), line_dash="dash", line_color=GREY, annotation_text="no skill")
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="random")
    pr_fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
    roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    left.plotly_chart(style(pr_fig, 400), width="stretch")
    right.plotly_chart(style(roc_fig, 400), width="stretch")
    if is_test:
        st.caption("Both models are shown on test for transparency only; the choice was made on validation beforehand.")

    st.subheader("Selected model at a chosen threshold")
    threshold = st.slider("Threshold", 0.05, 0.95, float(bundle["threshold"]), 0.01, key="perf_threshold")
    m = threshold_metrics(y, p_final, threshold)
    left, right = st.columns([1, 2])
    with left:
        cm = np.array([[len(y) - m["flagged"] - m["false_negatives"], m["false_positives"]],
                       [m["false_negatives"], m["true_positives"]]])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted stay", "Predicted churn"], y=["Actual stay", "Actual churn"], title="Confusion matrix")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 340), width="stretch")
    with right:
        a, b, c, d = st.columns(4)
        a.metric("Flagged", m["flagged"])
        b.metric("Precision", f"{m['precision']:.2f}")
        c.metric("Recall", f"{m['recall']:.2f}")
        d.metric("F1", f"{m['f1']:.2f}")
        table = threshold_table(y, p_final)
        fig = go.Figure()
        fig.add_scatter(x=table["threshold"], y=table["precision"], name="Precision", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=table["threshold"], y=table["recall"], name="Recall", line=dict(color=ORANGE, width=2))
        fig.add_vline(x=threshold, line_dash="dash", line_color=GREY)
        fig.update_layout(title="Threshold trade-off", xaxis_title="Threshold", yaxis_title="Score")
        st.plotly_chart(style(fig, 280), width="stretch")
    with st.expander("Threshold trade-off table (flagged volume, TP, FP, FN)"):
        st.dataframe(table.round(3), hide_index=True, width="stretch")

    st.subheader("Calibration")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="perfect")
        calibrated_col = "churn_probability" if is_test else "calibrated_out_of_fold"
        for label, col, color in [("Raw model", "raw_model_probability", ORANGE), ("Calibrated", calibrated_col, BLUE)]:
            ct = calibration_table(y, pred[col].to_numpy())
            brier = np.mean((pred[col].to_numpy() - y) ** 2)
            fig.add_scatter(x=ct["mean_predicted"], y=ct["observed_rate"], mode="lines+markers",
                            name=f"{label} (Brier {brier:.3f})", line=dict(color=color, width=2), marker=dict(size=9))
        fig.update_layout(title="Reliability diagram (5 equal-size bins)", xaxis_title="Mean predicted probability",
                          yaxis_title="Observed churn rate", xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(style(fig, 400), width="stretch")
    with right:
        st.write(f"- Calibration applied: **{selection['calibration_applied']}** (sigmoid / Platt scaling fitted on "
                 "validation predictions of the frozen model).")
        st.write(f"- Validation Brier, raw: **{selection['validation_brier_raw']:.3f}**; calibrated, "
                 f"5-fold cross-validated inside validation: **{selection['validation_brier_calibrated_cv']:.3f}**.")
        st.write(f"- Mean predicted on validation, raw: {selection['validation_mean_predicted_raw']:.3f} vs observed "
                 f"{selection['validation_observed_churn_rate']:.3f}. The raw model is over-confident about churn "
                 "because it was trained with balanced class weights.")
        if is_test:
            st.write(f"- Test mean predicted (calibrated) {p_final.mean():.3f} vs observed {y.mean():.3f}. "
                     "Test churn rate is lower than validation, so some over-prediction remains.")
        else:
            st.write("- On validation the 'Calibrated' line uses out-of-fold predictions, so it is not scored on the "
                     "rows the calibrator was fitted on.")
        st.write("- Calibration does not change the ranking, so PR-AUC, ROC-AUC and top-K metrics are unchanged.")

    st.subheader("Lift and cumulative gain (selected model)")
    left, right = st.columns(2)
    gain = gain_curve(y, p_final)
    fig = go.Figure()
    fig.add_scatter(x=gain["share_reviewed"], y=gain["churn_captured"], name="Model", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="Random")
    fig.update_layout(title="Cumulative gain", xaxis_title="Share of customers reviewed",
                      yaxis_title="Share of churners captured", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    left.plotly_chart(style(fig, 380), width="stretch")
    caps = capacity_table(y, p_final)
    fig = px.bar(caps, x=caps["share_reviewed"].map("{:.0%}".format), y="lift_at_k", text=caps["lift_at_k"].round(2),
                 title="Lift at review capacity", color_discrete_sequence=[BLUE])
    fig.add_hline(y=1, line_dash="dash", line_color=GREY)
    fig.update_layout(xaxis_title="Top share reviewed", yaxis_title="Lift vs random")
    right.plotly_chart(style(fig, 380), width="stretch")
    st.dataframe(caps.round(3), hide_index=True, width="stretch")


def page_explainability():
    st.header("Model explainability & segments")
    bundle = get_bundle()
    pop = scoring_population()

    st.subheader("Global SHAP feature importance (test period)")
    importance = read_report("shap_global_importance.csv").head(15).iloc[::-1]
    left, right = st.columns(2)
    fig = px.bar(importance, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=[BLUE],
                 title="Mean |SHAP| (log-odds of raw model)")
    fig.update_layout(xaxis_title="Mean absolute SHAP value", yaxis_title="")
    left.plotly_chart(style(fig, 480), width="stretch")
    beeswarm = FIGURES_DIR / "shap_beeswarm_test.png"
    if beeswarm.exists():
        right.image(str(beeswarm), caption="SHAP summary: each dot is a test customer; colour shows the feature value "
                                           "(grey = categorical).")
    top5 = importance.iloc[::-1].head(5)["feature"].tolist()
    st.caption(f"Strongest signals in this period: {', '.join(top5)}. These are associations the model uses, "
               "not proven causes.")

    st.subheader("Customer-level explanation")
    options = pop.sort_values("churn_probability", ascending=False)[ID_COL].tolist()
    chosen = st.selectbox("Customer ID (sorted by risk)", options)
    row = pop[pop[ID_COL] == chosen].head(1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{row['churn_probability'].iloc[0]:.1%}")
    c2.metric("Risk category", row["risk_category"].iloc[0])
    c3.metric("Segment", f"{row['contract_type'].iloc[0]} | {row['tenure_band'].iloc[0]}")
    show_local_explanation(row, row["raw_model_probability"].iloc[0], row["churn_probability"].iloc[0])

    st.subheader("Segmented performance (test period, operating threshold)")
    seg = read_report("test_segment_performance.csv")
    segment = st.selectbox("Segment", seg["segment"].unique().tolist())
    part = seg[seg["segment"] == segment]
    fig = go.Figure()
    fig.add_bar(x=part["segment_value"].astype(str), y=part["recall"], name="Recall", marker_color=BLUE)
    fig.add_bar(x=part["segment_value"].astype(str), y=part["precision"], name="Precision", marker_color=ORANGE)
    fig.update_layout(barmode="group", title=f"Precision and recall by {segment}", yaxis_range=[0, 1])
    st.plotly_chart(style(fig, 340), width="stretch")
    st.dataframe(part.drop(columns=["segment"]).round(3), hide_index=True, width="stretch")
    st.caption("Segments with fewer than 30 customers or 5 churners are marked; their metrics can swing a lot "
               "from one or two customers and should not be used to draw conclusions.")

    st.subheader("Limitations")
    st.markdown("""
- **Synthetic data.** All 1,020 customer snapshots were generated for this hackathon, and the churn label came from a
  logistic risk formula. Real-world performance would be different, and the strong signals reflect the generator.
- **Small evaluation sets.** Validation has 43 churners and test has 32, so the confidence intervals are wide
  (test PR-AUC 95% CI shown on the performance page).
- **Model choice and calibration share the validation set.** Hyperparameters, the model, the calibrator and the
  threshold were all chosen on the same 172 validation rows. The test set was used once, at the end.
- **Short-tenure history.** 21 customers with under 90 days' tenure have usage recorded three months back (before
  signup), an artefact of the generator. The rows were kept unchanged; `tenure_days` is not a model input.
- **Imputed CSAT.** 49 missing or invalid CSAT scores were set to 4.0 when the files were prepared, and the model
  cannot tell them apart from real 4.0 scores.
- **Feature-generation code not available.** 5 `usage_change_pct_m3_to_m1` values cannot be rebuilt from the raw
  file. They were left as provided.
- **One snapshot per customer.** Each customer appears once, so the model has no repeated history per customer.
- **Decision support only.** The tool ranks customers for human review. It does not show that any customer was
  retained, and SHAP explanations are not causal.
""")


PAGES = {
    "Overview": page_overview,
    "Churn prediction": page_prediction,
    "Retention prioritization": page_prioritization,
    "Model performance": page_performance,
    "Explainability & segments": page_explainability,
}

st.sidebar.title("Churn Risk & Retention")
st.sidebar.caption("HCLTech hackathon prototype - 30-day churn prediction")
choice = st.sidebar.radio("Page", list(PAGES))
try:
    get_bundle()
    PAGES[choice]()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nRun the pipeline first (see README): python -m src.train, then python -m src.evaluate, "
             "python -m src.explain and python -m src.prioritize.")
