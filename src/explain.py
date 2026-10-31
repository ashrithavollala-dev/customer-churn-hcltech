"""SHAP explanations for the selected model.

Run from the repository root:  python -m src.explain

Important when reading the numbers:
- SHAP values here are in LOG-ODDS of the RAW (uncalibrated) model. They add up exactly to the raw
  model's log-odds for that customer: base_value + sum(contributions) = log-odds of raw probability.
- They are NOT percentage-point changes in the calibrated probability. Calibration is a monotonic
  transform, so a factor that raises the log-odds also raises the calibrated probability, but the size
  in probability terms depends on where the customer starts.
- SHAP shows what the model relies on, not what causes churn.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool

from src.config import CATEGORICAL_FEATURES, FEATURES, FIGURES_DIR, ID_COL, NUMERIC_FEATURES, REPORTS_DIR


def _original_feature(encoded_name):
    """Map a ColumnTransformer output name such as 'cat__contract_type_One year' to 'contract_type'."""
    name = encoded_name.split("__", 1)[1]
    for col in CATEGORICAL_FEATURES:
        if name.startswith(col + "_"):
            return col
    return name


def shap_values_for(df, model, model_name, background):
    """Return (DataFrame of SHAP values with one column per original feature, base value)."""
    X = df[FEATURES]
    if model_name == "logistic_regression":
        preprocess = model.named_steps["preprocess"]
        linear = model.named_steps["model"]
        explainer = shap.LinearExplainer(linear, preprocess.transform(background[FEATURES]))
        values = explainer.shap_values(preprocess.transform(X))
        encoded_names = preprocess.get_feature_names_out()
        encoded = pd.DataFrame(values, columns=encoded_names, index=df.index)
        # A one-hot encoded category is one feature to the user, so its dummy columns are summed.
        # This is valid because SHAP values are additive.
        grouped = encoded.T.groupby([_original_feature(c) for c in encoded_names]).sum().T
        return grouped[FEATURES], float(np.ravel(explainer.expected_value)[0])

    if model_name == "catboost":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Pool(X, cat_features=CATEGORICAL_FEATURES))
        return pd.DataFrame(values, columns=FEATURES, index=df.index), float(np.ravel(explainer.expected_value)[0])

    raise ValueError(f"No SHAP explainer configured for model '{model_name}'")


def explain_dataframe(df, bundle):
    return shap_values_for(df, bundle["model"], bundle["model_name"], bundle["background_sample"])


def global_importance(shap_df):
    importance = shap_df.abs().mean().sort_values(ascending=False)
    direction = shap_df.mean()
    return pd.DataFrame({
        "feature": importance.index,
        "mean_abs_shap": importance.values,
        "mean_shap": direction[importance.index].values,
    })


def _format_value(value):
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def local_explanation(row_df, bundle):
    """Contribution table for a single customer, largest effect first."""
    shap_df, base_value = explain_dataframe(row_df, bundle)
    contributions = shap_df.iloc[0]
    table = pd.DataFrame({
        "feature": FEATURES,
        "value": [row_df.iloc[0][f] for f in FEATURES],
        "shap_log_odds": contributions[FEATURES].values,
    })
    table["effect"] = np.where(table["shap_log_odds"] > 0, "increases risk", "decreases risk")
    table = table.reindex(table["shap_log_odds"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return table, base_value


def top_factor_text(df, bundle, n_factors=3):
    """For each row: the n features that push the prediction up the most, as short text."""
    shap_df, _ = explain_dataframe(df, bundle)
    texts = []
    for idx in shap_df.index:
        row = shap_df.loc[idx]
        positive = row[row > 0].sort_values(ascending=False).head(n_factors)
        parts = [f"{feat}={_format_value(df.loc[idx, feat])}" for feat in positive.index]
        texts.append("; ".join(parts) if parts else "no factor raises risk above average")
    return texts


def plain_language_summary(table, raw_probability, calibrated_probability, n=3):
    """Short explanation built from the actual SHAP table (nothing is hard-coded per feature)."""
    up = table[table["shap_log_odds"] > 0].head(n)
    down = table[table["shap_log_odds"] < 0].head(n)
    lines = [f"Estimated churn probability: {calibrated_probability:.1%} "
             f"(raw model output before calibration: {raw_probability:.1%})."]
    if len(up):
        lines.append("Pushing the estimate up: " + ", ".join(f"{r.feature} = {_format_value(r.value)}" for r in up.itertuples()) + ".")
    if len(down):
        lines.append("Pulling the estimate down: " + ", ".join(f"{r.feature} = {_format_value(r.value)}" for r in down.itertuples()) + ".")
    lines.append("These are the factors the model relied on for this customer compared with an average training "
                 "customer. They are associations learned from data, not proven causes of churn.")
    return " ".join(lines)


def waterfall_figure(table, base_value, max_display=10):
    """SHAP waterfall plot (log-odds of the raw model) for one customer."""
    ordered = table.set_index("feature").loc[FEATURES]
    data = [v if f in NUMERIC_FEATURES else str(v) for f, v in zip(FEATURES, ordered["value"])]
    explanation = shap.Explanation(
        values=ordered["shap_log_odds"].to_numpy(dtype=float), base_values=base_value,
        data=np.array(data, dtype=object), feature_names=FEATURES,
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Contribution to log-odds of the raw (uncalibrated) model")
    fig.set_size_inches(8, 5.5)
    fig.tight_layout()
    return fig


def _numeric_matrix(df):
    """Feature values for colouring the beeswarm; categorical columns become NaN (shown grey)."""
    return np.column_stack([
        pd.to_numeric(df[f], errors="coerce").to_numpy() if f in NUMERIC_FEATURES else np.full(len(df), np.nan)
        for f in FEATURES
    ])


def main():
    from src.predict import load_bundle, score_dataframe
    from src.preprocessing import load_split

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    test = load_split("test")
    shap_df, base_value = explain_dataframe(test, bundle)

    # Sanity check: for logistic regression, base + sum of SHAP must equal the raw log-odds.
    raw = score_dataframe(test, bundle)["raw_model_probability"].to_numpy()
    reconstructed = 1 / (1 + np.exp(-(base_value + shap_df.sum(axis=1).to_numpy())))
    max_error = float(np.max(np.abs(reconstructed - raw)))
    print(f"SHAP additivity check (max |reconstructed - raw probability|): {max_error:.2e}")

    importance = global_importance(shap_df)
    importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    out = shap_df.copy()
    out.insert(0, ID_COL, test[ID_COL].values)
    out.to_csv(REPORTS_DIR / "shap_values_test.csv", index=False)

    explanation = shap.Explanation(values=shap_df.to_numpy(), base_values=np.full(len(shap_df), base_value),
                                   data=_numeric_matrix(test), feature_names=FEATURES)
    plt.figure()
    with warnings.catch_warnings():
        # Categorical columns have no numeric colour value (NaN); SHAP draws them grey and numpy warns.
        warnings.simplefilter("ignore", RuntimeWarning)
        shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.gcf().set_size_inches(9, 6.5)
    plt.title("SHAP summary - test period (log-odds of raw model)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm_test.png", dpi=120)
    plt.close("all")

    plt.figure()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.title("Mean |SHAP| - test period")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_importance_test.png", dpi=120)
    plt.close("all")

    # Example local explanation: the highest-risk test customer.
    top_idx = int(np.argmax(raw))
    table, base = local_explanation(test.iloc[[top_idx]], bundle)
    fig = waterfall_figure(table, base)
    fig.savefig(FIGURES_DIR / "shap_waterfall_example.png", dpi=120)
    plt.close("all")
    print(f"Example customer {test.iloc[top_idx][ID_COL]}:")
    print(table.head(8).to_string(index=False))
    print("\nTop global signals:")
    print(importance.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
