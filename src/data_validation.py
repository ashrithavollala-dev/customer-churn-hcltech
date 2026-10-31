"""Stage 1: data quality, chronology and leakage checks.

Run from the repository root:  python -m src.data_validation
Writes reports/data_validation_report.md plus small EDA tables used by the dashboard.

The raw file (data/raw/customer_churn_raw_1050.csv) is used ONLY to verify how engineered
features were built and to check the short-tenure issue. Nothing is trained on it.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    CATEGORICAL_FEATURES, DATE_COL, FEATURES, ID_COL, NUMERIC_FEATURES, RAW_FILE, REPORTS_DIR, TARGET,
)
from src.preprocessing import add_billing_segment, load_all_splits


def check_splits(splits):
    """Basic structure, missing values, duplicates, chronology and class balance."""
    lines = ["## 1. Split structure and quality", ""]
    columns = [list(df.columns) for df in splits.values()]
    lines.append(f"- Same columns in all three files: **{all(c == columns[0] for c in columns)}** "
                 f"({len(columns[0])} columns)")
    model_cols = [c for c in columns[0] if c not in (ID_COL, DATE_COL, TARGET)]
    lines.append(f"- Candidate feature columns: {len(model_cols)}; match the 24 configured features: "
                 f"**{sorted(model_cols) == sorted(FEATURES)}**")
    lines.append(f"- Excluded from model inputs: `{ID_COL}`, `{DATE_COL}`, `{TARGET}` (kept for reporting)")
    lines += ["", "| Split | Rows | Snapshot range | Churners | Churn rate | Missing cells | Duplicate rows | Duplicate customer+date |",
              "|---|---|---|---|---|---|---|---|"]
    for name, df in splits.items():
        lines.append(
            f"| {name} | {len(df)} | {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()} | "
            f"{int(df[TARGET].sum())} | {df[TARGET].mean():.3f} | {int(df.isna().sum().sum())} | "
            f"{int(df.duplicated().sum())} | {int(df.duplicated([ID_COL, DATE_COL]).sum())} |"
        )
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    chronological = tr[DATE_COL].max() < va[DATE_COL].min() and va[DATE_COL].max() < te[DATE_COL].min()
    lines.append("")
    lines.append(f"- Strictly chronological (train < validation < test): **{chronological}**")
    ids = {k: set(v[ID_COL]) for k, v in splits.items()}
    lines.append(f"- Customers shared between splits: train-validation {len(ids['train'] & ids['validation'])}, "
                 f"train-test {len(ids['train'] & ids['test'])}, validation-test {len(ids['validation'] & ids['test'])}")
    lines.append(f"- Each customer appears once in the whole dataset: "
                 f"**{sum(len(v) for v in ids.values()) == len(set.union(*ids.values()))}**")

    lines += ["", "Unseen categories in validation/test compared with train:"]
    for col in CATEGORICAL_FEATURES:
        unseen = (set(va[col]) | set(te[col])) - set(tr[col])
        lines.append(f"- `{col}`: {sorted(unseen) if unseen else 'none'}")
    return lines


def check_single_feature_leakage(train):
    """A single feature that almost perfectly separates churners is a typical sign of target leakage."""
    lines = ["", "## 2. Leakage screen (train only)", "",
             "Single-feature ROC-AUC on the training split. Values close to 1.0 would suggest a feature "
             "that encodes the outcome. Categorical features use their training churn rate as the score.", "",
             "| Feature | Single-feature ROC-AUC |", "|---|---|"]
    rows = []
    for col in FEATURES:
        if col in NUMERIC_FEATURES:
            score = train[col]
        else:
            score = train[col].map(train.groupby(col)[TARGET].mean())
        auc = roc_auc_score(train[TARGET], score)
        rows.append((col, max(auc, 1 - auc)))
    for col, auc in sorted(rows, key=lambda r: -r[1]):
        lines.append(f"| {col} | {auc:.3f} |")
    worst = max(r[1] for r in rows)
    lines.append("")
    lines.append(f"Highest single-feature AUC is {worst:.3f}; "
                 + ("no feature looks like a direct copy of the label." if worst < 0.9
                    else "**this needs investigation**."))
    return lines


def check_against_raw(splits):
    """Rebuild engineered features from the raw file and compare with the final files."""
    lines = ["", "## 3. Engineered features rebuilt from the raw file", ""]
    if not RAW_FILE.exists():
        return lines + ["Raw file not found, so these checks were skipped."]

    raw = pd.read_csv(RAW_FILE)
    lines.append(f"- Raw file: {len(raw)} rows, {int(raw.duplicated().sum())} exact duplicate rows "
                 f"(removed before the final files were built, giving {len(raw.drop_duplicates())} rows).")
    raw = raw.drop_duplicates()
    final = pd.concat(splits.values())
    final[DATE_COL] = final[DATE_COL].dt.strftime("%Y-%m-%d")
    m = final.merge(raw, on=[ID_COL, DATE_COL], suffixes=("", "_raw"))
    lines.append(f"- Rows matched between final files and raw file: {len(m)} of {len(final)}")
    lines.append(f"- Raw missing values: {raw.isna().sum()[raw.isna().sum() > 0].to_dict()}")

    snap = pd.to_datetime(m[DATE_COL])
    for col in ["last_login_date", "last_payment_date", "last_bill_due_date", "signup_date"]:
        after = (pd.to_datetime(m[col]) > snap).sum()
        lines.append(f"- `{col}` later than snapshot_date: {int(after)} rows")

    tickets_90d = m.support_tickets_m1 + m.support_tickets_m2 + m.support_tickets_m3
    rebuilt = {
        "billing_risk = late_payments_last_6m + billing_disputes_last_6m":
            (m.late_payments_last_6m + m.billing_disputes_last_6m) == m.billing_risk,
        "stickiness = num_addon_services + autopay + (contract is not month-to-month)":
            (m.num_addon_services + (m.payment_method == "Autopay") + (m.contract_type != "Month-to-month"))
            == m.stickiness,
        "service_pain_score = 2*complaints + unresolved tickets + (CSAT <= 2)":
            (2 * m.complaints_last_90d + (tickets_90d - m.tickets_resolved_last_90d).clip(lower=0)
             + (m.last_csat_score <= 2).astype(int)) == m.service_pain_score,
        "usage_change_pct_m3_to_m1 = (m1 - m3) / (m3 + 1) * 100":
            np.isclose((m.usage_hours_m1 - m.usage_hours_m3) / (m.usage_hours_m3 + 1) * 100,
                       m.usage_change_pct_m3_to_m1, atol=1e-4),
        "resolution_rate = resolved / tickets (1.0 when no tickets)":
            np.isclose(np.where(tickets_90d > 0, m.tickets_resolved_last_90d / tickets_90d.replace(0, 1), 1.0),
                       m.resolution_rate),
        "inactive_flag = days_since_login > 30": (m.days_since_login > 30).astype(int) == m.inactive_flag,
        "days_since_login = snapshot_date - last_login_date":
            (snap - pd.to_datetime(m.last_login_date)).dt.days == m.days_since_login,
        "unpaid_amount_6m = max(total_billed - total_paid, 0)":
            np.isclose((m.total_billed_last_6m - m.total_paid_last_6m).clip(lower=0), m.unpaid_amount_6m),
        "bill_unpaid_flag = no last_payment_date recorded": m.last_payment_date.isna().astype(int) == m.bill_unpaid_flag,
        "renewal_due_60d = contract ends within 60 days of snapshot":
            (pd.to_datetime(m.contract_end_date) - snap).dt.days.between(0, 60).astype(int) == m.renewal_due_60d,
        "region / acquisition_channel = 'Unknown' where raw value missing":
            (m.region_raw.isna() == (m.region == "Unknown")) & (m.acquisition_channel_raw.isna() == (m.acquisition_channel == "Unknown")),
    }
    lines += ["", "| Rule checked | Rows matching |", "|---|---|"]
    for rule, match in rebuilt.items():
        lines.append(f"| {rule} | {int(np.sum(match))} / {len(m)} ({np.mean(match):.1%}) |")

    usage_bad = ~rebuilt["usage_change_pct_m3_to_m1 = (m1 - m3) / (m3 + 1) * 100"]
    lines += ["",
              f"- usage_change mismatches: {int(usage_bad.sum())}; of these {int(m.loc[usage_bad, 'usage_hours_m3'].isna().sum())} "
              "have missing raw usage_hours_m3. The imputation used for these rows is not recorded, so the "
              "remaining mismatches cannot be explained without the original feature-generation code.",
              "- `billing_risk` correlates strongly with `late_payments_last_6m` (it contains it), which explains why "
              "`late_payments_last_6m` was dropped in the |corr| >= 0.9 redundancy step."]

    # CSAT handling
    raw_csat = m.last_csat_score_raw
    out_of_range = raw_csat.notna() & ~raw_csat.between(1, 5)
    lines += ["", "### last_csat_score handling", "",
              f"- Raw CSAT missing: {int(raw_csat.isna().sum())} rows; out of the 1-5 range (e.g. 0, 7, 9): {int(out_of_range.sum())} rows.",
              f"- Final values for those rows: {m.loc[raw_csat.isna() | out_of_range, 'last_csat_score'].value_counts().to_dict()}",
              f"- Raw median CSAT (all rows): {raw.last_csat_score.median()}; train-period median: "
              f"{splits['train']['last_csat_score'].median()}",
              "- Conclusion: missing and invalid CSAT were replaced by 4.0. The median appears to have been taken over "
              "the full raw file rather than train only; because the train-only median is the same value (4.0) this has "
              "no practical effect, but it cannot be confirmed from the code that produced the files.",
              "- No missing-CSAT indicator column exists, so the model cannot tell an imputed 4.0 from a real 4.0."]

    # Short-tenure / history issue
    tenure_days = (snap - pd.to_datetime(m.signup_date)).dt.days
    short = tenure_days < 90
    lines += ["", "### Short-tenure history issue", "",
              f"- Customers with tenure < 90 days: {int(short.sum())}; of these with non-zero `usage_hours_m3` "
              f"(usage recorded before signup): {int((short & (m.usage_hours_m3 > 0)).sum())}.",
              f"- Tenure < 60 days with non-zero `usage_hours_m2`: {int(((tenure_days < 60) & (m.usage_hours_m2 > 0)).sum())}.",
              "- These rows come from the synthetic generator. They are kept unchanged (not deleted or corrected) "
              "and are documented as a limitation. `tenure_days` is not in the final files, so this check is only "
              "possible with the raw file.",
              f"- Short-tenure rows by split: "
              f"{m.loc[short].assign(split=lambda d: np.where(pd.to_datetime(d[DATE_COL]) <= '2025-08-31', 'train', np.where(pd.to_datetime(d[DATE_COL]) <= '2025-10-31', 'validation', 'test'))).split.value_counts().to_dict()}"]
    return lines


def save_eda_tables(splits):
    """Monthly churn trend and churn rate by segment (all labelled history) for the dashboard."""
    all_rows = add_billing_segment(pd.concat([df.assign(split=name) for name, df in splits.items()]))
    monthly = (all_rows.groupby([DATE_COL, "split"])[TARGET].agg(customers="size", churners="sum", churn_rate="mean")
               .reset_index())
    monthly.to_csv(REPORTS_DIR / "eda_monthly_churn.csv", index=False)

    seg_rows = []
    for col in ["contract_type", "tenure_band", "region", "payment_method", "paperless_billing",
                "acquisition_channel", "billing_segment"]:
        g = all_rows.groupby(col)[TARGET].agg(customers="size", churners="sum", churn_rate="mean").reset_index()
        g = g.rename(columns={col: "segment_value"}).assign(segment=col)
        seg_rows.append(g)
    pd.concat(seg_rows)[["segment", "segment_value", "customers", "churners", "churn_rate"]].to_csv(
        REPORTS_DIR / "eda_segment_churn.csv", index=False)
    return ["", "## 4. Exploratory tables", "",
            "- `reports/eda_monthly_churn.csv`: customers and churn rate per monthly snapshot.",
            "- `reports/eda_segment_churn.csv`: churn rate per segment value (all labelled periods)."]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_all_splits()
    lines = ["# Data validation report", "",
             "Generated by `python -m src.data_validation`. The data is fully synthetic.", ""]
    lines += check_splits(splits)
    lines += check_single_feature_leakage(splits["train"])
    lines += check_against_raw(splits)
    lines += save_eda_tables(splits)
    report = "\n".join(lines) + "\n"
    (REPORTS_DIR / "data_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
