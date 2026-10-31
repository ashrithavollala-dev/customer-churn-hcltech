"""These tests need the trained bundle: run `python -m src.train` first."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import FEATURES, MODEL_BUNDLE_PATH
from src.explain import explain_dataframe, local_explanation, top_factor_text
from src.predict import load_bundle, predict_raw_probability, validate_and_score
from src.preprocessing import load_split

pytestmark = pytest.mark.skipif(not MODEL_BUNDLE_PATH.exists(), reason="model bundle not trained yet")


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def sample():
    return load_split("test").head(20)


def test_saved_model_scores_rows(bundle, sample):
    scored, errors = validate_and_score(sample[["customer_id"] + FEATURES], bundle)
    assert errors == []
    p = scored["churn_probability"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    assert set(scored["risk_category"]) <= {"Low", "Medium", "High"}
    # Calibration is monotonic, so it must not change the ranking.
    raw = scored["raw_model_probability"].to_numpy()
    assert (np.argsort(raw, kind="stable") == np.argsort(p, kind="stable")).all()


def test_predictions_are_deterministic(bundle, sample):
    assert np.array_equal(predict_raw_probability(bundle["model"], sample),
                          predict_raw_probability(bundle["model"], sample))


def test_shap_values_add_up_to_model_output(bundle, sample):
    shap_df, base = explain_dataframe(sample, bundle)
    assert list(shap_df.columns) == FEATURES
    logit = base + shap_df.sum(axis=1).to_numpy()
    raw = predict_raw_probability(bundle["model"], sample)
    assert np.allclose(1 / (1 + np.exp(-logit)), raw, atol=1e-6)
    table, _ = local_explanation(sample.head(1), bundle)
    assert len(table) == 24
    assert len(top_factor_text(sample, bundle)) == len(sample)


def test_catboost_candidate_can_be_explained(bundle, sample):
    from src.explain import shap_values_for
    shap_df, base = shap_values_for(sample, bundle["candidate_models"]["catboost"], "catboost", bundle["background_sample"])
    raw = predict_raw_probability(bundle["candidate_models"]["catboost"], sample)
    assert np.allclose(1 / (1 + np.exp(-(base + shap_df.sum(axis=1).to_numpy()))), raw, atol=1e-4)


@pytest.fixture(scope="module")
def client():
    from api import app
    with TestClient(app) as c:
        yield c


def _payload(sample, i=0):
    row = sample.iloc[i]
    data = {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in FEATURES}
    data["customer_id"] = row["customer_id"]
    return data


def test_api_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and len(body["features"]) == 24


def test_api_predict_matches_local_scoring(client, bundle, sample):
    response = client.post("/predict", json=_payload(sample))
    assert response.status_code == 200
    local, _ = validate_and_score(sample.head(1)[FEATURES], bundle)
    assert abs(response.json()["churn_probability"] - local["churn_probability"].iloc[0]) < 1e-4
    assert len(response.json()["contributions"]) > 0


def test_api_batch_and_optional_csat(client, sample):
    customers = [_payload(sample, i) for i in range(5)]
    del customers[0]["last_csat_score"]
    response = client.post("/predict-batch", json={"customers": customers})
    assert response.status_code == 200
    assert response.json()["count"] == 5


def test_api_rejects_invalid_input(client, sample):
    bad = _payload(sample)
    bad["contract_type"] = "Weekly"
    assert client.post("/predict", json=bad).status_code == 422
    missing = _payload(sample)
    del missing["net_charge"]
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict-batch", json={"customers": []}).status_code == 422
