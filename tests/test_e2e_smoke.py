"""
End-to-end API sanity: full pipeline shape, all domains, errors, OpenAPI.
Run: pytest tests/test_e2e_smoke.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

REQUIRED_PREDICT_KEYS = frozenset(
    {
        "disease_domain",
        "severity_score",
        "severity_label",
        "confidence",
        "top_risk_factors",
        "shap_values",
        "missing_features",
        "literature_citations",
        "explanation",
        "disclaimer",
    }
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec.get("openapi")
    paths = spec.get("paths", {})
    assert "/health" in paths
    assert "/predict" in paths


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def _assert_predict_shape(body: dict) -> None:
    assert REQUIRED_PREDICT_KEYS <= body.keys()
    assert isinstance(body["severity_score"], (int, float))
    assert 0 <= float(body["severity_score"]) <= 100
    assert body["severity_label"] in ("Low", "Moderate", "High", "Critical")
    assert body["disclaimer"]
    assert isinstance(body["top_risk_factors"], list)
    assert isinstance(body["shap_values"], dict)
    assert isinstance(body["literature_citations"], list)


def test_e2e_diabetes(client):
    r = client.post(
        "/predict",
        json={
            "query_text": "glucose 210 BMI 34 age 52 polyuria",
            "clinical_data": {},
            "patient_id": "e2e_d1",
        },
    )
    assert r.status_code == 200
    b = r.json()
    _assert_predict_shape(b)
    assert b["disease_domain"] == "diabetes"


def test_e2e_heart(client):
    r = client.post(
        "/predict",
        json={
            "query_text": "chest pain troponin concern cholesterol high",
            "clinical_data": {"cholesterol": 270, "resting_bp": 145, "age": 58},
        },
    )
    assert r.status_code == 200
    b = r.json()
    _assert_predict_shape(b)
    assert b["disease_domain"] == "heart_disease"


def test_e2e_pneumonia(client):
    r = client.post(
        "/predict",
        json={
            "query_text": "fever cough pneumonia spO2 low",
            "clinical_data": {
                "spo2": 90,
                "temperature_c": 38.8,
                "respiratory_rate": 26,
                "crp": 95,
                "age": 62,
            },
        },
    )
    assert r.status_code == 200
    b = r.json()
    _assert_predict_shape(b)
    assert b["disease_domain"] == "pneumonia"


def test_e2e_unknown_domain_422(client):
    r = client.post(
        "/predict",
        json={"query_text": "lorem ipsum dolor sit amet only", "clinical_data": {}},
    )
    assert r.status_code == 422


def test_e2e_clinical_only_routing(client):
    r = client.post(
        "/predict",
        json={"query_text": "", "clinical_data": {"glucose": 190, "bmi": 31}},
    )
    assert r.status_code == 200
    assert r.json()["disease_domain"] == "diabetes"


def test_e2e_patient_input_optional_fields(client):
    r = client.post(
        "/predict",
        json={"query_text": "diabetes screening glucose borderline"},
    )
    assert r.status_code == 200
    _assert_predict_shape(r.json())
