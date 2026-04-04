import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_diabetes(client):
    payload = {
        "query_text": "Type 2 concern: glucose 210, BMI 34, age 52, frequent urination",
        "clinical_data": {},
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["disease_domain"] == "diabetes"
    assert 0 <= body["severity_score"] <= 100
    assert body["disclaimer"]


def test_predict_422_unknown_domain(client):
    r = client.post("/predict", json={"query_text": "random words only xyz", "clinical_data": {}})
    assert r.status_code == 422


def test_predict_heart(client):
    r = client.post(
        "/predict",
        json={
            "query_text": "Patient with chest pain and high cholesterol",
            "clinical_data": {"cholesterol": 280, "resting_bp": 150, "age": 62},
        },
    )
    assert r.status_code == 200
    assert r.json()["disease_domain"] == "heart_disease"


def test_predict_pneumonia(client):
    r = client.post(
        "/predict",
        json={
            "query_text": "Suspected pneumonia with cough and fever",
            "clinical_data": {
                "spo2": 89,
                "temperature_c": 38.5,
                "respiratory_rate": 24,
                "crp": 80,
                "age": 55,
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["disease_domain"] == "pneumonia"


def test_predict_infer_domain_from_clinical_only(client):
    """Layer 2 fallback: empty narrative but structured labs imply diabetes."""
    r = client.post(
        "/predict",
        json={"query_text": "", "clinical_data": {"glucose": 200, "bmi": 33}},
    )
    assert r.status_code == 200
    assert r.json()["disease_domain"] == "diabetes"
