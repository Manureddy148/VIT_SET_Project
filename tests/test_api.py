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
