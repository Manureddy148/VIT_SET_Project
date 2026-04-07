"""Runtime smoke tests for deployed-like API behavior.

These tests focus on quick confidence checks:
- app starts and serves health/docs-openapi routes
- core predict endpoint works
- report PDF endpoint returns a valid PDF payload
"""

from fastapi.testclient import TestClient

from src.api.main import app


def test_smoke_health_and_openapi() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json().get("status") == "ok"

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        body = openapi.json()
        assert body.get("info", {}).get("title") == "Medical AI Severity API"


def test_smoke_predict_diabetes() -> None:
    with TestClient(app) as client:
        payload = {
            "query_text": "glucose 210 bmi 34 age 52 frequent urination",
            "clinical_data": {},
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        out = r.json()
        assert out["disease_domain"] == "diabetes"
        assert 0 <= out["severity_score"] <= 100
        assert out["severity_label"] in ("Low", "Moderate", "High", "Critical")
        assert len(out.get("shap_values", {})) > 0


def test_smoke_predict_liver_structured_only() -> None:
    with TestClient(app) as client:
        payload = {
            "query_text": "",
            "clinical_data": {
                "total_bilirubin": 4.5,
                "direct_bilirubin": 2.0,
                "alamine_aminotransferase": 220.0,
                "aspartate_aminotransferase": 190.0,
                "albumin": 2.8,
                "age": 54,
            },
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        out = r.json()
        assert out["disease_domain"] == "liver_disease"
        assert 0 <= out["severity_score"] <= 100
        assert len(out.get("literature_citations", [])) > 0


def test_smoke_stream_and_pdf() -> None:
    with TestClient(app) as client:
        payload = {
            "query_text": "Patient with chest pain and high cholesterol",
            "clinical_data": {"cholesterol": 290, "resting_bp": 152, "age": 63},
        }

        stream = client.post("/predict/stream", json=payload)
        assert stream.status_code == 200
        assert "event: summary" in stream.text
        assert "event: done" in stream.text

        pdf = client.post("/report/pdf", json=payload)
        assert pdf.status_code == 200
        assert pdf.headers.get("content-type", "").startswith("application/pdf")
        assert pdf.content[:4] == b"%PDF"
