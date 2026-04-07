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


def test_pipeline_status(client):
    r = client.get("/pipeline/status")
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline_layers"] == 7
    assert "implemented" in body
    assert "planned_next" in body
    assert "layer6_synthesis" in body["implemented"]
    assert "fastapi_audio_api" in body["implemented"]["layer7_delivery"]
    assert "fastapi_genomics_api" in body["implemented"]["layer7_delivery"]


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
    assert "faithfulness_passed" in body
    assert "recommended_actions" in body
    assert "audit_log_id" in body


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


def test_predict_image_upload(client):
    r = client.post(
        "/predict/image",
        data={"query_text": "Portable chest xray for pneumonia triage", "clinical_json": '{"age": 71}'},
        files={"file": ("chest_xray.png", b"\x89PNG\r\n\x1a\n" + bytes(range(64)), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disease_domain"] == "pneumonia"
    assert 0 <= body["severity_score"] <= 100
    assert isinstance(body["safety_flags"], list)


def test_predict_ecg_upload(client):
    payload = "0.1,0.4,1.2,0.2,-0.1,0.7,1.5,0.3\n" * 20
    r = client.post(
        "/predict/ecg",
        data={"query_text": "ECG upload for cardiac review", "clinical_json": '{"age": 62}'},
        files={"file": ("ecg_signal.csv", payload.encode("utf-8"), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disease_domain"] == "heart_disease"
    assert 0 <= body["severity_score"] <= 100


def test_predict_stream(client):
    r = client.post(
        "/predict/stream",
        json={"query_text": "glucose 190 bmi 31 age 47", "clinical_data": {}},
    )
    assert r.status_code == 200
    assert "event: summary" in r.text
    assert "event: report" in r.text


def test_predict_audio_upload(client):
    audio_bytes = bytes([10, 20, 40, 120, 200, 90, 30, 15]) * 200
    r = client.post(
        "/predict/audio",
        data={"query_text": "lung audio review", "clinical_json": '{"age": 59}'},
        files={"file": ("lungs.wav", audio_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disease_domain"] == "pneumonia"
    assert 0 <= body["severity_score"] <= 100


def test_predict_genomics_upload(client):
    genomic_text = "#CHROM POS ID REF ALT QUAL FILTER INFO\n1 123 rs1 A G . . pathogenic risk CYP2D6"
    r = client.post(
        "/predict/genomics",
        data={"query_text": "genomic risk triage", "clinical_json": '{"age": 48}'},
        files={"file": ("sample.vcf", genomic_text.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disease_domain"] == "diabetes"
    assert 0 <= body["severity_score"] <= 100
