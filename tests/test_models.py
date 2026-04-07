from src.models.diabetes_model import DiabetesModel
from src.models.pneumonia_model import PneumoniaModel
from src.models.registry import ModelRegistry


def test_diabetes_demo_train_and_predict():
    m = DiabetesModel()
    m.train_demo(n_samples=200, random_state=0)
    raw = {
        "pregnancies": 2,
        "glucose": 180,
        "blood_pressure": 80,
        "skin_thickness": 25,
        "insulin": 40,
        "bmi": 35,
        "diabetes_pedigree": 0.5,
        "age": 50,
    }
    x = m.preprocess(raw)
    r = m.predict(x)
    assert r.disease_domain == "diabetes"
    assert 0 <= r.severity_score <= 100
    assert r.severity_label in ("Low", "Moderate", "High", "Critical")


def test_registry_routes_diabetes():
    reg = ModelRegistry()
    dm = DiabetesModel()
    dm.train_demo(n_samples=150, random_state=1)
    reg.register(dm, keywords=["diabetes", "glucose"])
    text = "patient has high blood glucose"
    r = reg.route_and_predict(text, {"glucose": 200, "bmi": 34})
    assert r.disease_domain == "diabetes"


def test_pneumonia_stub():
    m = PneumoniaModel()
    m.train_demo()
    x = m.preprocess({"spo2": 88, "temperature_c": 39, "respiratory_rate": 28, "crp": 120, "age": 70})
    r = m.predict(x)
    assert r.severity_score >= 50
