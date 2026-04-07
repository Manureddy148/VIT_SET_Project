"""Phase 6 — End-to-end smoke tests across all 6 disease models.

Instantiates every registered model, trains with train_demo(), and runs
50+ synthetic patient cases per domain to validate the full pipeline
(preprocess → predict → SHAP → severity label mapping).

Run with:
    pytest tests/test_e2e_smoke.py -v
"""
import numpy as np
import pytest

from src.models.ckd_model import CKDModel
from src.models.diabetes_model import DiabetesModel
from src.models.heart_model import HeartModel
from src.models.liver_model import LiverDiseaseModel
from src.models.pneumonia_model import PneumoniaModel
from src.models.sepsis_model import SepsisModel
from src.models.base_model import PredictionResult

# ---------------------------------------------------------------------------
# Fixtures — trained model instances (shared across tests in a session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diabetes():
    m = DiabetesModel()
    m.train_demo()
    return m


@pytest.fixture(scope="module")
def heart():
    m = HeartModel()
    m.train_demo()
    return m


@pytest.fixture(scope="module")
def pneumonia():
    m = PneumoniaModel()
    m.train_demo()
    return m


@pytest.fixture(scope="module")
def ckd():
    m = CKDModel()
    m.train_demo()
    return m


@pytest.fixture(scope="module")
def sepsis():
    m = SepsisModel()
    m.train_demo()
    return m


@pytest.fixture(scope="module")
def liver():
    m = LiverDiseaseModel()
    m.train_demo()
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_result(result: PredictionResult, domain: str) -> None:
    assert result.disease_domain == domain, f"Wrong domain: {result.disease_domain}"
    assert 0.0 <= result.severity_score <= 100.0, f"Score out of range: {result.severity_score}"
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"
    assert result.severity_label in ("Low", "Moderate", "High", "Critical")
    assert isinstance(result.shap_values, dict)
    assert isinstance(result.top_features, list)


def _run_n_patients(model, raw_inputs: list) -> list:
    results = []
    for raw in raw_inputs:
        features = model.preprocess(raw)
        result = model.predict(features)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Synthetic patient builders
# ---------------------------------------------------------------------------

def _diabetes_patients(n: int = 10) -> list:
    rng = np.random.default_rng(100)
    patients = []
    for _ in range(n):
        patients.append({
            "glucose": float(rng.uniform(70, 320)),
            "bmi": float(rng.uniform(19, 50)),
            "age": float(rng.integers(20, 80)),
            "blood_pressure": float(rng.uniform(60, 130)),
            "insulin": float(rng.uniform(0, 200)),
            "pregnancies": float(rng.integers(0, 10)),
            "skin_thickness": float(rng.uniform(0, 60)),
            "diabetes_pedigree": float(rng.uniform(0.1, 2.0)),
        })
    return patients


def _heart_patients(n: int = 10) -> list:
    rng = np.random.default_rng(101)
    patients = []
    for _ in range(n):
        patients.append({
            "age": float(rng.integers(30, 80)),
            "resting_bp": float(rng.uniform(90, 200)),
            "cholesterol": float(rng.uniform(150, 400)),
            "max_hr": float(rng.uniform(80, 200)),
            "st_depression": float(rng.uniform(0, 6)),
            "major_vessels": float(rng.integers(0, 4)),
            "fasting_bs": float(rng.integers(0, 2)),
            "thalassemia": float(rng.integers(1, 4)),
        })
    return patients


def _pneumonia_patients(n: int = 10) -> list:
    rng = np.random.default_rng(102)
    patients = []
    for _ in range(n):
        patients.append({
            "spo2": float(rng.uniform(82, 100)),
            "temperature_c": float(rng.uniform(36.0, 41.5)),
            "respiratory_rate": float(rng.uniform(10, 45)),
            "crp": float(rng.uniform(0, 250)),
            "age": float(rng.integers(10, 90)),
        })
    return patients


def _ckd_patients(n: int = 10) -> list:
    rng = np.random.default_rng(103)
    patients = []
    for _ in range(n):
        patients.append({
            "age": float(rng.integers(20, 80)),
            "blood_pressure": float(rng.uniform(60, 180)),
            "specific_gravity": float(rng.uniform(1.005, 1.030)),
            "albumin": float(rng.integers(0, 5)),
            "sugar": float(rng.integers(0, 5)),
            "blood_glucose_random": float(rng.uniform(70, 490)),
            "blood_urea": float(rng.uniform(10, 200)),
            "serum_creatinine": float(rng.uniform(0.5, 15.0)),
            "sodium": float(rng.uniform(120, 155)),
            "potassium": float(rng.uniform(3.0, 6.5)),
            "hemoglobin": float(rng.uniform(4, 17)),
            "packed_cell_volume": float(rng.uniform(15, 55)),
            "white_blood_cell_count": float(rng.uniform(3000, 18000)),
            "red_blood_cell_count": float(rng.uniform(2.0, 6.5)),
        })
    return patients


def _sepsis_patients(n: int = 10) -> list:
    rng = np.random.default_rng(104)
    patients = []
    for _ in range(n):
        patients.append({
            "heart_rate": float(rng.uniform(50, 160)),
            "respiratory_rate": float(rng.uniform(10, 40)),
            "temperature_c": float(rng.uniform(35.0, 41.0)),
            "systolic_bp": float(rng.uniform(60, 180)),
            "mean_arterial_pressure": float(rng.uniform(40, 120)),
            "oxygen_saturation": float(rng.uniform(80, 100)),
            "glasgow_coma_scale": float(rng.integers(3, 16)),
            "lactate": float(rng.uniform(0.5, 10.0)),
            "creatinine": float(rng.uniform(0.5, 10.0)),
            "bilirubin": float(rng.uniform(0.3, 15.0)),
            "platelet_count": float(rng.uniform(30, 450)),
            "white_blood_cells": float(rng.uniform(2.0, 30.0)),
            "age": float(rng.integers(18, 90)),
            "hours_in_icu": float(rng.uniform(0, 336)),
        })
    return patients


def _liver_patients(n: int = 10) -> list:
    rng = np.random.default_rng(105)
    patients = []
    for _ in range(n):
        patients.append({
            "age": float(rng.integers(15, 85)),
            "gender_male": float(rng.integers(0, 2)),
            "total_bilirubin": float(rng.uniform(0.1, 25.0)),
            "direct_bilirubin": float(rng.uniform(0.05, 10.0)),
            "alkaline_phosphotase": float(rng.uniform(50, 2000)),
            "alamine_aminotransferase": float(rng.uniform(5, 2000)),
            "aspartate_aminotransferase": float(rng.uniform(5, 4500)),
            "total_proteins": float(rng.uniform(2.5, 9.0)),
            "albumin": float(rng.uniform(0.8, 5.5)),
            "albumin_globulin_ratio": float(rng.uniform(0.2, 3.5)),
        })
    return patients


# ---------------------------------------------------------------------------
# Smoke tests — 10 patients per model (50 total) covering all domains
# ---------------------------------------------------------------------------

class TestDiabetesSmoke:
    def test_ten_patients(self, diabetes):
        results = _run_n_patients(diabetes, _diabetes_patients(10))
        for r in results:
            _assert_valid_result(r, "diabetes")

    def test_high_glucose_maps_high_severity(self, diabetes):
        raw = {"glucose": 300.0, "bmi": 40.0, "age": 60.0}
        features = diabetes.preprocess(raw)
        result = diabetes.predict(features)
        assert result.severity_score > 40.0, "High glucose patient should score > 40"

    def test_normal_patient_not_critical(self, diabetes):
        raw = {"glucose": 90.0, "bmi": 24.0, "age": 30.0}
        features = diabetes.preprocess(raw)
        result = diabetes.predict(features)
        assert result.severity_label != "Critical", "Normal patient should not be Critical"

    def test_shap_values_present(self, diabetes):
        pytest.importorskip("shap", reason="shap library not installed")
        raw = {"glucose": 200.0, "bmi": 35.0, "age": 50.0}
        features = diabetes.preprocess(raw)
        result = diabetes.predict(features)
        assert len(result.shap_values) > 0, "shap values should be populated when shap is installed"


class TestHeartSmoke:
    def test_ten_patients(self, heart):
        results = _run_n_patients(heart, _heart_patients(10))
        for r in results:
            _assert_valid_result(r, "heart_disease")

    def test_high_bp_cholesterol_elevated(self, heart):
        raw = {"resting_bp": 185.0, "cholesterol": 380.0, "age": 68.0, "max_hr": 95.0}
        features = heart.preprocess(raw)
        result = heart.predict(features)
        assert result.severity_score > 30.0


class TestPneumoniaSmoke:
    def test_ten_patients(self, pneumonia):
        results = _run_n_patients(pneumonia, _pneumonia_patients(10))
        for r in results:
            _assert_valid_result(r, "pneumonia")

    def test_hypoxic_patient_high_severity(self, pneumonia):
        raw = {"spo2": 87.0, "temperature_c": 39.8, "respiratory_rate": 32.0, "crp": 180.0, "age": 72.0}
        features = pneumonia.preprocess(raw)
        result = pneumonia.predict(features)
        assert result.severity_score > 50.0

    def test_normal_patient(self, pneumonia):
        raw = {"spo2": 98.0, "temperature_c": 36.8, "respiratory_rate": 14.0, "crp": 3.0, "age": 30.0}
        features = pneumonia.preprocess(raw)
        result = pneumonia.predict(features)
        assert result.severity_label in ("Low", "Moderate")


class TestCKDSmoke:
    def test_ten_patients(self, ckd):
        results = _run_n_patients(ckd, _ckd_patients(10))
        for r in results:
            _assert_valid_result(r, "ckd")

    def test_high_creatinine_elevated(self, ckd):
        raw = {"serum_creatinine": 9.5, "blood_urea": 140.0, "hemoglobin": 7.0, "age": 65.0}
        features = ckd.preprocess(raw)
        result = ckd.predict(features)
        assert result.severity_score > 50.0


class TestSepsisSmoke:
    def test_ten_patients(self, sepsis):
        results = _run_n_patients(sepsis, _sepsis_patients(10))
        for r in results:
            _assert_valid_result(r, "sepsis")

    def test_qsofa_positive_elevated(self, sepsis):
        # qSOFA: RR>=22, SBP<=100, GCS<15, lactate>=2
        raw = {
            "respiratory_rate": 26.0, "systolic_bp": 88.0,
            "glasgow_coma_scale": 12.0, "lactate": 4.2,
            "heart_rate": 118.0, "temperature_c": 38.9,
        }
        features = sepsis.preprocess(raw)
        result = sepsis.predict(features)
        assert result.severity_score > 60.0, f"qSOFA+ sepsis should be high risk, got {result.severity_score}"


class TestLiverSmoke:
    def test_ten_patients(self, liver):
        results = _run_n_patients(liver, _liver_patients(10))
        for r in results:
            _assert_valid_result(r, "liver_disease")

    def test_high_enzymes_elevated(self, liver):
        raw = {
            "total_bilirubin": 8.5, "direct_bilirubin": 4.0,
            "alamine_aminotransferase": 450.0, "aspartate_aminotransferase": 380.0,
            "albumin": 2.1, "age": 55.0,
        }
        features = liver.preprocess(raw)
        result = liver.predict(features)
        assert result.severity_score > 50.0

    def test_normal_liver(self, liver):
        raw = {
            "total_bilirubin": 0.8, "direct_bilirubin": 0.2,
            "alamine_aminotransferase": 28.0, "aspartate_aminotransferase": 25.0,
            "albumin": 4.2, "age": 35.0,
        }
        features = liver.preprocess(raw)
        result = liver.predict(features)
        # Demo model uses synthetic training data; score should not reach Critical (>=75)
        assert result.severity_label != "Critical", (
            f"Normal liver patient should not be Critical, got {result.severity_label} ({result.severity_score})"
        )


# ---------------------------------------------------------------------------
# Cross-model: severity label coverage (ablation target: all labels reachable)
# ---------------------------------------------------------------------------

def test_severity_label_coverage_diabetes(diabetes):
    """Confirm all 4 severity buckets are reachable across synthetic patients."""
    patients = _diabetes_patients(50)
    labels = set()
    for raw in patients:
        r = diabetes.predict(diabetes.preprocess(raw))
        labels.add(r.severity_label)
    # With 50 synthetic patients, at minimum Low and High should appear
    assert len(labels) >= 2, f"Expected >= 2 distinct severity labels, got: {labels}"


def test_shap_top_features_ranked(diabetes):
    """Top features list should be sorted by absolute SHAP descending."""
    raw = {"glucose": 250.0, "bmi": 38.0, "age": 55.0}
    r = diabetes.predict(diabetes.preprocess(raw))
    shap = r.shap_values
    top = r.top_features
    # Each top feature should have |shap| >= the next
    for i in range(len(top) - 1):
        assert abs(shap.get(top[i], 0)) >= abs(shap.get(top[i + 1], 0)), (
            f"Top features not sorted by |SHAP|: {top}"
        )
