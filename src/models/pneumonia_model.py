from typing import Dict, List

import numpy as np

from src.models.base_model import BaseMedicalModel, PredictionResult


class PneumoniaModel(BaseMedicalModel):
    """
    Rule-based placeholder for pneumonia (Phase 1 stub).
    Replace with a trained imaging/clinical model when data is available.
    """

    FEATURE_MEDIANS = {
        "spo2": 96.0,
        "temperature_c": 37.0,
        "respiratory_rate": 18.0,
        "crp": 10.0,
        "age": 45.0,
    }

    @property
    def domain(self) -> str:
        return "pneumonia"

    @property
    def required_features(self) -> List[str]:
        return list(self.FEATURE_MEDIANS.keys())

    def preprocess(self, raw_input: Dict) -> np.ndarray:
        row = []
        for k, med in self.FEATURE_MEDIANS.items():
            v = raw_input.get(k)
            if v is None:
                v = med
            row.append(float(v))
        return np.array(row, dtype=np.float64).reshape(1, -1)

    def predict(self, features: np.ndarray) -> PredictionResult:
        spo2, temp, rr, crp, age = features[0].tolist()
        score = 20.0
        if spo2 < 94:
            score += (94 - spo2) * 4
        if temp > 38.0:
            score += (temp - 38.0) * 8
        if rr > 22:
            score += (rr - 22) * 2
        if crp > 50:
            score += min(25.0, (crp - 50) * 0.2)
        if age > 65:
            score += min(15.0, (age - 65) * 0.4)
        severity_score = float(min(100.0, round(score, 1)))
        shap = {
            "spo2": -2.0 if spo2 < 94 else 0.5,
            "temperature_c": 1.5 if temp > 38 else -0.2,
            "respiratory_rate": 1.2 if rr > 22 else 0.0,
            "crp": 0.8 if crp > 50 else 0.0,
            "age": 0.5 if age > 65 else 0.0,
        }
        top = sorted(shap, key=lambda k: abs(shap[k]), reverse=True)[:5]
        return PredictionResult(
            disease_domain=self.domain,
            severity_score=severity_score,
            confidence=0.65,
            severity_label=self.score_to_label(severity_score),
            shap_values=shap,
            top_features=top,
            missing_features=[],
        )
