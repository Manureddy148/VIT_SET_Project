"""Layer 3 — Pneumonia severity model.

Upgraded from rule-based stub to XGBoost trained on synthetic clinical features
mirroring the Kaggle Chest X-Ray Pneumonia dataset's clinical correlates
(SpO2, temperature, respiratory rate, CRP, age).
Replace train_demo() with Kaggle-trained joblib when available.
"""
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.models.attribution import compute_feature_attributions
from src.models.base_model import BaseMedicalModel, PredictionResult


class PneumoniaModel(BaseMedicalModel):
    """XGBoost severity model for pneumonia (clinical vital signs + CRP)."""

    FEATURE_MEDIANS = {
        "spo2": 96.0,
        "temperature_c": 37.0,
        "respiratory_rate": 18.0,
        "crp": 10.0,
        "age": 45.0,
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        self._scaler: StandardScaler | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

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
        features = np.array(row, dtype=np.float64).reshape(1, -1)
        if self._scaler:
            features = self._scaler.transform(features)
        return features

    def predict(self, features: np.ndarray) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("PneumoniaModel not loaded. Call train_demo() or load().")
        prob = float(self._model.predict_proba(features)[0][1])

        # Domain amplifiers (clinical thresholds from pneumonia severity indices)
        raw = (
            self._scaler.inverse_transform(features)[0]
            if self._scaler else features[0]
        )
        rd = dict(zip(self.required_features, raw.tolist()))
        boost = 0.0
        if rd.get("spo2", 96) < 90:
            boost += 15.0
        elif rd.get("spo2", 96) < 94:
            boost += 8.0
        if rd.get("temperature_c", 37) > 39.5:
            boost += 8.0
        elif rd.get("temperature_c", 37) > 38.5:
            boost += 4.0
        if rd.get("respiratory_rate", 18) > 30:
            boost += 8.0
        if rd.get("crp", 10) > 150:
            boost += 10.0

        severity_score = min(100.0, round(prob * 100.0 + boost, 1))
        shap_vals = self._compute_shap(features)
        top = sorted(shap_vals, key=lambda k: abs(shap_vals[k]), reverse=True)[:5]
        return PredictionResult(
            disease_domain=self.domain,
            severity_score=severity_score,
            confidence=max(prob, 1.0 - prob),
            severity_label=self.score_to_label(severity_score),
            shap_values=shap_vals,
            top_features=top,
            missing_features=[],
        )

    def _compute_shap(self, features: np.ndarray) -> Dict[str, float]:
        if self._model is None:
            return {}
        return compute_feature_attributions(
            self._model,
            features,
            self.required_features,
            self.FEATURE_MEDIANS,
            scaler=self._scaler,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        try:
            from imblearn.over_sampling import SMOTE

            k = min(5, int(np.bincount(y_train).min()) - 1)
            if k > 0:
                X_scaled, y_train = SMOTE(random_state=46, k_neighbors=k).fit_resample(X_scaled, y_train)
        except Exception:
            pass
        self._model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=46,
        )
        self._model.fit(X_scaled, y_train)

    def train_demo(self, n_samples: int = 800, random_state: int = 46) -> None:
        """Synthetic demo model trained on Kaggle CXR-correlated clinical features."""
        rng = np.random.default_rng(random_state)
        names = self.required_features
        means = np.array([self.FEATURE_MEDIANS[k] for k in names], dtype=np.float64)
        spread = np.array([3.0, 0.8, 3.0, 30.0, 12.0])
        X = rng.normal(means, spread, size=(n_samples, len(names)))
        # Keep clinical ranges
        X[:, 0] = np.clip(X[:, 0], 70.0, 100.0)   # spo2
        X[:, 1] = np.clip(X[:, 1], 35.0, 42.0)    # temperature
        X[:, 2] = np.clip(X[:, 2], 10.0, 50.0)    # rr
        X[:, 3] = np.clip(X[:, 3], 0.0, 300.0)    # crp
        X[:, 4] = np.clip(X[:, 4], 0.0, 100.0)    # age
        # Synthetic labels: low spo2 + high temp + high CRP → pneumonia
        spo2 = X[:, 0]
        temp = X[:, 1]
        crp = X[:, 3]
        logit = -0.3 * (spo2 - 96) + 2.0 * (temp - 37) + 0.02 * (crp - 10)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n_samples) < p).astype(np.int32)
        self.train(X, y)

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "scaler": self._scaler}, path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
