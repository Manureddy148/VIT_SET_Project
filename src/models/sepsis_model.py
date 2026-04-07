"""Layer 3 — Sepsis model.

Features derived from PhysioNet 2019 Sepsis Challenge (qSOFA + SOFA components).
Trains a synthetic demo XGBoost model; replace with joblib from real PhysioNet data.
"""
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseMedicalModel, PredictionResult


class SepsisModel(BaseMedicalModel):
    """XGBoost severity model for Sepsis (qSOFA/SOFA-derived features).

    Severity score is amplified by qSOFA score (0–3) components as per docs.
    """

    # qSOFA + SOFA key lab components per PhysioNet 2019
    FEATURE_MEDIANS = {
        "heart_rate": 85.0,
        "respiratory_rate": 18.0,
        "temperature_c": 37.0,
        "systolic_bp": 120.0,
        "mean_arterial_pressure": 80.0,
        "oxygen_saturation": 97.0,
        "glasgow_coma_scale": 15.0,
        "lactate": 1.5,
        "creatinine": 0.9,
        "bilirubin": 0.8,
        "platelet_count": 200.0,
        "white_blood_cells": 8.0,
        "age": 52.0,
        "hours_in_icu": 12.0,
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        self._scaler: StandardScaler | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def domain(self) -> str:
        return "sepsis"

    @property
    def required_features(self) -> List[str]:
        return list(self.FEATURE_MEDIANS.keys())

    def preprocess(self, raw_input: Dict) -> np.ndarray:
        row = []
        for feature, median in self.FEATURE_MEDIANS.items():
            value = raw_input.get(feature)
            if value is None:
                value = median
            row.append(float(value))
        features = np.array(row, dtype=np.float64).reshape(1, -1)
        if self._scaler:
            features = self._scaler.transform(features)
        return features

    def predict(self, features: np.ndarray) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("SepsisModel not loaded. Call train_demo() or load().")
        prob = float(self._model.predict_proba(features)[0][1])

        # qSOFA-based boost (doc-specified: qSOFA + SOFA → score)
        raw = (
            self._scaler.inverse_transform(features)[0]
            if self._scaler else features[0]
        )
        names = self.required_features
        rd = dict(zip(names, raw.tolist()))

        # qSOFA components (each worth ~10 pts)
        qsofa = 0
        if rd.get("respiratory_rate", 18) >= 22:
            qsofa += 1
        if rd.get("systolic_bp", 120) <= 100:
            qsofa += 1
        if rd.get("glasgow_coma_scale", 15) < 15:
            qsofa += 1

        # Lactate >= 2 mmol/L (lactate boost)
        lactate_boost = 10.0 if rd.get("lactate", 1.5) >= 2.0 else 0.0

        severity_score = min(100.0, round(prob * 100.0 + qsofa * 10.0 + lactate_boost, 1))
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
        try:
            import shap

            explainer = shap.TreeExplainer(self._model)
            vals = explainer.shap_values(features)
            if isinstance(vals, list):
                vals = vals[1]
            return dict(zip(self.required_features, vals[0].tolist()))
        except Exception:
            return {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        try:
            from imblearn.over_sampling import SMOTE

            k = min(5, int(np.bincount(y_train).min()) - 1)
            if k > 0:
                X_scaled, y_train = SMOTE(random_state=45, k_neighbors=k).fit_resample(X_scaled, y_train)
        except Exception:
            pass
        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=45,
        )
        self._model.fit(X_scaled, y_train)

    def train_demo(self, n_samples: int = 600, random_state: int = 45) -> None:
        """Synthetic demo model mirroring PhysioNet sepsis distributions."""
        rng = np.random.default_rng(random_state)
        names = self.required_features
        means = np.array([self.FEATURE_MEDIANS[k] for k in names], dtype=np.float64)
        spread = np.clip(means * 0.15 + 1.0, 0.1, None)
        X = rng.normal(means, spread, size=(n_samples, len(names)))
        X = np.clip(X, 0.0, None)

        # Synthetic labels: high HR + low BP + high RR → sepsis
        hr = X[:, names.index("heart_rate")]
        rr = X[:, names.index("respiratory_rate")]
        sbp = X[:, names.index("systolic_bp")]
        lactate = X[:, names.index("lactate")]
        logit = (
            0.04 * (hr - 85)
            + 0.1 * (rr - 18)
            - 0.03 * (sbp - 120)
            + 0.5 * (lactate - 1.5)
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n_samples) < p).astype(np.int32)
        self.train(X, y)

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "scaler": self._scaler}, path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
