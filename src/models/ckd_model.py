"""Layer 3 — CKD (Chronic Kidney Disease) model.

Features derived from the UCI CKD dataset (400 rows, 24 attributes).
Trains a synthetic demo model; replace with joblib exported from real UCI CKD data.
"""
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.models.attribution import compute_feature_attributions
from src.models.base_model import BaseMedicalModel, PredictionResult


class CKDModel(BaseMedicalModel):
    """XGBoost severity model for Chronic Kidney Disease (UCI CKD features)."""

    FEATURE_MEDIANS = {
        "age": 48.0,
        "blood_pressure": 76.0,
        "specific_gravity": 1.020,
        "albumin": 1.0,
        "sugar": 0.0,
        "blood_glucose_random": 121.0,
        "blood_urea": 36.0,
        "serum_creatinine": 1.2,
        "sodium": 137.0,
        "potassium": 4.6,
        "hemoglobin": 12.5,
        "packed_cell_volume": 38.0,
        "white_blood_cell_count": 8000.0,
        "red_blood_cell_count": 4.7,
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        self._scaler: StandardScaler | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def domain(self) -> str:
        return "ckd"

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
            raise RuntimeError("CKDModel not loaded. Call train_demo() or load().")
        prob = float(self._model.predict_proba(features)[0][1])

        # Domain amplifiers: elevated creatinine/urea raise score
        raw = features[0]
        feat_names = self.required_features
        raw_dict = dict(zip(feat_names, raw.tolist() if self._scaler is None else
                        self._scaler.inverse_transform(features)[0].tolist()))
        boost = 0.0
        if raw_dict.get("serum_creatinine", 0) > 4.0:
            boost += 15.0
        elif raw_dict.get("serum_creatinine", 0) > 2.0:
            boost += 8.0
        if raw_dict.get("blood_urea", 0) > 80:
            boost += 10.0
        if raw_dict.get("hemoglobin", 15) < 9.0:
            boost += 7.0

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
                X_scaled, y_train = SMOTE(random_state=44, k_neighbors=k).fit_resample(X_scaled, y_train)
        except Exception:
            pass
        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=44,
        )
        self._model.fit(X_scaled, y_train)

    def train_demo(self, n_samples: int = 600, random_state: int = 44) -> None:
        """Train a synthetic demo model mirroring UCI CKD distributions."""
        rng = np.random.default_rng(random_state)
        names = self.required_features
        means = np.array([self.FEATURE_MEDIANS[k] for k in names], dtype=np.float64)
        spread = np.clip(means * 0.18 + 1.0, 0.2, None)
        X = rng.normal(means, spread, size=(n_samples, len(names)))
        X = np.clip(X, 0.0, None)
        # Synthetic labels: high creatinine + low hemoglobin → CKD positive
        creatinine = X[:, names.index("serum_creatinine")]
        hemoglobin = X[:, names.index("hemoglobin")]
        urea = X[:, names.index("blood_urea")]
        logit = (
            0.6 * (creatinine - 1.2)
            + 0.3 * (urea - 36) / 10
            - 0.3 * (hemoglobin - 12.5)
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
