from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseMedicalModel, PredictionResult


class HeartModel(BaseMedicalModel):
    """XGBoost-style model for heart disease risk (Cleveland-style features, simplified)."""

    FEATURE_MEDIANS = {
        "age": 54.0,
        "resting_bp": 131.0,
        "cholesterol": 246.0,
        "max_hr": 149.0,
        "st_depression": 1.0,
        "major_vessels": 0.0,
    }

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        self._scaler: StandardScaler | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def domain(self) -> str:
        return "heart_disease"

    @property
    def required_features(self) -> List[str]:
        return list(self.FEATURE_MEDIANS.keys())

    def preprocess(self, raw_input: Dict) -> np.ndarray:
        row = []
        for feature, median in self.FEATURE_MEDIANS.items():
            value = raw_input.get(feature)
            if value is None or (
                value == 0 and feature not in ("st_depression", "major_vessels")
            ):
                value = median
            row.append(float(value))
        features = np.array(row, dtype=np.float64).reshape(1, -1)
        if self._scaler:
            features = self._scaler.transform(features)
        return features

    def predict(self, features: np.ndarray) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call train_demo(), train(), or load().")
        prob = float(self._model.predict_proba(features)[0][1])
        severity_score = min(100.0, round(float(prob) * 100.0, 1))
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
        self._model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.08,
            random_state=43,
        )
        self._model.fit(X_scaled, y_train)

    def train_demo(self, n_samples: int = 800, random_state: int = 43) -> None:
        rng = np.random.default_rng(random_state)
        names = list(self.FEATURE_MEDIANS.keys())
        means = np.array([self.FEATURE_MEDIANS[k] for k in names], dtype=np.float64)
        X = rng.normal(means, means * 0.15 + 2.0, size=(n_samples, len(names)))
        X = np.clip(X, 0.0, None)
        chol = X[:, 2]
        bp = X[:, 1]
        age = X[:, 0]
        logit = 0.015 * (chol - 246) + 0.02 * (bp - 131) + 0.03 * (age - 54)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n_samples) < p).astype(np.int32)
        self.train(X, y)

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "scaler": self._scaler}, path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
