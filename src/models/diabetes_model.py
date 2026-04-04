from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseMedicalModel, PredictionResult


class DiabetesModel(BaseMedicalModel):
    """XGBoost severity model for diabetes (Pima-style features)."""

    FEATURE_MEDIANS = {
        "pregnancies": 3.0,
        "glucose": 117.0,
        "blood_pressure": 72.0,
        "skin_thickness": 23.0,
        "insulin": 30.5,
        "bmi": 32.0,
        "diabetes_pedigree": 0.372,
        "age": 29.0,
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        self._scaler: StandardScaler | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def domain(self) -> str:
        return "diabetes"

    @property
    def required_features(self) -> List[str]:
        return list(self.FEATURE_MEDIANS.keys())

    def preprocess(self, raw_input: Dict) -> np.ndarray:
        row = []
        for feature, median in self.FEATURE_MEDIANS.items():
            value = raw_input.get(feature)
            if value is None or (feature != "pregnancies" and value == 0):
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
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
        self._model.fit(X_scaled, y_train)

    def train_demo(self, n_samples: int = 1000, random_state: int = 42) -> None:
        """Train a small synthetic model so the API works without a Kaggle export."""
        rng = np.random.default_rng(random_state)
        names = list(self.FEATURE_MEDIANS.keys())
        means = np.array([self.FEATURE_MEDIANS[k] for k in names], dtype=np.float64)
        spread = np.clip(means * 0.2 + 1.0, 0.5, None)
        X = rng.normal(means, spread, size=(n_samples, len(names)))
        X = np.clip(X, 0.0, None)
        glucose = X[:, 1]
        bmi = X[:, 5]
        age = X[:, 7]
        logit = 0.025 * (glucose - 117) + 0.12 * (bmi - 32) + 0.02 * (age - 29)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n_samples) < p).astype(np.int32)
        self.train(X, y)

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "scaler": self._scaler}, path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
