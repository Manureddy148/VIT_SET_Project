"""Layer 3 — LiverDiseaseModel: XGBoost on ILPD (Indian Liver Patient Dataset).

Features mirror the UCI ILPD dataset (583 rows, 10 features).
SMOTE is applied during training to address the 3:1 class imbalance.
"""
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.models.attribution import compute_feature_attributions
from src.models.base_model import BaseMedicalModel, PredictionResult


class LiverDiseaseModel(BaseMedicalModel):
    """XGBoost + SMOTE model for liver disease (ILPD dataset features)."""

    FEATURE_MEDIANS = {
        "age": 45.0,
        "gender_male": 1.0,          # 1=Male, 0=Female
        "total_bilirubin": 1.0,       # mg/dL  (normal <1.2)
        "direct_bilirubin": 0.3,      # mg/dL  (normal <0.3)
        "alkaline_phosphotase": 208.0, # U/L   (normal 44-147)
        "alamine_aminotransferase": 35.0,  # ALT U/L (normal <56)
        "aspartate_aminotransferase": 30.0, # AST U/L (normal <40)
        "total_proteins": 6.8,         # g/dL   (normal 6.3-8.2)
        "albumin": 3.3,                # g/dL   (normal 3.5-5.0)
        "albumin_globulin_ratio": 1.0, # (normal 1.1-2.5)
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        self._scaler: StandardScaler | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def domain(self) -> str:
        return "liver_disease"

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
            raise RuntimeError("LiverDiseaseModel not loaded. Call train_demo() or load().")
        prob = float(self._model.predict_proba(features)[0][1])

        # Domain amplifiers (clinical thresholds for liver enzymes)
        raw = (
            self._scaler.inverse_transform(features)[0]
            if self._scaler else features[0]
        )
        rd = dict(zip(self.required_features, raw.tolist()))
        boost = 0.0
        tbili = rd.get("total_bilirubin", 1.0)
        alt = rd.get("alamine_aminotransferase", 35.0)
        ast = rd.get("aspartate_aminotransferase", 30.0)
        alb = rd.get("albumin", 3.3)
        if tbili > 3.0:
            boost += 12.0
        elif tbili > 1.5:
            boost += 6.0
        if alt > 200 or ast > 200:
            boost += 10.0
        elif alt > 100 or ast > 100:
            boost += 5.0
        if alb < 2.5:
            boost += 8.0
        elif alb < 3.0:
            boost += 4.0

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
        """Train with SMOTE for 3:1 class imbalance in ILPD."""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        # Apply SMOTE if available
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=42, k_neighbors=min(5, int(np.bincount(y_train).min()) - 1))
            X_scaled, y_train = sm.fit_resample(X_scaled, y_train)
        except Exception:
            pass  # graceful fallback; imbalanced-learn optional
        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=3.0,  # extra guard for imbalance
            random_state=47,
        )
        self._model.fit(X_scaled, y_train)

    def train_demo(self, n_samples: int = 700, random_state: int = 47) -> None:
        """Synthetic demo model mirroring ILPD liver patient statistics."""
        rng = np.random.default_rng(random_state)
        names = self.required_features
        means = np.array([self.FEATURE_MEDIANS[k] for k in names], dtype=np.float64)
        spread = np.array([12.0, 0.5, 1.5, 0.6, 100.0, 40.0, 35.0, 0.8, 0.6, 0.4])
        X = rng.normal(means, spread, size=(n_samples, len(names)))
        X[:, 0] = np.clip(X[:, 0], 4.0, 90.0)     # age
        X[:, 1] = np.round(np.clip(X[:, 1], 0.0, 1.0))  # gender binary
        X[:, 2] = np.clip(X[:, 2], 0.1, 40.0)     # total_bilirubin
        X[:, 3] = np.clip(X[:, 3], 0.05, 15.0)    # direct_bilirubin
        X[:, 4] = np.clip(X[:, 4], 20.0, 2000.0)  # alkaline_phosphotase
        X[:, 5] = np.clip(X[:, 5], 1.0, 2000.0)   # ALT
        X[:, 6] = np.clip(X[:, 6], 1.0, 5000.0)   # AST
        X[:, 7] = np.clip(X[:, 7], 2.0, 9.0)      # total_proteins
        X[:, 8] = np.clip(X[:, 8], 0.5, 5.5)      # albumin
        X[:, 9] = np.clip(X[:, 9], 0.1, 4.0)      # A/G ratio
        # Logit: elevated enzymes and bilirubin → liver patient
        logit = (
            0.4 * (X[:, 2] - 1.0)          # bilirubin
            + 0.005 * (X[:, 5] - 35)        # ALT
            + 0.004 * (X[:, 6] - 30)        # AST
            - 0.6 * (X[:, 8] - 3.3)         # low albumin → positive risk
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
