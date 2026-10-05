"""Saved real-data pipeline with its own schema (CDC is NOT Pima)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.models.base_model import BaseMedicalModel, PredictionResult

class ResearchTabularModel(BaseMedicalModel):
    def __init__(self, path):
        bundle = joblib.load(Path(path))
        self.pipeline, self.feature_names, self.metadata = bundle['pipeline'], bundle['features'], bundle['metadata']
    @property
    def domain(self): return self.metadata['domain']
    @property
    def required_features(self): return self.feature_names
    def preprocess(self, raw_input):
        if not any(raw_input.get(f) is not None for f in self.feature_names):
            raise ValueError('No features match the trained dataset schema')
        return pd.DataFrame([[raw_input.get(f, np.nan) for f in self.feature_names]], columns=self.feature_names)
    def predict(self, features):
        p = float(self.pipeline.predict_proba(features)[0, 1])
        # SHAP below explains the underlying tree, not the calibrated probability.
        import xgboost as xgb
        base = self.pipeline.estimator
        # XGBoost native TreeSHAP avoids third-party loader incompatibilities.
        transformed = base.named_steps['imputer'].transform(features)
        values = base.named_steps['model'].get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[:, :-1]
        attribution = dict(zip(self.feature_names, values[0].astype(float).tolist()))
        score = round(100*p, 1)
        return PredictionResult(self.domain, score, max(p, 1-p), self.score_to_label(score), attribution,
                                sorted(attribution, key=lambda f: abs(attribution[f]), reverse=True)[:5])
