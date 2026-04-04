from typing import Dict, List, Tuple

import numpy as np


class SHAPExplainer:
    """Centralized SHAP explainer for tree models; bridges Layer 4 → RAG queries."""

    def __init__(self, model, feature_names: List[str]) -> None:
        import shap

        self.feature_names = feature_names
        self._explainer = shap.TreeExplainer(model)

    def explain(self, features: np.ndarray, top_k: int = 5) -> Tuple[Dict[str, float], List[str]]:
        shap_values = self._explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        vals = shap_values[0]
        shap_dict = dict(zip(self.feature_names, vals.tolist()))
        top_features = sorted(shap_dict, key=lambda k: abs(shap_dict[k]), reverse=True)[:top_k]
        return shap_dict, top_features

    def features_to_rag_query(self, shap_dict: Dict[str, float], domain: str) -> str:
        top_features = sorted(shap_dict, key=lambda k: abs(shap_dict[k]), reverse=True)[:3]
        parts = [domain]
        for feat in top_features:
            direction = "high" if shap_dict[feat] > 0 else "low"
            label = feat.replace("_", " ")
            parts.append(f"{direction} {label}")
        return " ".join(parts) + " clinical risk factors severity"
