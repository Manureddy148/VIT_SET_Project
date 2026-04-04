from typing import Any, Dict, List, Optional

from src.models.base_model import PredictionResult


def shap_to_rag_query(shap_values: Dict[str, float], domain: str) -> str:
    top_features = sorted(shap_values, key=lambda k: abs(shap_values[k]), reverse=True)[:3]
    parts = [domain.replace("_", " ")]
    for feat in top_features:
        direction = "high" if shap_values[feat] > 0 else "low"
        label = feat.replace("_", " ")
        parts.append(f"{direction} {label}")
    return " ".join(parts) + " clinical risk factors severity"


def retrieve_literature(
    store: Any,
    prediction: PredictionResult,
    n_results: int = 4,
) -> List[Dict[str, Any]]:
    if store is None:
        return []
    q = shap_to_rag_query(prediction.shap_values, prediction.disease_domain)
    try:
        return store.query(q, n_results=n_results, domain_filter=None)
    except Exception:
        return []
