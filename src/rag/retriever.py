from typing import Any, Dict, List, Optional

from src.models.base_model import PredictionResult


_FALLBACK_CITATIONS: Dict[str, List[Dict[str, Any]]] = {
    "diabetes": [
        {
            "source": "ADA Standards of Care (summary)",
            "text": "Persistent hyperglycemia and elevated BMI are major risk factors for diabetes progression and complications.",
            "domain": "diabetes",
            "distance": 0.0,
        }
    ],
    "heart_disease": [
        {
            "source": "ACC/AHA Prevention Guidance (summary)",
            "text": "Elevated LDL cholesterol, hypertension and age are associated with increased cardiovascular event risk.",
            "domain": "heart_disease",
            "distance": 0.0,
        }
    ],
    "pneumonia": [
        {
            "source": "CAP Severity Review (summary)",
            "text": "Hypoxemia, tachypnea and inflammatory markers are key indicators of pneumonia severity and escalation need.",
            "domain": "pneumonia",
            "distance": 0.0,
        }
    ],
    "ckd": [
        {
            "source": "KDIGO CKD Guidance (summary)",
            "text": "Serum creatinine elevation and reduced kidney function markers correlate with CKD severity and progression.",
            "domain": "ckd",
            "distance": 0.0,
        }
    ],
    "sepsis": [
        {
            "source": "Surviving Sepsis Campaign (summary)",
            "text": "Hypotension, elevated lactate and organ dysfunction features are associated with higher sepsis mortality risk.",
            "domain": "sepsis",
            "distance": 0.0,
        }
    ],
    "liver_disease": [
        {
            "source": "Liver Disease Severity Review (summary)",
            "text": "Elevated bilirubin/transaminases and low albumin are clinically associated with worsening liver dysfunction.",
            "domain": "liver_disease",
            "distance": 0.0,
        }
    ],
}


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
        return _FALLBACK_CITATIONS.get(prediction.disease_domain, [])[:n_results]
    q = shap_to_rag_query(prediction.shap_values, prediction.disease_domain)
    try:
        results = store.query(q, n_results=n_results, domain_filter=prediction.disease_domain)
        if results:
            return results
        # Fallback: if domain-specific index is sparse, retrieve global medical evidence.
        global_results = store.query(q, n_results=n_results, domain_filter=None)
        if global_results:
            return global_results
        return _FALLBACK_CITATIONS.get(prediction.disease_domain, [])[:n_results]
    except Exception:
        return _FALLBACK_CITATIONS.get(prediction.disease_domain, [])[:n_results]
