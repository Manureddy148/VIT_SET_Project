import re
from typing import Optional

from src.models.registry import ModelRegistry


def classify_domain_text(registry: ModelRegistry, text: str) -> str:
    return registry.classify_domain(text)


def infer_domain_from_labs(
    text: str,
    clinical: dict,
    registry: Optional[ModelRegistry] = None,
) -> Optional[str]:
    """Infer domain from structured labs.

    If a registry is provided, score domains by overlap with each model's
    required features so newly registered diseases can route without new
    hardcoded rules.
    """
    if registry is not None:
        clinical_keys = set(clinical.keys())
        best_domain = None
        best_score = 0.0
        for domain in registry.list_domains():
            req = set(registry.required_features_for_domain(domain))
            if not req:
                continue
            overlap = len(req & clinical_keys)
            if overlap == 0:
                continue
            # Weight: prioritize count first, ratio second.
            ratio = overlap / max(len(req), 1)
            score = overlap + ratio
            if score > best_score:
                best_score = score
                best_domain = domain
        if best_domain is not None:
            return best_domain

    """Light heuristic when free text is empty but structured labs imply a domain."""
    t = text.lower()
    if any(k in clinical for k in ("glucose", "bmi", "insulin")) or "glucose" in t:
        return "diabetes"
    if any(k in clinical for k in ("cholesterol", "resting_bp", "max_hr")):
        return "heart_disease"
    if any(k in clinical for k in ("spo2", "respiratory_rate", "crp")):
        return "pneumonia"
    if any(k in clinical for k in ("serum_creatinine", "blood_urea", "specific_gravity")):
        return "ckd"
    if any(k in clinical for k in ("lactate", "glasgow_coma_scale", "systolic_bp")):
        return "sepsis"
    if any(k in clinical for k in ("total_bilirubin", "alamine_aminotransferase", "aspartate_aminotransferase")):
        return "liver_disease"
    return None
