import re
from typing import Optional

from src.models.registry import ModelRegistry


def classify_domain_text(registry: ModelRegistry, text: str) -> str:
    return registry.classify_domain(text)


def infer_domain_from_labs(text: str, clinical: dict) -> Optional[str]:
    """Light heuristic when free text is empty but structured labs imply a domain."""
    t = text.lower()
    if any(k in clinical for k in ("glucose", "bmi", "insulin")) or "glucose" in t:
        return "diabetes"
    if any(k in clinical for k in ("cholesterol", "resting_bp", "max_hr")):
        return "heart_disease"
    if any(k in clinical for k in ("spo2", "respiratory_rate", "crp")):
        return "pneumonia"
    return None
