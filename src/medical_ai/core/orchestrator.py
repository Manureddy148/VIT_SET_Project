"""Deterministic orchestrator output (JSON-serializable). See configs/orchestrator_output.schema.json."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _extract_first_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_entities(query_text: str, structured: Dict[str, Any]) -> Dict[str, Any]:
    q = query_text or ""
    s = structured or {}
    entities: Dict[str, Any] = {}

    for key in ("glucose", "hba1c", "bmi", "age", "cholesterol", "systolic_bp", "blood_pressure"):
        v = s.get(key)
        if v is not None:
            entities[key] = v

    if "glucose" not in entities:
        g = _extract_first_float(r"glucose\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
        if g is not None:
            entities["glucose"] = g
    if "hba1c" not in entities:
        h = _extract_first_float(r"hba1c\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
        if h is not None:
            entities["hba1c"] = h
    if "bmi" not in entities:
        b = _extract_first_float(r"bmi\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
        if b is not None:
            entities["bmi"] = b
    if "age" not in entities:
        a = _extract_first_float(r"age\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
        if a is not None:
            entities["age"] = a
    if "cholesterol" not in entities:
        c = _extract_first_float(r"cholesterol\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
        if c is not None:
            entities["cholesterol"] = c
    if "systolic_bp" not in entities and "blood_pressure" not in entities:
        sbp = _extract_first_float(r"(?:sbp|systolic)\s*[:=]?\s*(\d+(?:\.\d+)?)", "")
        if sbp is None:
            sbp = _extract_first_float(r"bp\s*[:=]?\s*(\d+)\s*/", q)
        if sbp is not None:
            entities["systolic_bp"] = sbp

    return entities


def _score_domains(query_text: str, structured: Dict[str, Any]) -> List[Tuple[str, float]]:
    q = (query_text or "").lower()
    scores: List[Tuple[str, float]] = []

    def add(domain: str, w: float) -> None:
        scores.append((domain, w))

    if any(k in q for k in ["glucose", "hba1c", "blood sugar", "diabetes", "insulin"]):
        add("hybrid_diabetes_risk", 10.0)
    if any(k in q for k in ["chest pain", "heart", "cardiac", "ecg", "cholesterol"]) or "blood pressure" in q or re.search(
        r"\bbp\b", q
    ):
        add("hybrid_heart_risk", 10.0)
    if any(k in q for k in ["skin lesion", "melanoma", "mole", "dermoscopy", "skin cancer", "derm"]):
        add("imaging_skin_lesion", 10.0)
    if any(k in q for k in ["x-ray", "xray", "chest x", "cxr"]):
        add("imaging_xray_pneumonia", 10.0)
    if "mri" in q or "brain mri" in q:
        add("imaging_mri_brain_tumor", 10.0)

    if structured.get("glucose") is not None or structured.get("hba1c") is not None:
        add("hybrid_diabetes_risk", 6.0)
    if structured.get("systolic_bp") is not None or structured.get("cholesterol") is not None:
        add("hybrid_heart_risk", 6.0)

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def suggest_missing(
    domain: str,
    entities: Dict[str, Any],
    query_text: str,
    *,
    image_path: Optional[str] = None,
) -> List[str]:
    out: List[str] = []
    q = (query_text or "").lower()

    if domain.startswith("imaging_") and not image_path:
        out.append("Provide an image_path (or upload) for imaging domains.")

    if domain == "hybrid_diabetes_risk":
        if entities.get("glucose") is None and "glucose" not in q and entities.get("hba1c") is None:
            out.append("Ask for fasting glucose or HbA1c if available.")
        if entities.get("bmi") is None and "bmi" not in q:
            out.append("Ask for BMI or height/weight if relevant.")
    elif domain == "hybrid_heart_risk":
        if entities.get("systolic_bp") is None and "blood pressure" not in q and " bp " not in f" {q} ":
            out.append("Ask for blood pressure if available.")
        if entities.get("cholesterol") is None:
            out.append("Ask for lipid panel (total cholesterol) if available.")
    elif domain.startswith("imaging_"):
        if "lesion" not in q and "skin" not in q and domain == "imaging_skin_lesion":
            out.append("Confirm lesion duration, change in size/color, and provide a dermoscopy or clinical photo if applicable.")

    return out


def build_orchestrator_output(
    *,
    query_text: str,
    structured_input: Dict[str, Any],
    resolved_domain: str,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    entities = extract_entities(query_text, structured_input)
    ranked = _score_domains(query_text, structured_input)
    missing = suggest_missing(resolved_domain, entities, query_text, image_path=image_path)

    candidates = [{"domain": d, "score": float(s)} for d, s in ranked]
    if not any(c["domain"] == resolved_domain for c in candidates):
        candidates.insert(0, {"domain": resolved_domain, "score": 0.0})

    return {
        "version": 1,
        "intent": "clinical_risk_assessment",
        "entities": entities,
        "routing": {
            "resolved_domain": resolved_domain,
            "domain_candidates": candidates[:5],
        },
        "missing_data": missing,
        "tool_plan": [
            "orchestrator.extract_entities",
            f"model_registry.run:{resolved_domain}",
            "knowledge_base.retrieve",
            "synthesis.report",
        ],
    }
