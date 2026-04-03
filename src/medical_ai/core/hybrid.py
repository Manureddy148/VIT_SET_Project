from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_first_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _clamp100(x: float) -> float:
    return 0.0 if x < 0.0 else 100.0 if x > 100.0 else x


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class RulesResult:
    severity_score: float
    confidence: float
    rag_query: str
    signals: Dict[str, Any]


def rules_diabetes_v1(query_text: str, structured: Dict[str, Any]) -> RulesResult:
    q = (query_text or "").lower()
    glucose = _num(structured.get("glucose")) or _extract_first_float(r"glucose\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
    hba1c = _num(structured.get("hba1c")) or _extract_first_float(r"hba1c\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
    bmi = _num(structured.get("bmi")) or _extract_first_float(r"bmi\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
    age = _num(structured.get("age")) or _extract_first_float(r"age\s*[:=]?\s*(\d+(?:\.\d+)?)", q)

    score = 15.0
    conf = 0.55
    signals: Dict[str, Any] = {"glucose": glucose, "hba1c": hba1c, "bmi": bmi, "age": age}

    # Symptom-based bumps
    if any(k in q for k in ["frequent urination", "polyuria", "excessive thirst", "polydipsia", "blurred vision"]):
        score += 15
    if "weight loss" in q:
        score += 10

    # Simple guideline-inspired thresholds (not diagnostic)
    if glucose is not None:
        if glucose >= 200:
            score += 45
            conf += 0.15
        elif glucose >= 126:
            score += 30
            conf += 0.10
        elif glucose >= 100:
            score += 15
            conf += 0.05

    if hba1c is not None:
        if hba1c >= 6.5:
            score += 45
            conf += 0.15
        elif hba1c >= 5.7:
            score += 20
            conf += 0.08

    if bmi is not None:
        if bmi >= 35:
            score += 15
            conf += 0.05
        elif bmi >= 30:
            score += 10
            conf += 0.04

    if age is not None and age >= 45:
        score += 5

    score = _clamp100(score)
    conf = _clamp01(conf)
    rag_query = "diabetes risk thresholds glucose hba1c bmi symptoms guidance"

    return RulesResult(
        severity_score=round(score, 1),
        confidence=round(conf, 3),
        rag_query=rag_query,
        signals={"rule_version": "rules_diabetes_v1", **signals},
    )


def rules_heart_v1(query_text: str, structured: Dict[str, Any]) -> RulesResult:
    q = (query_text or "").lower()
    systolic = _num(structured.get("systolic_bp")) or _num(structured.get("blood_pressure")) or _extract_first_float(
        r"(?:sbp|systolic)\s*[:=]?\s*(\d+(?:\.\d+)?)", q
    )
    cholesterol = _num(structured.get("cholesterol")) or _extract_first_float(r"cholesterol\s*[:=]?\s*(\d+(?:\.\d+)?)", q)
    age = _num(structured.get("age")) or _extract_first_float(r"age\s*[:=]?\s*(\d+(?:\.\d+)?)", q)

    score = 15.0
    conf = 0.55
    signals: Dict[str, Any] = {"systolic_bp": systolic, "cholesterol": cholesterol, "age": age}

    # Red flags / symptoms
    if any(k in q for k in ["chest pain", "pressure in chest", "tightness in chest"]):
        score += 25
        conf += 0.10
    if any(k in q for k in ["shortness of breath", "breathless", "dyspnea"]):
        score += 15
    if any(k in q for k in ["radiating", "left arm", "jaw pain", "sweating", "diaphoresis"]):
        score += 20
        conf += 0.10

    # Simple risk factor thresholds
    if systolic is not None:
        if systolic >= 180:
            score += 25
            conf += 0.10
        elif systolic >= 140:
            score += 15
            conf += 0.06
        elif systolic >= 130:
            score += 10
            conf += 0.04

    if cholesterol is not None:
        if cholesterol >= 240:
            score += 15
            conf += 0.05
        elif cholesterol >= 200:
            score += 10
            conf += 0.04

    if age is not None and age >= 50:
        score += 8
        conf += 0.03

    score = _clamp100(score)
    conf = _clamp01(conf)
    rag_query = "acute coronary syndrome chest pain red flags blood pressure cholesterol guidance"

    return RulesResult(
        severity_score=round(score, 1),
        confidence=round(conf, 3),
        rag_query=rag_query,
        signals={"rule_version": "rules_heart_v1", **signals},
    )


_ZERO_SHOT_PIPE = None


def hf_zero_shot_v1(query_text: str, candidate_labels: List[str], hf_id: str) -> Tuple[float, float, Dict[str, Any], str]:
    """
    Returns: severity_score(0..100), confidence(0..1), signals, rag_query
    """
    global _ZERO_SHOT_PIPE

    if os.getenv("MEDAI_DISABLE_HF", "").strip() == "1":
        # Keep E2E runnable in environments without model downloads.
        return 50.0, 0.2, {"disabled": True, "labels": candidate_labels}, "medical guidance evidence"

    from transformers import pipeline

    if _ZERO_SHOT_PIPE is None or getattr(_ZERO_SHOT_PIPE, "_medai_model_id", None) != hf_id:
        _ZERO_SHOT_PIPE = pipeline("zero-shot-classification", model=hf_id, device=-1)
        setattr(_ZERO_SHOT_PIPE, "_medai_model_id", hf_id)

    out = _ZERO_SHOT_PIPE(query_text, candidate_labels=candidate_labels, multi_label=True)
    labels = out.get("labels", [])
    scores = out.get("scores", [])
    by_label = {l: float(s) for l, s in zip(labels, scores)}

    # Map to severity using label weights
    # "high risk" dominates; otherwise "moderate", else "low".
    high = by_label.get("high risk", 0.0)
    moderate = by_label.get("moderate risk", 0.0)
    low = by_label.get("low risk", 0.0)

    severity = high * 90.0 + moderate * 60.0 + low * 20.0
    conf = max(high, moderate, low)
    rag_query = f"{candidate_labels[0]} clinical guidance risk factors"

    return (
        round(_clamp100(severity), 1),
        round(_clamp01(conf), 3),
        {"by_label": by_label, "hf_task": "zero-shot-classification"},
        rag_query,
    )

