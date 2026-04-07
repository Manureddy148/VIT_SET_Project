"""Layer 2 clinical extraction: regex fallback + optional biomedical NER."""

import os
import re
from functools import lru_cache
from typing import Dict, List


NEGATION_TOKENS = ("no", "denies", "without", "negative for")


def _is_negated(text: str, phrase: str) -> bool:
    text_l = text.lower()
    phrase_l = phrase.lower()
    idx = text_l.find(phrase_l)
    if idx < 0:
        return False
    window = text_l[max(0, idx - 30):idx]
    return any(token in window for token in NEGATION_TOKENS)


@lru_cache(maxsize=1)
def _try_load_ner_pipeline():
    if os.getenv("MEDICAL_AI_USE_BIOMED_NER", "0").lower() not in ("1", "true", "yes"):
        return None
    try:
        from transformers import pipeline

        model = os.getenv("MEDICAL_AI_NER_MODEL", "d4data/biomedical-ner-all")
        return pipeline("token-classification", model=model, aggregation_strategy="simple")
    except Exception:
        return None


def extract_biomedical_entities(text: str) -> List[dict]:
    ner = _try_load_ner_pipeline()
    if ner is None or not text:
        return []
    try:
        entities = ner(text)
    except Exception:
        return []
    out: List[dict] = []
    for ent in entities:
        phrase = str(ent.get("word", "")).strip()
        if not phrase:
            continue
        out.append(
            {
                "label": str(ent.get("entity_group", "UNKNOWN")),
                "text": phrase,
                "score": float(ent.get("score", 0.0)),
                "negated": _is_negated(text, phrase),
            }
        )
    return out


def extract_labs_from_text(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not text:
        return out
    patterns = [
        (r"glucose\s*[:=]?\s*(\d+(?:\.\d+)?)", "glucose"),
        (r"bmi\s*[:=]?\s*(\d+(?:\.\d+)?)", "bmi"),
        (r"age\s*[:=]?\s*(\d+)", "age"),
        (r"hba1c\s*[:=]?\s*(\d+(?:\.\d+)?)", "hba1c"),
        (r"cholesterol\s*[:=]?\s*(\d+(?:\.\d+)?)", "cholesterol"),
        (r"blood pressure\s*[:=]?\s*(\d+)\s*/\s*(\d+)", "resting_bp"),
        (r"spo2\s*[:=]?\s*(\d+(?:\.\d+)?)", "spo2"),
        (r"respiratory rate\s*[:=]?\s*(\d+(?:\.\d+)?)", "respiratory_rate"),
        (r"crp\s*[:=]?\s*(\d+(?:\.\d+)?)", "crp"),
    ]
    for pat, key in patterns:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        out[key] = float(m.group(1))
    return out


def extract_negated_findings(text: str) -> List[str]:
    findings = ["chest pain", "fever", "cough", "shortness of breath"]
    return [f for f in findings if _is_negated(text, f)]
