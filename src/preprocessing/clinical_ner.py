"""Layer 2: lightweight pattern extraction (replace with NER in production)."""

import re
from typing import Dict


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
    ]
    for pat, key in patterns:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        if key == "resting_bp":
            out[key] = float(m.group(1))
        else:
            out[key] = float(m.group(1))
    return out
