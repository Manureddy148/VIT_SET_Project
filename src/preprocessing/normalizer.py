from typing import Any, Dict


def normalize_clinical_dict(data: Dict[str, Any]) -> Dict[str, float]:
    """Coerce numeric values to float; skip non-numeric keys."""
    out: Dict[str, float] = {}
    for k, v in data.items():
        if v is None:
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out
