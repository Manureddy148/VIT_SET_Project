from typing import Any, Dict, Mapping, Sequence

import numpy as np


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        return arr.reshape(arr.shape[0], -1)
    return arr.reshape(1, -1)


def _fallback_attribution(
    features: np.ndarray,
    feature_names: Sequence[str],
    medians: Mapping[str, float] | None,
    scaler: Any = None,
) -> Dict[str, float]:
    raw = features
    try:
        if scaler is not None:
            raw = scaler.inverse_transform(features)
    except Exception:
        raw = features

    row = np.asarray(raw[0], dtype=np.float64)
    out: Dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        baseline = 0.0
        if medians is not None:
            baseline = float(medians.get(name, 0.0))
        denom = max(abs(baseline), 1.0)
        # Signed deviation from domain baseline; clipped for stability.
        score = float(np.clip((row[idx] - baseline) / denom, -3.0, 3.0))
        out[str(name)] = round(score, 6)
    return out


def compute_feature_attributions(
    model: Any,
    features: np.ndarray,
    feature_names: Sequence[str],
    medians: Mapping[str, float] | None,
    scaler: Any = None,
) -> Dict[str, float]:
    """Return SHAP attributions when available, else deterministic fallback values."""
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(features)
        if isinstance(vals, list):
            vals = vals[-1]
        vals_2d = _as_2d(vals)
        row = vals_2d[0].tolist()
        if len(row) == len(feature_names):
            return dict(zip(feature_names, [float(v) for v in row]))
    except Exception:
        pass

    return _fallback_attribution(features, feature_names, medians, scaler=scaler)
