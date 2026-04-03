from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from .schemas import EnsembleResult, ModelRunResult


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _clamp100(x: float) -> float:
    return 0.0 if x < 0.0 else 100.0 if x > 100.0 else x


def weighted_mean_fusion(
    domain: str,
    runs: List[ModelRunResult],
    weights: List[float],
    disagreement_threshold: float,
) -> EnsembleResult:
    if not runs:
        raise ValueError("No model runs to fuse")
    if len(runs) != len(weights):
        raise ValueError("weights length must match runs length")

    w_sum = sum(weights) or 1.0
    w_norm = [w / w_sum for w in weights]

    score = sum(r.severity_score * w for r, w in zip(runs, w_norm))
    conf = sum(_clamp01(r.confidence) * w for r, w in zip(runs, w_norm))

    # Disagreement: normalized average absolute deviation around fused score
    avg_abs_dev = sum(abs(r.severity_score - score) * w for r, w in zip(runs, w_norm))
    disagreement = _clamp01(avg_abs_dev / 50.0)  # 50 = "large" spread on 0-100

    notes = []
    if disagreement >= disagreement_threshold:
        notes.append("high_disagreement")

    return EnsembleResult(
        domain=domain,
        severity_score=round(_clamp100(score), 1),
        confidence=round(_clamp01(conf), 3),
        disagreement=round(disagreement, 3),
        models_used=[r.model_key for r in runs],
        runs=runs,
        notes=notes,
    )

