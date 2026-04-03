from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SafetyResult:
    disclaimer: str
    flags: List[str]
    recommended_action: str


def safety_rails(severity_score: float, domain: str) -> SafetyResult:
    flags: List[str] = []

    disclaimer = (
        "AI-assisted output for informational purposes only. "
        "Not a medical diagnosis. Seek professional medical advice."
    )

    recommended_action = "Consider consulting a clinician for interpretation."

    if severity_score >= 75:
        flags.append("high_severity")
        recommended_action = "High-risk output: seek urgent medical evaluation, especially if symptoms are severe."
    elif severity_score >= 50:
        flags.append("moderate_severity")
        recommended_action = "Moderate-risk output: consider clinical review and follow-up."

    if domain.startswith("imaging_"):
        flags.append("imaging_requires_specialist_review")

    return SafetyResult(disclaimer=disclaimer, flags=flags, recommended_action=recommended_action)

