from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .rag import EvidenceHit
from .safety import SafetyResult


@dataclass(frozen=True)
class FinalReport:
    summary: str
    explanation: str
    citations: List[EvidenceHit]
    safety: SafetyResult


def generate_report(
    *,
    query_text: str,
    domain: str,
    severity_score: float,
    confidence: float,
    disagreement: float,
    evidence: List[EvidenceHit],
    safety: SafetyResult,
) -> FinalReport:
    summary = (
        f"Domain: {domain}. Severity score: {severity_score}/100. "
        f"Confidence: {confidence}. Disagreement: {disagreement}."
    )

    evidence_lines = ""
    if evidence:
        evidence_lines = "\n".join(
            [f"- {e.title} ({e.source})" for e in evidence]
        )
    else:
        evidence_lines = "- No local citations available for this query/domain."

    explanation = (
        "This result was produced by running multiple open-source models for the selected domain and fusing their outputs. "
        "Evidence below is retrieved from a small curated local knowledge base (expandable).\n\n"
        f"User query: {query_text}\n\n"
        "Evidence used:\n"
        f"{evidence_lines}\n\n"
        f"Recommended action: {safety.recommended_action}"
    )

    return FinalReport(
        summary=summary,
        explanation=explanation,
        citations=evidence,
        safety=safety,
    )

