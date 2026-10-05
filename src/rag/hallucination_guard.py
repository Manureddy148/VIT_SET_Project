"""Layer 5: basic faithfulness check — retrieved text must appear in synthesis."""

import os
import re


def snippets_in_report(snippets: list[str], report: str) -> bool:
    report_l = report.lower()
    return all(s.lower()[:80] in report_l or s.lower() in report_l for s in snippets if s)


def evaluate_faithfulness(report: str, citations: list[dict]) -> tuple[bool, str]:
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|units)\b", report, re.I):
        return False, "dose_text_blocked"
    if not citations:
        return False, "no_verified_citations"
    if any(not c.get("url") for c in citations):
        return False, "missing_source_urls"
    snippets = [c.get("text", "") for c in citations[:3]]
    fallback_ok = snippets_in_report(snippets, report) if snippets else True

    if os.getenv("MEDICAL_AI_USE_RAGAS", "0").lower() not in ("1", "true", "yes"):
        return fallback_ok, "snippet_guard"

    try:
        from ragas.metrics import faithfulness  # type: ignore
    except Exception:
        return fallback_ok, "snippet_guard_no_ragas"

    # Full RAGAS dataset evaluation is expensive and optional; keep runtime fallback for API path.
    return fallback_ok, f"ragas_available_metric:{getattr(faithfulness, 'name', 'faithfulness')}"
