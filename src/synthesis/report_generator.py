from typing import Any, Dict, List

from src.models.base_model import PredictionResult
from src.synthesis.llm_reporter import generate_llm_clinical_report
from src.synthesis.safety_rails import MEDICAL_DISCLAIMER


def build_clinical_report(
    prediction: PredictionResult,
    query_text: str,
    citations: List[Dict[str, Any]],
) -> str:
    llm_report = generate_llm_clinical_report(prediction, query_text, citations)
    if llm_report:
        return f"{llm_report}\n\n{MEDICAL_DISCLAIMER}"

    tops = ", ".join(prediction.top_features[:5]) or "n/a"
    cite_lines = []
    for i, c in enumerate(citations[:5], 1):
        src = c.get("source", "Unknown")
        excerpt = (c.get("text") or "")[:280].replace("\n", " ")
        cite_lines.append(f"[{i}] {src}: {excerpt}")
    cites_block = "\n".join(cite_lines) if cite_lines else "No literature retrieved."

    return (
        f"Severity: {prediction.severity_score:.1f}/100 ({prediction.severity_label}) "
        f"for domain '{prediction.disease_domain}'. "
        f"Primary modeled drivers (SHAP-ranked): {tops}.\n\n"
        f"Input summary: {query_text[:400]}\n\n"
        f"Evidence excerpts:\n{cites_block}\n\n"
        f"{MEDICAL_DISCLAIMER}"
    )
