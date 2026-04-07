import os
from typing import Any, Dict, List, Optional

from src.models.base_model import PredictionResult


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _build_prompt_context(
    prediction: PredictionResult,
    query_text: str,
    citations: List[Dict[str, Any]],
) -> Dict[str, str]:
    tops = ", ".join(prediction.top_features[:5]) or "n/a"
    citation_block = []
    for i, c in enumerate(citations[:5], 1):
        src = c.get("source", "Unknown")
        text = (c.get("text") or "").replace("\n", " ")[:220]
        citation_block.append(f"[{i}] {src}: {text}")

    return {
        "domain": prediction.disease_domain,
        "severity_score": f"{prediction.severity_score:.1f}",
        "severity_label": prediction.severity_label,
        "confidence": f"{prediction.confidence:.3f}",
        "top_features": tops,
        "query_text": query_text[:600],
        "citations": "\n".join(citation_block) if citation_block else "No retrieved evidence.",
    }


def generate_llm_clinical_report(
    prediction: PredictionResult,
    query_text: str,
    citations: List[Dict[str, Any]],
) -> Optional[str]:
    """Generate Layer-6 narrative via LangChain + Groq when enabled.

    Returns None when disabled/unavailable so caller can safely fall back to template synthesis.
    """
    if not _env_flag("MEDICAL_AI_USE_LLM", "0"):
        return None

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq
    except Exception:
        return None

    model_name = os.getenv("MEDICAL_AI_LLM_MODEL", "llama-3.1-8b-instant")
    temperature = float(os.getenv("MEDICAL_AI_LLM_TEMPERATURE", "0.2"))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical AI assistant. Produce a concise, evidence-grounded summary. "
                "Do not provide medication doses. Avoid definitive diagnosis language.",
            ),
            (
                "human",
                "Domain: {domain}\n"
                "Severity score: {severity_score}/100 ({severity_label})\n"
                "Confidence: {confidence}\n"
                "Top SHAP risk factors: {top_features}\n"
                "Input summary: {query_text}\n\n"
                "Retrieved evidence:\n{citations}\n\n"
                "Write a short report with sections: 1) Clinical summary 2) Evidence basis 3) Safety note.",
            ),
        ]
    )

    llm = ChatGroq(model=model_name, temperature=temperature, api_key=api_key)
    chain = prompt | llm
    context = _build_prompt_context(prediction, query_text, citations)
    response = chain.invoke(context)
    content = getattr(response, "content", "")
    return content.strip() if isinstance(content, str) and content.strip() else None