from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .registry import ModelRegistry
from .schemas import AnalyzeRequest, EnsembleResult
from .rag import LocalKnowledgeBase
from .safety import safety_rails
from .synthesis import FinalReport, generate_report
from .orchestrator import build_orchestrator_output


def default_paths() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[3]
    reg = str(root / "configs" / "model_registry.json")
    kb = str(root / "configs" / "knowledge_base.json")
    return reg, kb


def run_full_analyze(
    req: AnalyzeRequest,
    *,
    registry_path: str | None = None,
    knowledge_path: str | None = None,
) -> Dict[str, Any]:
    reg_path = registry_path or os.getenv("MODEL_REGISTRY_PATH") or default_paths()[0]
    kb_path = knowledge_path or os.getenv("KNOWLEDGE_BASE_PATH") or default_paths()[1]

    reg = ModelRegistry(config_path=reg_path)
    kb = LocalKnowledgeBase.from_json(kb_path)

    resolved_domain = reg.resolve_domain(req)
    orch = build_orchestrator_output(
        query_text=req.query_text,
        structured_input=req.structured_input or {},
        resolved_domain=resolved_domain,
        image_path=req.image_path,
    )

    result = reg.run(req)

    rag_query = req.query_text
    if result.runs:
        rag_query = str((result.runs[0].signals or {}).get("rag_query") or rag_query)
    evidence = kb.query(rag_query, domain=result.domain, k=3)
    safety = safety_rails(result.severity_score, result.domain)
    report = generate_report(
        query_text=req.query_text,
        domain=result.domain,
        severity_score=result.severity_score,
        confidence=result.confidence,
        disagreement=result.disagreement,
        evidence=evidence,
        safety=safety,
    )

    return assemble_response(result=result, evidence=evidence, report=report, orchestrator=orch)


def assemble_response(
    *,
    result: EnsembleResult,
    evidence: list,
    report: FinalReport,
    orchestrator: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "domain": result.domain,
        "severity_score": result.severity_score,
        "confidence": result.confidence,
        "disagreement": result.disagreement,
        "models_used": result.models_used,
        "notes": result.notes,
        "orchestrator": orchestrator,
        "report": {
            "summary": report.summary,
            "explanation": report.explanation,
            "grounding_note": report.grounding_note,
            "safety": {
                "disclaimer": report.safety.disclaimer,
                "flags": report.safety.flags,
                "recommended_action": report.safety.recommended_action,
            },
            "citations": [
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "title": c.title,
                    "score": c.score,
                    "snippet": c.snippet,
                }
                for c in report.citations
            ],
        },
        "runs": [
            {
                "model_key": r.model_key,
                "model_id": r.model_id,
                "severity_score": r.severity_score,
                "confidence": r.confidence,
                "signals": r.signals,
            }
            for r in result.runs
        ],
    }
