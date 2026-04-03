from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core.registry import ModelRegistry
from ..core.schemas import AnalyzeRequest
from ..core.rag import LocalKnowledgeBase
from ..core.safety import safety_rails
from ..core.synthesis import generate_report


class AnalyzePayload(BaseModel):
    query_text: str = Field(..., description="Natural language symptoms / context")
    domain_hint: Optional[str] = Field(None, description="Optional explicit domain key")
    structured_input: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None
    modality: Optional[str] = Field(None, description="xray|ct|mri|us|oct|derm|path")
    patient_id: Optional[str] = None


app = FastAPI(title="Medical AI (Open-Source Inference)", version="0.1.0")


def _default_registry_path() -> str:
    # repo_root / configs / model_registry.json
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return str(repo_root / "configs" / "model_registry.json")

def _default_kb_path() -> str:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return str(repo_root / "configs" / "knowledge_base.json")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/domains")
def domains() -> Dict[str, Any]:
    reg = ModelRegistry(config_path=os.getenv("MODEL_REGISTRY_PATH", _default_registry_path()))
    return {"domains": reg.list_domains()}


@app.post("/analyze")
def analyze(payload: AnalyzePayload) -> Dict[str, Any]:
    try:
        reg = ModelRegistry(config_path=os.getenv("MODEL_REGISTRY_PATH", _default_registry_path()))
        kb = LocalKnowledgeBase.from_json(os.getenv("KNOWLEDGE_BASE_PATH", _default_kb_path()))
        req = AnalyzeRequest(
            query_text=payload.query_text,
            domain_hint=payload.domain_hint,
            structured_input=payload.structured_input,
            image_path=payload.image_path,
            modality=payload.modality,
            patient_id=payload.patient_id,
        )
        result = reg.run(req)

        # Evidence query: prefer model-provided rag_query when available
        rag_query = payload.query_text
        if result.runs:
            r0 = result.runs[0]
            rag_query = str((r0.signals or {}).get("rag_query") or payload.query_text)
        evidence = kb.query(rag_query, domain=result.domain, k=3)
        safety = safety_rails(result.severity_score, result.domain)
        report = generate_report(
            query_text=payload.query_text,
            domain=result.domain,
            severity_score=result.severity_score,
            confidence=result.confidence,
            disagreement=result.disagreement,
            evidence=evidence,
            safety=safety,
        )

        return {
            "domain": result.domain,
            "severity_score": result.severity_score,
            "confidence": result.confidence,
            "disagreement": result.disagreement,
            "models_used": result.models_used,
            "notes": result.notes,
            "report": {
                "summary": report.summary,
                "explanation": report.explanation,
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
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

