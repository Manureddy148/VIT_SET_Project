from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core.pipeline import run_full_analyze
from ..core.schemas import AnalyzeRequest


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
    from ..core.registry import ModelRegistry

    reg = ModelRegistry(config_path=os.getenv("MODEL_REGISTRY_PATH", _default_registry_path()))
    return {"domains": reg.list_domains()}


@app.get("/models")
def models() -> Dict[str, Any]:
    """Full registry introspection (per docs/ROADMAP Phase 5)."""
    path = os.getenv("MODEL_REGISTRY_PATH", _default_registry_path())
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.post("/analyze")
def analyze(payload: AnalyzePayload) -> Dict[str, Any]:
    try:
        req = AnalyzeRequest(
            query_text=payload.query_text,
            domain_hint=payload.domain_hint,
            structured_input=payload.structured_input,
            image_path=payload.image_path,
            modality=payload.modality,
            patient_id=payload.patient_id,
        )
        return run_full_analyze(
            req,
            registry_path=os.getenv("MODEL_REGISTRY_PATH", _default_registry_path()),
            knowledge_path=os.getenv("KNOWLEDGE_BASE_PATH", _default_kb_path()),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

