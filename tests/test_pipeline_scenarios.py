"""Scenario tests: hybrid + routing. Set MEDAI_DISABLE_HF=1 in CI to avoid HF downloads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _env():
    os.environ.setdefault("MEDAI_DISABLE_HF", "1")
    os.environ["MODEL_REGISTRY_PATH"] = str(ROOT / "configs" / "model_registry.json")
    os.environ["KNOWLEDGE_BASE_PATH"] = str(ROOT / "configs" / "knowledge_base.json")
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))


def _run(query: str, structured: dict | None = None, domain_hint: str | None = None):
    from medical_ai.core.pipeline import run_full_analyze
    from medical_ai.core.schemas import AnalyzeRequest

    req = AnalyzeRequest(
        query_text=query,
        structured_input=structured or {},
        domain_hint=domain_hint,
    )
    return run_full_analyze(req)


def test_diabetes_routing_and_scores():
    out = _run("Patient glucose 210 hba1c 7.2 frequent urination age 52", {"glucose": 210, "hba1c": 7.2, "age": 52})
    assert out["domain"] == "hybrid_diabetes_risk"
    assert 0 <= out["severity_score"] <= 100
    assert out["orchestrator"]["routing"]["resolved_domain"] == "hybrid_diabetes_risk"
    assert "entities" in out["orchestrator"]


def test_heart_routing():
    out = _run("chest pain and sweating left arm blood pressure 170 cholesterol 250 age 60")
    assert out["domain"] == "hybrid_heart_risk"
    assert out["report"]["citations"]  # domain-scoped knowledge exists


def test_domain_hint_overrides():
    out = _run("random unrelated text", domain_hint="hybrid_diabetes_risk")
    assert out["domain"] == "hybrid_diabetes_risk"


def test_orchestrator_schema_shape():
    out = _run("diabetes insulin resistance", {"bmi": 32})
    o = out["orchestrator"]
    assert o["version"] == 1
    assert "routing" in o and o["routing"]["resolved_domain"]
    assert isinstance(o.get("missing_data"), list)


def test_unrouteable_raises():
    from medical_ai.core.registry import ModelRegistry
    from medical_ai.core.schemas import AnalyzeRequest

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    reg = ModelRegistry(os.environ["MODEL_REGISTRY_PATH"])
    with pytest.raises(ValueError, match="Cannot route"):
        reg.run(AnalyzeRequest(query_text="hello world nothing medical"))


def test_knowledge_base_has_domains():
    from medical_ai.core.rag import LocalKnowledgeBase

    kb = LocalKnowledgeBase.from_json(os.environ["KNOWLEDGE_BASE_PATH"])
    tags = set()
    for d in kb.docs:
        tags.update(d.domain_tags)
    for need in ("hybrid_diabetes_risk", "hybrid_heart_risk", "imaging_xray_pneumonia", "imaging_skin_lesion"):
        assert need in tags
