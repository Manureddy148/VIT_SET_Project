from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Medical AI E2E (open-source, no training)")
    parser.add_argument("--query", required=True, help="User query / symptoms context")
    parser.add_argument("--image", default=None, help="Optional image path (for imaging domains)")
    parser.add_argument("--modality", default=None, help="Optional modality hint: xray|ct|mri|us|oct|derm|path")
    parser.add_argument("--domain", default=None, help="Optional explicit domain key")
    args = parser.parse_args()

    os.environ.setdefault("MODEL_REGISTRY_PATH", str(_repo_root() / "configs" / "model_registry.json"))
    os.environ.setdefault("KNOWLEDGE_BASE_PATH", str(_repo_root() / "configs" / "knowledge_base.json"))

    import sys

    sys.path.append(str(_repo_root() / "src"))

    from medical_ai.core.registry import ModelRegistry
    from medical_ai.core.schemas import AnalyzeRequest
    from medical_ai.core.rag import LocalKnowledgeBase
    from medical_ai.core.safety import safety_rails
    from medical_ai.core.synthesis import generate_report

    reg = ModelRegistry(config_path=os.environ["MODEL_REGISTRY_PATH"])
    kb = LocalKnowledgeBase.from_json(os.environ["KNOWLEDGE_BASE_PATH"])

    req = AnalyzeRequest(
        query_text=args.query,
        domain_hint=args.domain,
        image_path=args.image,
        modality=args.modality,
    )

    result = reg.run(req)

    rag_query = args.query
    if result.runs:
        rag_query = str((result.runs[0].signals or {}).get("rag_query") or rag_query)
    evidence = kb.query(rag_query, domain=result.domain, k=3)
    safety = safety_rails(result.severity_score, result.domain)
    report = generate_report(
        query_text=args.query,
        domain=result.domain,
        severity_score=result.severity_score,
        confidence=result.confidence,
        disagreement=result.disagreement,
        evidence=evidence,
        safety=safety,
    )

    print("\n=== FINAL REPORT ===\n")
    print(report.summary)
    print()
    print(report.explanation)
    print()
    print("Disclaimer:", report.safety.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

