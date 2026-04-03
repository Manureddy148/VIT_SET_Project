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

    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))

    from medical_ai.core.schemas import AnalyzeRequest
    from medical_ai.core.pipeline import run_full_analyze

    req = AnalyzeRequest(
        query_text=args.query,
        domain_hint=args.domain,
        structured_input={},
        image_path=args.image,
        modality=args.modality,
    )

    out = run_full_analyze(req)

    print("\n=== ORCHESTRATOR ===\n")
    print(out.get("orchestrator", {}))

    print("\n=== FINAL REPORT ===\n")
    rep = out.get("report", {})
    print(rep.get("summary", ""))
    print()
    print(rep.get("explanation", ""))
    print()
    print("Disclaimer:", (rep.get("safety") or {}).get("disclaimer", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

