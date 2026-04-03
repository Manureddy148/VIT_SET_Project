from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fusion import weighted_mean_fusion
from .schemas import AnalyzeRequest, DomainSpec, EnsembleResult, ModelRunResult, ModelSpec


@dataclass(frozen=True)
class RegistryPolicy:
    min_models_per_domain: int = 2
    max_models_per_domain: int = 3
    disagreement_threshold: float = 0.35
    default_fusion: str = "weighted_mean"


class ModelRegistry:
    def __init__(self, config_path: str):
        self.config_path = str(config_path)
        self.policy = RegistryPolicy()
        self.domains: Dict[str, DomainSpec] = {}

        self._load_config()

    def _load_config(self) -> None:
        raw = json.loads(Path(self.config_path).read_text(encoding="utf-8"))

        policy = raw.get("policy", {}) or {}
        self.policy = RegistryPolicy(
            min_models_per_domain=int(policy.get("min_models_per_domain", 2)),
            max_models_per_domain=int(policy.get("max_models_per_domain", 3)),
            disagreement_threshold=float(policy.get("disagreement_threshold", 0.35)),
            default_fusion=str(policy.get("default_fusion", "weighted_mean")),
        )

        domains: Dict[str, Any] = raw.get("domains", {}) or {}
        parsed: Dict[str, DomainSpec] = {}
        for domain_key, spec in domains.items():
            models: List[ModelSpec] = []
            for m in (spec.get("models", []) or []):
                models.append(
                    ModelSpec(
                        key=str(m["key"]),
                        provider=str(m.get("provider", "huggingface")),
                        hf_id=m.get("hf_id"),
                        weight=float(m.get("weight", 1.0)),
                        extra={k: v for k, v in m.items() if k not in {"key", "provider", "hf_id", "weight"}},
                    )
                )

            parsed[domain_key] = DomainSpec(
                key=domain_key,
                type=str(spec.get("type", "text")),  # type: ignore[arg-type]
                description=str(spec.get("description", "")),
                models=models,
            )

        self.domains = parsed

    def list_domains(self) -> List[str]:
        return sorted(self.domains.keys())

    def resolve_domain(self, req: AnalyzeRequest) -> str:
        if req.domain_hint and req.domain_hint in self.domains:
            return req.domain_hint

        q = (req.query_text or "").lower()

        # Minimal heuristic routing for imaging
        if any(k in q for k in ["x-ray", "xray", "chest x", "cxr"]):
            if "imaging_xray_pneumonia" in self.domains:
                return "imaging_xray_pneumonia"
        if "mri" in q or "brain mri" in q:
            if "imaging_mri_brain_tumor" in self.domains:
                return "imaging_mri_brain_tumor"

        # Fallback: first domain (keeps demo running)
        if self.domains:
            return next(iter(self.domains.keys()))

        raise ValueError("No domains configured in registry")

    def run(self, req: AnalyzeRequest) -> EnsembleResult:
        domain_key = self.resolve_domain(req)
        domain = self.domains[domain_key]

        if domain.type == "imaging":
            runs, weights = self._run_imaging_domain(domain, req)
        else:
            raise NotImplementedError(f"Domain type not implemented yet: {domain.type}")

        return weighted_mean_fusion(
            domain=domain.key,
            runs=runs,
            weights=weights,
            disagreement_threshold=self.policy.disagreement_threshold,
        )

    def _run_imaging_domain(self, domain: DomainSpec, req: AnalyzeRequest) -> tuple[list[ModelRunResult], list[float]]:
        if not req.image_path:
            raise ValueError("image_path is required for imaging domains")

        # Import locally to avoid forcing imaging deps for non-imaging use.
        from ...imaging_extension import ImageRouter  # type: ignore

        router = ImageRouter()

        runs: list[ModelRunResult] = []
        weights: list[float] = []

        modality = req.modality
        if modality is None:
            # allow auto-detect in imaging_extension
            modality = None

        for model in domain.models[: self.policy.max_models_per_domain]:
            # imaging_extension expects model override by key (registry key), not hf_id
            result = router.analyze(req.image_path, modality=modality, model=model.key)

            runs.append(
                ModelRunResult(
                    domain=domain.key,
                    model_key=model.key,
                    model_id=model.hf_id or model.key,
                    severity_score=float(result.severity_score),
                    confidence=float(result.confidence),
                    signals={
                        "severity_label": result.severity_label,
                        "rag_query": result.rag_query,
                        "raw_predictions": result.raw_predictions,
                    },
                )
            )
            weights.append(float(model.weight))

        if len(runs) < self.policy.min_models_per_domain:
            raise ValueError(
                f"Domain '{domain.key}' must run at least {self.policy.min_models_per_domain} models; "
                f"configured={len(domain.models)}, ran={len(runs)}"
            )

        return runs, weights

