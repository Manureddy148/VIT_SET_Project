from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


DomainType = Literal["imaging", "text", "tabular"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    hf_id: Optional[str] = None
    weight: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainSpec:
    key: str
    type: DomainType
    description: str
    models: List[ModelSpec]


@dataclass(frozen=True)
class AnalyzeRequest:
    query_text: str
    domain_hint: Optional[str] = None
    structured_input: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None
    modality: Optional[str] = None
    patient_id: Optional[str] = None


@dataclass(frozen=True)
class ModelRunResult:
    domain: str
    model_key: str
    model_id: str
    severity_score: float  # 0..100
    confidence: float  # 0..1
    signals: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleResult:
    domain: str
    severity_score: float  # 0..100
    confidence: float  # 0..1
    disagreement: float  # 0..1 (higher = more disagreement)
    models_used: List[str]
    runs: List[ModelRunResult]
    notes: List[str] = field(default_factory=list)

