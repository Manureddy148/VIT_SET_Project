from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PredictionResult:
    """Standardized output from every model in the registry."""

    disease_domain: str
    severity_score: float
    confidence: float
    severity_label: str
    shap_values: Dict[str, float]
    top_features: List[str]
    missing_features: List[str] = field(default_factory=list)


class BaseMedicalModel(ABC):
    """Abstract contract every disease model must satisfy."""

    @property
    @abstractmethod
    def domain(self) -> str:
        pass

    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        pass

    @abstractmethod
    def preprocess(self, raw_input: Dict) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionResult:
        pass

    def score_to_label(self, score: float) -> str:
        if score < 25:
            return "Low"
        if score < 50:
            return "Moderate"
        if score < 75:
            return "High"
        return "Critical"

    def get_missing_features(self, raw_input: Dict) -> List[str]:
        return [
            f
            for f in self.required_features
            if f not in raw_input or raw_input[f] is None
        ]
