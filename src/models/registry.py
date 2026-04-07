from typing import Dict, List, Optional

from dataclasses import replace

from src.models.base_model import BaseMedicalModel, PredictionResult


class ModelRegistry:
    """
    Central registry of all disease models.
    Router selects the right model at inference time.
    """

    def __init__(self) -> None:
        self._models: Dict[str, BaseMedicalModel] = {}
        self._domain_keywords: Dict[str, List[str]] = {}

    def register(self, model: BaseMedicalModel, keywords: Optional[List[str]] = None) -> None:
        self._models[model.domain] = model
        self._domain_keywords[model.domain] = keywords or [model.domain]

    def classify_domain(self, text: str) -> str:
        text_lower = text.lower()
        scores = {
            domain: sum(kw in text_lower for kw in kws)
            for domain, kws in self._domain_keywords.items()
        }
        if not any(scores.values()):
            return "unknown"
        return max(scores, key=scores.get)

    def route_and_predict(self, query_text: str, patient_data: Dict) -> PredictionResult:
        domain = self.classify_domain(query_text)
        if domain == "unknown" or domain not in self._models:
            raise ValueError(
                f"Cannot route to a known disease model. "
                f"Available domains: {list(self._models.keys())}"
            )
        return self.predict_for_domain(domain, patient_data)

    def predict_for_domain(self, domain: str, patient_data: Dict) -> PredictionResult:
        if domain not in self._models:
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {list(self._models.keys())}"
            )
        model = self._models[domain]
        missing = model.get_missing_features(patient_data)
        features = model.preprocess(patient_data)
        result = model.predict(features)
        return replace(result, missing_features=missing)

    def list_domains(self) -> List[str]:
        return list(self._models.keys())

    def required_features_for_domain(self, domain: str) -> List[str]:
        model = self._models.get(domain)
        return list(model.required_features) if model is not None else []
