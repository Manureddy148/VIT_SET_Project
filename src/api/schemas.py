from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    query_text: str = ""
    clinical_data: Dict[str, float] = Field(default_factory=dict)
    patient_id: Optional[str] = None


class SeverityResponse(BaseModel):
    disease_domain: str
    severity_score: float
    severity_label: str
    confidence: float
    top_risk_factors: List[str]
    shap_values: Dict[str, float]
    missing_features: List[str]
    missing_feature_prompts: List[str] = Field(default_factory=list)
    literature_citations: List[Dict[str, Any]]
    faithfulness_passed: bool
    safety_flags: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    audit_log_id: Optional[str] = None
    explanation: str
    disclaimer: str
