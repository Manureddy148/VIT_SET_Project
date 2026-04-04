from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.bootstrap import build_registry, seed_vector_store_if_empty, try_vector_store
from src.api.schemas import PatientInput, SeverityResponse
from src.preprocessing.clinical_ner import extract_labs_from_text
from src.preprocessing.domain_classifier import infer_domain_from_labs
from src.preprocessing.normalizer import normalize_clinical_dict
from src.rag.retriever import retrieve_literature
from src.synthesis.report_generator import build_clinical_report
from src.synthesis.safety_rails import MEDICAL_DISCLAIMER


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = build_registry()
    app.state.vector_store = try_vector_store()
    seed_vector_store_if_empty(app.state.vector_store)
    yield


app = FastAPI(
    title="Medical AI Severity API",
    description="7-layer clinical intelligence pipeline (demo scaffold)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=SeverityResponse)
def predict(patient: PatientInput) -> Any:
    registry = app.state.registry
    merged = dict(patient.clinical_data)
    merged.update(extract_labs_from_text(patient.query_text))
    merged = normalize_clinical_dict(merged)

    domain = registry.classify_domain(patient.query_text)
    if domain == "unknown":
        inferred = infer_domain_from_labs(patient.query_text, merged)
        if inferred:
            domain = inferred

    if domain == "unknown" or domain not in registry.list_domains():
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not infer disease domain from text or structured fields. "
                f"Try keywords for: {registry.list_domains()}"
            ),
        )

    try:
        result = registry.predict_for_domain(domain, merged)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    citations = retrieve_literature(app.state.vector_store, result)
    explanation = build_clinical_report(result, patient.query_text, citations)

    return SeverityResponse(
        disease_domain=result.disease_domain,
        severity_score=result.severity_score,
        severity_label=result.severity_label,
        confidence=result.confidence,
        top_risk_factors=result.top_features,
        shap_values=result.shap_values,
        missing_features=result.missing_features,
        literature_citations=citations,
        explanation=explanation,
        disclaimer=MEDICAL_DISCLAIMER,
    )
