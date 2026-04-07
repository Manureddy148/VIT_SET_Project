from contextlib import asynccontextmanager
import json
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.audit import append_audit_record
from src.api.bootstrap import build_registry, seed_vector_store_if_empty, try_vector_store
from src.api.schemas import PatientInput, SeverityResponse
from src.preprocessing.clinical_ner import extract_labs_from_text
from src.preprocessing.domain_classifier import infer_domain_from_labs
from src.preprocessing.multimodal_router import (
    audio_upload_to_clinical_features,
    ecg_upload_to_clinical_features,
    genomics_upload_to_clinical_features,
    image_upload_to_clinical_features,
)
from src.preprocessing.normalizer import normalize_clinical_dict
from src.rag.hallucination_guard import evaluate_faithfulness
from src.rag.retriever import retrieve_literature
from src.synthesis.report_generator import build_clinical_report
from src.synthesis.safety_rails import (
    MEDICAL_DISCLAIMER,
    build_missing_feature_prompts,
    build_recommended_actions,
    build_safety_flags,
)


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


def _pipeline_result(query_text: str, clinical_data: dict[str, float], forced_domain: str | None = None) -> dict[str, Any]:
    registry = app.state.registry
    merged = dict(clinical_data)
    merged.update(extract_labs_from_text(query_text))
    merged = normalize_clinical_dict(merged)

    domain = forced_domain or registry.classify_domain(query_text)
    if domain == "unknown":
        inferred = infer_domain_from_labs(query_text, merged)
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
    explanation = build_clinical_report(result, query_text, citations)
    faithfulness_passed, faithfulness_method = evaluate_faithfulness(explanation, citations)
    missing_feature_prompts = build_missing_feature_prompts(result.missing_features)
    safety_flags = build_safety_flags(result.severity_score, result.severity_label, faithfulness_passed)
    recommended_actions = build_recommended_actions(result.severity_score, result.severity_label)
    audit_log_id = append_audit_record(
        {
            "domain": result.disease_domain,
            "severity_score": result.severity_score,
            "severity_label": result.severity_label,
            "confidence": result.confidence,
            "top_risk_factors": result.top_features,
            "missing_features": result.missing_features,
            "faithfulness_passed": faithfulness_passed,
            "faithfulness_method": faithfulness_method,
            "safety_flags": safety_flags,
        }
    )
    return {
        "result": result,
        "query_text": query_text,
        "citations": citations,
        "explanation": explanation,
        "faithfulness_passed": faithfulness_passed,
        "faithfulness_method": faithfulness_method,
        "missing_feature_prompts": missing_feature_prompts,
        "safety_flags": safety_flags,
        "recommended_actions": recommended_actions,
        "audit_log_id": audit_log_id,
    }


def _build_response(bundle: dict[str, Any]) -> SeverityResponse:
    result = bundle["result"]
    return SeverityResponse(
        disease_domain=result.disease_domain,
        severity_score=result.severity_score,
        severity_label=result.severity_label,
        confidence=result.confidence,
        top_risk_factors=result.top_features,
        shap_values=result.shap_values,
        missing_features=result.missing_features,
        missing_feature_prompts=bundle["missing_feature_prompts"],
        literature_citations=bundle["citations"],
        faithfulness_passed=bundle["faithfulness_passed"],
        safety_flags=bundle["safety_flags"],
        recommended_actions=bundle["recommended_actions"],
        audit_log_id=bundle["audit_log_id"],
        explanation=bundle["explanation"],
        disclaimer=MEDICAL_DISCLAIMER,
    )


def _run_prediction(query_text: str, clinical_data: dict[str, float], forced_domain: str | None = None):
    return _build_response(_pipeline_result(query_text, clinical_data, forced_domain=forced_domain))


def _parse_clinical_json(clinical_json: str) -> dict[str, float]:
    try:
        loaded = json.loads(clinical_json or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail="clinical_json must be valid JSON") from e
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=422, detail="clinical_json must decode to an object")
    return loaded


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    return {
        "pipeline_layers": 7,
        "implemented": {
            "layer1_input": [
                "text",
                "structured_clinical_json",
                "image_upload",
                "ecg_upload",
                "audio_upload",
                "genomics_upload",
            ],
            "layer2_preprocessing": ["lab_extraction", "normalization", "domain_inference", "missing_feature_prompts"],
            "layer3_routing": ["model_registry", "keyword_domain_classifier"],
            "layer4_inference": ["xgboost_tabular_models", "shap_feature_attribution"],
            "layer5_retrieval": ["chromadb_vector_store", "shap_to_rag_query", "faithfulness_guard"],
            "layer6_synthesis": ["template_report", "optional_langchain_groq_llm", "safety_flags"],
            "layer7_delivery": [
                "fastapi_predict_api",
                "fastapi_image_api",
                "fastapi_ecg_api",
                "fastapi_audio_api",
                "fastapi_genomics_api",
                "report_stream_api",
                "streamlit_frontend",
                "jsonl_audit_log",
            ],
        },
        "planned_next": {
            "multimodal_inputs": ["dicom", "nifti"],
            "extra_endpoints": ["/predict/video", "/predict/fhir"],
            "rag_quality": ["pubmed_ingestion_pipeline", "ragas_eval"],
        },
    }


@app.post("/predict", response_model=SeverityResponse)
def predict(patient: PatientInput) -> Any:
    return _run_prediction(patient.query_text, patient.clinical_data)


@app.post("/predict/stream")
def predict_stream(patient: PatientInput):
    bundle = _pipeline_result(patient.query_text, patient.clinical_data)
    response = _build_response(bundle)

    def event_stream():
        chunks = [
            f"event: summary\ndata: {json.dumps({'domain': response.disease_domain, 'severity_score': response.severity_score, 'severity_label': response.severity_label})}\n\n",
            f"event: safety\ndata: {json.dumps({'flags': response.safety_flags, 'faithfulness_passed': response.faithfulness_passed})}\n\n",
            f"event: report\ndata: {json.dumps({'explanation': response.explanation})}\n\n",
            f"event: done\ndata: {json.dumps({'audit_log_id': response.audit_log_id})}\n\n",
        ]
        for chunk in chunks:
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/predict/image", response_model=SeverityResponse)
async def predict_image(
    file: UploadFile = File(...),
    query_text: str = Form(""),
    clinical_json: str = Form("{}"),
) -> Any:
    payload = await file.read()
    clinical = _parse_clinical_json(clinical_json)
    try:
        forced_domain, routed_clinical, routed_query = image_upload_to_clinical_features(
            file.filename or "upload",
            payload,
            query_text,
            clinical,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _run_prediction(routed_query, routed_clinical, forced_domain=forced_domain)


@app.post("/predict/ecg", response_model=SeverityResponse)
async def predict_ecg(
    file: UploadFile = File(...),
    query_text: str = Form(""),
    clinical_json: str = Form("{}"),
) -> Any:
    payload = await file.read()
    clinical = _parse_clinical_json(clinical_json)
    try:
        forced_domain, routed_clinical, routed_query = ecg_upload_to_clinical_features(
            file.filename or "upload",
            payload,
            query_text,
            clinical,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _run_prediction(routed_query, routed_clinical, forced_domain=forced_domain)


@app.post("/predict/audio", response_model=SeverityResponse)
async def predict_audio(
    file: UploadFile = File(...),
    query_text: str = Form(""),
    clinical_json: str = Form("{}"),
) -> Any:
    payload = await file.read()
    clinical = _parse_clinical_json(clinical_json)
    try:
        forced_domain, routed_clinical, routed_query = audio_upload_to_clinical_features(
            file.filename or "upload",
            payload,
            query_text,
            clinical,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _run_prediction(routed_query, routed_clinical, forced_domain=forced_domain)


@app.post("/predict/genomics", response_model=SeverityResponse)
async def predict_genomics(
    file: UploadFile = File(...),
    query_text: str = Form(""),
    clinical_json: str = Form("{}"),
) -> Any:
    payload = await file.read()
    clinical = _parse_clinical_json(clinical_json)
    try:
        forced_domain, routed_clinical, routed_query = genomics_upload_to_clinical_features(
            file.filename or "upload",
            payload,
            query_text,
            clinical,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _run_prediction(routed_query, routed_clinical, forced_domain=forced_domain)
