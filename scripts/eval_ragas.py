"""Phase 6 — RAGAS faithfulness evaluation script.

Evaluates the RAG pipeline across all registered domains by:
  1. Generating synthetic clinical queries
  2. Running the full pipeline (predict → retrieve → synthesise)
  3. Scoring each report with RAGAS faithfulness

Usage:
    python scripts/eval_ragas.py [--n-patients N] [--output audit/ragas_report.json]

Environment variables required for LLM evaluation:
    GROQ_API_KEY=...
    MEDICAL_AI_USE_LLM=1  (optional; enables LLM-generated reports for richer eval)

RAGAS faithfulness target: ≥ 0.90 across 50-report audit (Phase 6 exit criterion).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.bootstrap import build_registry, try_vector_store
from src.models.base_model import PredictionResult
from src.rag.hallucination_guard import evaluate_faithfulness
from src.rag.retriever import retrieve_literature
from src.synthesis.report_generator import build_clinical_report


# ---------------------------------------------------------------------------
# Synthetic patient queries per domain
# ---------------------------------------------------------------------------

DOMAIN_QUERIES: dict[str, list[tuple[str, dict]]] = {
    "diabetes": [
        ("Patient with glucose 280, BMI 38, age 55, frequent urination and thirst", {"glucose": 280.0, "bmi": 38.0, "age": 55.0}),
        ("HbA1c 9.2%, fasting glucose 190 mg/dL, weight 95 kg, hypertension", {"glucose": 190.0, "bmi": 32.0, "age": 48.0}),
        ("Type 2 diabetes management; glucose 145, insulin resistant, age 62", {"glucose": 145.0, "bmi": 30.0, "age": 62.0}),
        ("Gestational diabetes screening: glucose 160, BMI 28, age 32, pregnancies 2", {"glucose": 160.0, "pregnancies": 2.0, "bmi": 28.0, "age": 32.0}),
        ("Diabetic ketoacidosis risk: glucose 350, dehydrated, age 24", {"glucose": 350.0, "age": 24.0, "bmi": 22.0}),
    ],
    "heart_disease": [
        ("Chest pain, ST depression 2.1mm, cholesterol 320, BP 165/98, age 62", {"cholesterol": 320.0, "resting_bp": 165.0, "st_depression": 2.1, "age": 62.0}),
        ("Exertional angina, max HR 95 bpm, major vessels 2, age 70", {"max_hr": 95.0, "major_vessels": 2.0, "age": 70.0, "resting_bp": 150.0}),
        ("Atypical chest pain, cholesterol 280, normal ECG, age 45", {"cholesterol": 280.0, "age": 45.0, "resting_bp": 130.0}),
        ("NSTEMI presentation, troponin elevated, BP 140/90, age 58", {"resting_bp": 140.0, "age": 58.0, "cholesterol": 260.0}),
        ("Heart failure workup, reduced EF, age 75, BP 100/65", {"resting_bp": 100.0, "age": 75.0, "max_hr": 88.0}),
    ],
    "pneumonia": [
        ("SOB, SpO2 89%, temperature 39.8C, RR 28, CRP 180", {"spo2": 89.0, "temperature_c": 39.8, "respiratory_rate": 28.0, "crp": 180.0, "age": 68.0}),
        ("Productive cough, SpO2 93%, bilateral infiltrates, CRP 95, age 72", {"spo2": 93.0, "crp": 95.0, "respiratory_rate": 22.0, "temperature_c": 38.5, "age": 72.0}),
        ("COVID-19 pneumonia, SpO2 94%, high CRP, age 55", {"spo2": 94.0, "crp": 140.0, "temperature_c": 38.2, "age": 55.0}),
        ("Aspiration pneumonia, RR 32, SpO2 91%, age 80", {"spo2": 91.0, "respiratory_rate": 32.0, "temperature_c": 38.7, "age": 80.0}),
        ("Mild pneumonia, SpO2 96%, afebrile, low CRP, age 40", {"spo2": 96.0, "crp": 20.0, "temperature_c": 37.5, "age": 40.0}),
    ],
    "ckd": [
        ("CKD stage 3: creatinine 2.8, eGFR 28, proteinuria, BP 150/90", {"serum_creatinine": 2.8, "blood_urea": 55.0, "hemoglobin": 10.5, "age": 60.0}),
        ("End-stage renal disease: creatinine 8.5, urea 180, hemoglobin 7.5", {"serum_creatinine": 8.5, "blood_urea": 180.0, "hemoglobin": 7.5, "age": 65.0}),
        ("AKI on CKD: creatinine 4.2, oliguria, potassium 5.8", {"serum_creatinine": 4.2, "potassium": 5.8, "blood_urea": 95.0, "age": 55.0}),
    ],
    "sepsis": [
        ("Septic shock: HR 130, RR 26, BP 78/45, lactate 5.2, GCS 12", {"heart_rate": 130.0, "respiratory_rate": 26.0, "systolic_bp": 78.0, "lactate": 5.2, "glasgow_coma_scale": 12.0}),
        ("Sepsis: fever 39.5C, WBC 18, HR 110, lactate 2.8, creatinine 2.1", {"temperature_c": 39.5, "heart_rate": 110.0, "lactate": 2.8, "creatinine": 2.1, "age": 58.0}),
        ("qSOFA positive: confusion, RR 24, SBP 95, suspected bacteremia", {"respiratory_rate": 24.0, "systolic_bp": 95.0, "glasgow_coma_scale": 13.0, "lactate": 2.2}),
    ],
    "liver_disease": [
        ("Cirrhosis decompensation: bilirubin 8.2, albumin 2.0, ALT 320, AST 280", {"total_bilirubin": 8.2, "albumin": 2.0, "alamine_aminotransferase": 320.0, "aspartate_aminotransferase": 280.0}),
        ("Acute hepatitis: ALT 1200, AST 980, bilirubin 4.5, age 35", {"alamine_aminotransferase": 1200.0, "aspartate_aminotransferase": 980.0, "total_bilirubin": 4.5, "age": 35.0}),
        ("NAFLD screening: ALT 85, AST 72, bilirubin 1.1, BMI 38", {"alamine_aminotransferase": 85.0, "aspartate_aminotransferase": 72.0, "total_bilirubin": 1.1, "age": 48.0}),
    ],
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_domain(
    domain: str,
    queries: list[tuple[str, dict]],
    registry: Any,
    vector_store: Any,
) -> list[dict]:
    """Run pipeline on all queries for a domain and return RAGAS results."""
    records = []
    for query_text, clinical_data in queries:
        try:
            # Normalise
            from src.preprocessing.normalizer import normalize_clinical_dict
            merged = normalize_clinical_dict(dict(clinical_data))

            # Predict
            result: PredictionResult = registry.predict_for_domain(domain, merged)

            # Retrieve
            citations = retrieve_literature(vector_store, result, n_results=5)

            # Synthesise
            report = build_clinical_report(result, query_text, citations)

            # Evaluate faithfulness
            passed, method = evaluate_faithfulness(report, citations)

            records.append({
                "domain": domain,
                "query": query_text[:120],
                "severity_score": result.severity_score,
                "severity_label": result.severity_label,
                "faithfulness_passed": passed,
                "faithfulness_method": method,
                "n_citations": len(citations),
                "report_length": len(report),
                "answer": report,
                "contexts": [c.get("text", "") for c in citations if c.get("text")],
            })
        except Exception as exc:
            records.append({
                "domain": domain,
                "query": query_text[:120],
                "error": str(exc),
                "faithfulness_passed": False,
                "faithfulness_method": "error",
            })
    return records


def _compute_ragas_scores(records: list[dict]) -> dict[str, Any]:
    """Compute dataset-level RAGAS scores if dependencies are available."""
    valid = [r for r in records if not r.get("error") and r.get("answer") and r.get("contexts")]
    if not valid:
        return {
            "ragas_available": False,
            "ragas_reason": "no_valid_records",
            "faithfulness": None,
            "answer_relevancy": None,
        }

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore
    except Exception as exc:
        return {
            "ragas_available": False,
            "ragas_reason": f"dependency_missing:{exc}",
            "faithfulness": None,
            "answer_relevancy": None,
        }

    ds = Dataset.from_dict(
        {
            "question": [r["query"] for r in valid],
            "answer": [r["answer"] for r in valid],
            "contexts": [r["contexts"] for r in valid],
        }
    )

    try:
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        # ragas result behaves like a mapping in current versions
        faith = float(result["faithfulness"])
        rel = float(result["answer_relevancy"])
        return {
            "ragas_available": True,
            "ragas_reason": "ok",
            "faithfulness": faith,
            "answer_relevancy": rel,
        }
    except Exception as exc:
        return {
            "ragas_available": False,
            "ragas_reason": f"evaluation_failed:{exc}",
            "faithfulness": None,
            "answer_relevancy": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS-style faithfulness evaluation")
    parser.add_argument("--output", default="audit/ragas_report.json", help="Output JSON path")
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_QUERIES.keys()))
    args = parser.parse_args()

    print("[eval] Building model registry...", flush=True)
    registry = build_registry()
    registered_domains = set(registry.list_domains())

    print("[eval] Connecting to vector store...", flush=True)
    vector_store = try_vector_store()

    all_records = []
    for domain in args.domains:
        if domain not in registered_domains:
            print(f"[eval] Domain '{domain}' not in registry — skipping", flush=True)
            continue
        queries = DOMAIN_QUERIES.get(domain, [])
        if not queries:
            print(f"[eval] No queries for domain '{domain}' — skipping", flush=True)
            continue
        print(f"[eval] Evaluating {len(queries)} queries for domain='{domain}'...", flush=True)
        records = evaluate_domain(domain, queries, registry, vector_store)
        all_records.extend(records)
        passed = sum(1 for r in records if r.get("faithfulness_passed"))
        print(f"  → faithfulness: {passed}/{len(records)} passed ({100*passed/len(records):.0f}%)")

    # Summary (fallback snippet-guard pass rate)
    total = len(all_records)
    passed_total = sum(1 for r in all_records if r.get("faithfulness_passed"))
    faithfulness_rate = (passed_total / total * 100) if total else 0.0
    ragas_scores = _compute_ragas_scores(all_records)

    ragas_faith = ragas_scores.get("faithfulness")
    ragas_rel = ragas_scores.get("answer_relevancy")
    ragas_targets_met = (
        ragas_scores.get("ragas_available") is True
        and ragas_faith is not None
        and ragas_rel is not None
        and ragas_faith >= 0.90
        and ragas_rel >= 0.80
    )

    summary = {
        "total_queries": total,
        "target_queries": 50,
        "faithfulness_passed": passed_total,
        "faithfulness_rate_pct": round(faithfulness_rate, 1),
        "target_pct": 90.0,
        "ragas": ragas_scores,
        "targets": {
            "snippet_guard_faithfulness_pct": 90.0,
            "ragas_faithfulness": 0.90,
            "ragas_answer_relevancy": 0.80,
        },
        "meets_target": ragas_targets_met if ragas_scores.get("ragas_available") else (faithfulness_rate >= 90.0),
        "records": all_records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"RAGAS Faithfulness Audit")
    print(f"  Queries evaluated : {total}")
    print(f"  Passed            : {passed_total}")
    print(f"  Faithfulness rate : {faithfulness_rate:.1f}% (target: 90%)")
    if ragas_scores.get("ragas_available"):
        print(f"  RAGAS faithfulness: {ragas_faith:.3f} (target: >=0.90)")
        print(f"  RAGAS relevancy   : {ragas_rel:.3f} (target: >=0.80)")
    else:
        print(f"  RAGAS status      : unavailable ({ragas_scores.get('ragas_reason')})")
    print(f"  Meets target      : {'YES ✓' if summary['meets_target'] else 'NO ✗'}")
    print(f"  Report written to : {out_path}")
    print(f"{'='*60}")
    return 0 if summary["meets_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
