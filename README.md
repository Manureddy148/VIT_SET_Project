# Medical AI — Severity Prediction System

**A 7-layer clinical intelligence pipeline combining ML prediction, explainability, and retrieval-augmented generation.**

## ✨ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-rag.txt

# 2. Fetch real PubMed abstracts (optional but recommended)
python scripts/ingest_pubmed.py --domains diabetes heart_disease pneumonia --limit 30

# 3. Start API
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Start Streamlit UI (another terminal)
streamlit run frontend/app.py

# 5. Run tests
pytest -q
```

## What It Does

### Input
- **Natural language** patient description
- **Structured clinical data** (glucose, BMI, BP, etc.)
- **Multimodal uploads** (DICOM/PNG images, ECG waveforms, audio, genomics files)

### Output
Severity prediction (0–100) with:
- ✅ **Top risk drivers** identified via SHAP explainability
- ✅ **Peer-reviewed citations** retrieved from PubMed
- ✅ **Safety flags** for urgent cases
- ✅ **Recommended actions** (escalate/review/monitor)
- ✅ **Audit trail** of faithfulness evaluation method used

## 7-Layer Architecture

| Layer | Component | Status |
|-------|-----------|--------|
| **1** | User Input (5 modalities) | ✅ Complete |
| **2** | Preprocessing (NER, lab extraction, negation) | ✅ Complete |
| **3** | Dynamic Routing (domain classifier, model registry) | ✅ Complete (3 diseases) |
| **4** | ML Inference (XGBoost + SHAP) | ✅ Complete |
| **5** | RAG Retrieval (**PubMed abstracts** + ChromaDB) | ✅ **NEW!** |
| **6** | Synthesis (reports + safety rails) | ✅ Complete |
| **7** | Delivery (FastAPI + Streamlit + audit logging) | ✅ Complete |

## Disease Domains

Implemented:
- ✅ **Diabetes** (Type 2 severity prediction)
- ✅ **Heart Disease** (Cardiovascular risk)
- ✅ **Pneumonia** (Respiratory severity)

Future (Phase 2):
- [ ] CKD (Chronic Kidney Disease)
- [ ] Sepsis

## API Endpoints

```bash
# Text prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query_text": "glucose 210, BMI 34, age 52", "clinical_data": {}}'

# Image (DICOM/PNG)
curl -X POST http://localhost:8000/predict/image \
  -F "file=@chest_xray.png" \
  -F "query_text=chest imaging"

# ECG waveform
curl -X POST http://localhost:8000/predict/ecg \
  -F "file=@ecg_data.csv"

# Audio (WAV/MP3)
curl -X POST http://localhost:8000/predict/audio \
  -F "file=@lung_sounds.wav"

# Genomics (VCF/FASTQ)
curl -X POST http://localhost:8000/predict/genomics \
  -F "file=@variant_data.vcf"

# Streaming report
curl -X POST http://localhost:8000/predict/stream \
  -H "Content-Type: application/json" \
  -d '{"query_text": "...", "clinical_data": {}}' \
  --stream

# Health check
curl http://localhost:8000/health

# Pipeline status
curl http://localhost:8000/pipeline/status
```

## Core Features

### 🧠 Explainability
- SHAP values identify which clinical variables drove the prediction
- Top 5 risk factors displayed in report and Streamlit gauge

### 📚 Retrieval-Augmented Generation
- **NEW**: Real PubMed abstracts fetched on demand
- Semantic search using SHAP top features as queries
- 73+ abstracts per ingestion run (diabetes, heart, pneumonia)
- Citations are grounded in indexed literature

### 🛡️ Safety & Compliance
- Hard-coded clinical disclaimers injected into all reports
- Faithfulness checking (citations must appear in explanation)
- Safety flags for urgent/critical cases
- Audit logging of all predictions (JSONL format)

### 🔌 Optional Advanced Features
- **Biomedical NER**: transformers model (d4data/biomedical-ner-all) with negation detection
- **ECG Processing**: NeuroKit2 for R-peak detection and HRV calculation
- **LLM Synthesis**: LangChain + Groq Llama models (replace template reports)
- **Faithfulness Metrics**: RAGAS for scientific citation validation

All optional features degrade gracefully if unavailable.

## Project Structure

```
medical-ai-severity/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app + 8 endpoints
│   │   ├── bootstrap.py            # Initialize registry + RAG
│   │   ├── schemas.py              # Pydantic models
│   │   └── audit.py                # JSONL audit logging
│   ├── models/
│   │   ├── registry.py             # Model router + selector
│   │   ├── diabetes_model.py       # XGBoost (Pima dataset)
│   │   ├── heart_model.py          # XGBoost (UCI dataset)
│   │   └── pneumonia_model.py      # XGBoost (synthetic)
│   ├── preprocessing/
│   │   ├── clinical_ner.py         # Biomedical NER (optional)
│   │   ├── domain_classifier.py    # Disease domain inference
│   │   ├── multimodal_router.py    # Image/ECG/audio/genomics handlers
│   │   ├── ecg_processor.py        # NeuroKit2 signal processing
│   │   └── normalizer.py           # Feature normalization
│   ├── inference/
│   │   ├── ensemble.py             # Multi-model voting
│   │   └── shap_explainer.py       # Feature attribution
│   ├── rag/
│   │   ├── vector_store.py         # ChromaDB setup
│   │   ├── retriever.py            # Semantic search
│   │   └── hallucination_guard.py  # Faithfulness checking
│   └── synthesis/
│       ├── report_generator.py     # Report templates + LLM synthesis
│       ├── llm_reporter.py         # LangChain + Groq path
│       └── safety_rails.py         # Safety flags + recommendations
├── frontend/
│   └── app.py                      # Streamlit 5-tab UI
├── scripts/
│   ├── ingest_pubmed.py            # **NEW**: Fetch PubMed abstracts
│   ├── setup_pubmed.sh/.bat        # **NEW**: Setup scripts
│   └── train_models.py             # Train XGBoost models
├── tests/
│   ├── test_api.py                 # 15 unit tests
│   ├── test_models.py
│   └── test_rag.py
├── data/
│   ├── raw/                        # Kaggle datasets (gitignored)
│   └── processed/                  # **pubmed_docs.json** ← NEW
├── requirements.txt                # Core dependencies
├── requirements-rag.txt            # Optional advanced packages
├── PUBMED_INTEGRATION.md           # **NEW**: PubMed setup guide
└── medical_ai_e2e.py              # Smoke test (8 endpoints)
```

## PubMed Integration (NEW!)

The system now fetches real medical abstracts from NCBI to ground predictions in peer-reviewed literature.

```bash
# Run PubMed ingestion (Windows)
.\scripts\setup_pubmed.bat

# Or Linux/Mac
bash scripts/setup_pubmed.sh

# Generates: data/processed/pubmed_docs.json (73 abstracts)
```

See [PUBMED_INTEGRATION.md](PUBMED_INTEGRATION.md) for details.

## Testing

```bash
# Unit tests (15 tests, all passing)
pytest -q

# Specific endpoint
pytest tests/test_api.py::test_predict_diabetes -v

# End-to-end smoke test (8 endpoints)
python medical_ai_e2e.py
```

## Environment Variables

```bash
# Optional features
MEDICAL_AI_USE_LLM=1                    # Enable LLM synthesis (requires GROQ_API_KEY)
MEDICAL_AI_USE_BIOMED_NER=1             # Enable biomedical NER (slower startup)
MEDICAL_AI_USE_RAGAS=1                  # Enable RAGAS faithfulness metric
MEDICAL_AI_SKIP_RAG=0                   # Disable RAG entirely

# API credentials
GROQ_API_KEY=gsk_...                    # Groq LLM API key

# Storage
CHROMA_PERSIST_DIR=./chroma_db          # Vector store location
```

## Dependencies

### Core
- XGBoost, scikit-learn, SHAP, NumPy, Pandas
- FastAPI, Uvicorn, Pydantic
- ChromaDB, sentence-transformers
- Streamlit, Plotly
- pytest

### Optional (RAG Advanced)
- transformers (biomedical NER)
- neurokit2 (ECG processing)
- LangChain, Groq SDK (LLM synthesis)
- RAGAS (faithfulness metric)
- FAISS (alternative vector search)
- pydicom (DICOM imaging)

## Docker

```bash
docker build -t medical-ai .
docker run -p 8000:8000 medical-ai
```

## Authors & License

Part of Medical AI research initiative at VIT.

---

**Latest Update**: April 2026 — PubMed integration enabled with 73 real abstracts per ingestion run.

## Full-scope research milestone (2026-10-05)

See [scope status](docs/SCOPE_STATUS.md) for every implemented/run, built/unrun and outstanding item. This milestone is NOT the completed full roadmap. See [dataset terms](docs/DATASETS.md) and [paper outline](docs/PAPER_OUTLINE.md).

Default API now loads only real research tabular artifacts. It does not silently train synthetic models. Legacy demonstrations require `MEDICAL_AI_ALLOW_DEMO=1` and are not model evaluation. Missing real image checkpoint returns 503. ECG/audio/genomic proxy diagnoses are disabled by default.

### Reproduce actual tabular results

```bash
python -m pip install -r requirements.txt
python scripts/train_real_tabular.py --domain heart_disease
python scripts/train_real_tabular.py --domain diabetes
MEDICAL_AI_ALLOW_DEMO=0 uvicorn src.api.main:app
# /models returns exact trained schema: CDC is NOT Pima.
```

Reports use fixed stratified train/calibration/test, seed 42. Heart n=303, test n=61, AUC=0.948052. CDC n=253680, test n=50736, AUC=0.825507 (target >0.85 NOT met). Platt scaling worsened ECE on both runs. These are single held-out research results, not external or clinical validation.

### Real literature and image pipeline

```bash
python -m pip install -r requirements-rag.txt
python scripts/ingest_pubmed_verified.py --limit 500
MEDICAL_AI_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 python scripts/index_pubmed.py
python -m pip install -r requirements-imaging.txt
# Obtain permitted Kermany images and documented patient IDs; make manifest:
# path,label,patient_id,split ; label=0/1 ; split=train/calibration/test
python scripts/train_pneumonia_image.py --manifest data/raw/pneumonia_manifest.csv --epochs 10
python scripts/evaluate_retrieval.py --judgments data/processed/relevance_judgments.json
# Judgments list requires relevant_ids, shap_ranked_ids, keyword_ranked_ids per query.
```

Executed locally: 500 real abstracts each for three primary diseases; persistent Chroma count 1500; MiniLM/MMR queries return five sources per domain. This is not a precision/recall/RAGAS benchmark. Corpus and checkpoints are excluded from Git for size, terms and provenance. Image pipeline forward/leakage tests ran, real image training did not.

Optional dependencies: `requirements-extensions.txt` for actual audio/volume loaders. Synthetic proxies remain only for backwards-compatible explicit demos; never enable demo mode for scientific performance or patient use. Later trained modality models, image SHAP, RAGAS, deployment, clinician validation and novelty ablations remain outstanding as listed in the scope table.
