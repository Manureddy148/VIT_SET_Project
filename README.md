# Medical AI Severity System — README v1

**Research & course project scaffold** — not a medical device, not for diagnosis or treatment.

This repository implements a **7-layer clinical intelligence pipeline** (see `medical-ai-project-guide.md` and `Create Roadmap/Medical_AI_Team_Guide.html`): multimodal-style input → light clinical parsing → **domain routing** → **XGBoost / rule models** → **SHAP-style attributions** → optional **RAG** (ChromaDB) → **templated report** + **safety disclaimer** → **FastAPI + Streamlit**.

---

## Who should read this

- **Teammates** onboarding onto the VIT SET project  
- **Reviewers** who need a single map of folders, APIs, and how to run everything  
- **Researchers** extending models or the RAG layer  

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/api/` | **Layer 7** — FastAPI app, `/health`, `/predict` |
| `src/models/` | **Layers 3–4** — registry, `DiabetesModel`, `HeartModel`, `PneumoniaModel` |
| `src/preprocessing/` | **Layer 2** — regex lab extraction, normalization, domain hints |
| `src/inference/` | **Layer 4** — SHAP helper, ensemble placeholder |
| `src/rag/` | **Layer 5** — Chroma vector store, retriever, guard stub |
| `src/synthesis/` | **Layer 6** — report text + disclaimer |
| `frontend/` | Streamlit UI calling the API |
| `tests/` | Pytest — API, models, **e2e smoke** (`test_e2e_smoke.py`) |
| `scripts/` | `train_models.py`, `ingest_pubmed.py` stub, `run_api.*`, `check_env.bat` |
| `launch_api.py` | **Recommended** one-command API start + browser |

---

## Quick start (Windows)

### 1. Dependencies

```powershell
cd path\to\VIT_SET_Project
python -m pip install -r requirements.txt
```

Optional vector RAG (Layer 5):

```powershell
python -m pip install -r requirements-rag.txt
```

### 2. Run automated tests (sanity / CI)

```powershell
pytest
```

Includes **17 tests**: unit API, models, and **end-to-end** checks (`tests/test_e2e_smoke.py`) — OpenAPI, all three disease routes, 422 on unknown domain, response shape validation.

### 3. Start the API

**Keep the terminal open** while using the browser.

```powershell
python launch_api.py
```

Then open **http://127.0.0.1:8000/docs** (Swagger).

Alternatives: `scripts\run_api.bat`, `.\scripts\run_api.ps1`.

### 4. Start the Streamlit UI

**Terminal 1:** `python launch_api.py`  
**Terminal 2:**

```powershell
streamlit run frontend/app.py
```

The UI checks `/health`, offers **preset cases** (diabetes / heart / pneumonia), shows **severity**, **SHAP bar chart**, **citations** (when RAG is enabled), and the **full explanation + disclaimer**.

---

## API contract (short)

- **`GET /health`** → `{ "status": "ok" }`  
- **`POST /predict`** — body:

```json
{
  "query_text": "free text with symptoms / keywords",
  "clinical_data": { "glucose": 210, "bmi": 34 },
  "patient_id": "optional"
}
```

Response includes `disease_domain`, `severity_score` (0–100), `severity_label`, `confidence`, `top_risk_factors`, `shap_values`, `literature_citations`, `explanation`, `disclaimer`.

**Routing:** keyword + structured hints map to `diabetes`, `heart_disease`, or `pneumonia`. If nothing matches → **422**.

---

## Environment variables

| Variable | Meaning |
|----------|---------|
| `MEDICAL_AI_SKIP_RAG` | `1` — skip Chroma (default in tests / light installs) |
| `MEDICAL_API_URL` | Streamlit → API base (default `http://127.0.0.1:8000`) |
| `MODEL_REGISTRY_DIR` | Where `.joblib` bundles are stored |
| `DIABETES_MODEL_PATH` / `HEART_MODEL_PATH` | Override model files |

See `.env.example`.

---

## Models & training

- If **no** `models/diabetes_v1.joblib` / `heart_v1.joblib`, the API **trains small synthetic XGBoost models on startup** so the demo runs without Kaggle.  
- For **real** weights: train on Kaggle / local data, then:

```powershell
python scripts/train_models.py
```

---

## Word document for teammates (README v1 .docx)

A **pre-built** handout lives at **`docs/Medical_AI_Project_README_v1.docx`** (regenerate after big doc changes).

```powershell
python -m pip install -r requirements-dev.txt
python scripts/build_teammate_docx.py
```

---

## Ethics & scope

- Outputs are **educational / research prototypes**.  
- **Always** show and respect the API/UI **disclaimer**.  
- Do **not** present scores as diagnoses or replace clinician judgment.

---

## Further reading

- `medical-ai-project-guide.md` — semester roadmap, Kaggle datasets, stack  
- `QUICKSTART.txt` — copy-paste commands  
- `starter_code.py` — original monolith reference (architecture smoke test)  

**Questions:** trace any request from `src/api/main.py` → `bootstrap.build_registry()` → `registry.predict_for_domain()`.
