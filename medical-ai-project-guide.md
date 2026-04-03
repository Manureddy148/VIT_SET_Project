# Medical AI — Project Start Guide
### Architecture: 7-Layer Clinical Intelligence System

---

## IMPORTANT — This repo’s current direction

This project is now aligned to **open-source inference only**:
- **No custom training** of models in this repo.
- Use **Hugging Face / open-source checkpoints** and run **2–3 models per disease/domain** (ensemble + disagreement handling).

See:
- `docs/ROADMAP.md`
- `docs/MODEL_CATALOG.md`

---

## ⚡ START HERE FIRST — The Right Order

Before writing a single line of model code, get your foundation right. This is the order that will save you months of pain.

```
Week 1–2  →  GitHub repo + project structure
Week 3–4  →  Kaggle data collection + EDA
Week 5–8  →  ML models (XGBoost + SHAP) per disease
Week 9–12 →  RAG pipeline (ChromaDB + LangChain)
Week 13–16 → FastAPI backend + Streamlit UI
Week 17–20 → Integration, testing, RAGAS evaluation
Week 21–52 → Iteration, research paper, deployment
```

---

## 1. GITHUB — Set Up This Structure First

### Create this exact repo structure on Day 1

```
medical-ai-severity/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # Auto-run tests on push
│
├── data/
│   ├── raw/                        # Kaggle downloads go here (gitignored)
│   ├── processed/                  # Cleaned CSVs
│   └── embeddings/                 # Vector store files (gitignored)
│
├── notebooks/
│   ├── 01_eda_diabetes.ipynb
│   ├── 02_eda_heart.ipynb
│   ├── 03_eda_pneumonia.ipynb
│   └── 04_model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clinical_ner.py         # Layer 2: Entity extraction
│   │   ├── domain_classifier.py    # Layer 2: Disease domain detection
│   │   └── normalizer.py           # Layer 2: Feature normalization
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py             # Layer 3: Model registry + router
│   │   ├── base_model.py           # Abstract base class
│   │   ├── diabetes_model.py
│   │   ├── heart_model.py
│   │   └── pneumonia_model.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── ensemble.py             # Layer 4: Multi-model voting
│   │   └── shap_explainer.py       # Layer 4: SHAP feature importance
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py         # Layer 5: ChromaDB setup
│   │   ├── retriever.py            # Layer 5: Semantic search
│   │   └── hallucination_guard.py  # Layer 5: RAGAS faithfulness
│   │
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── report_generator.py     # Layer 6: LLM report writing
│   │   └── safety_rails.py         # Layer 6: Guardrails + disclaimers
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # Layer 7: FastAPI app
│       └── schemas.py              # Pydantic input/output models
│
├── frontend/
│   └── app.py                      # Streamlit UI
│
├── tests/
│   ├── test_models.py
│   ├── test_rag.py
│   └── test_api.py
│
├── scripts/
│   ├── ingest_pubmed.py            # Download + embed PubMed abstracts
│   └── train_models.py             # Train all ML models
│
├── models/                         # Saved .joblib model files (gitignored)
│   └── .gitkeep
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml
```

### .gitignore (critical — put this in on Day 1)
```
data/raw/
data/embeddings/
models/*.joblib
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
chroma_db/
```

### Commit strategy
Use conventional commits so your history is readable:
- `feat: add XGBoost diabetes model`
- `fix: normalize missing lab values before inference`
- `data: add heart disease EDA notebook`
- `docs: update README with setup instructions`

---

## 2. KAGGLE — Your Free Data + GPU Compute

### Datasets to download first (all free on Kaggle)

| Disease | Dataset | Link |
|---------|---------|------|
| Diabetes | Pima Indians Diabetes | `datasets/uciml/pima-indians-diabetes-database` |
| Heart Disease | Cleveland Heart Disease (UCI) | `datasets/ronitf/heart-disease-uci` |
| Pneumonia | Chest X-Ray Pneumonia | `datasets/paultimothymooney/chest-xray-pneumonia` |
| Multi-disease | CDC Chronic Disease Dataset | `datasets/cdc/chronic-disease-indicators` |

### How to use Kaggle efficiently

**For EDA and model training — use Kaggle Notebooks (free GPU)**
- Go to kaggle.com → Code → New Notebook
- Enable GPU: Settings → Accelerator → GPU T4 x2
- You get **30 free GPU hours/week** — use for XGBoost training
- Download trained model → save to GitHub as `.joblib`

**Kaggle API — pull data locally**
```bash
pip install kaggle
# Put your kaggle.json API token in ~/.kaggle/
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data/raw/
kaggle datasets download -d ronitf/heart-disease-uci -p data/raw/
```

**Kaggle Competitions** — search for medical AI competitions. They often have better-labelled clinical datasets than public repos.

---

## 3. GCP — DO NOT rely on the free trial for a 1-year project

### The honest breakdown

| | GCP Free Trial | GCP After Trial | Better Alternative |
|--|--|--|--|
| Duration | 2 months | Pay-as-you-go | N/A |
| Cost | $300 credit | ~$50–200/month for your workload | Free |
| Best use | Test deployment | Production | See below |

### Recommended free-tier stack for 1-year development

**For the first 6–9 months (building + experimenting):**

| Need | Free Tool | Cost |
|------|-----------|------|
| Model training | Kaggle Notebooks | Free (30 GPU hrs/wk) |
| Model training (heavy) | Google Colab Free | Free |
| Vector DB (ChromaDB) | Run locally | Free |
| LLM | Ollama + Llama 3 local | Free |
| LLM (API) | Groq API (Llama 3 hosted) | Free tier |
| Backend hosting | Render.com (free tier) | Free |
| Frontend hosting | Streamlit Community Cloud | Free |
| Storage | GitHub LFS + HuggingFace Hub | Free |

**Use GCP free trial strategically:**
- Save it for Month 9–10 when you need to demo a deployed production version
- Use Cloud Run (serverless) — only pay when requests come in
- Use Firestore for patient session state
- Use GCP free tier (always-free): Cloud Storage 5GB, Cloud Run 2M requests/month

### If you need GCP long-term
```
Cloud Run (containerized FastAPI)   ~$5–15/month
Cloud Storage (models + embeddings) ~$2–5/month
Vertex AI (optional)                ~$20–50/month
Total estimate                      ~$30–70/month
```

**Verdict:** Build locally + Kaggle for 9 months. Use GCP free trial for your final demo deployment. This is the smart approach.

---

## 4. TECH STACK — Exact Packages

### requirements.txt (start with this)
```
# Core ML
xgboost==2.0.3
scikit-learn==1.4.0
shap==0.44.1
imbalanced-learn==0.12.0
joblib==1.3.2

# NLP + LLM
langchain==0.2.0
langchain-community==0.2.0
langchain-groq==0.1.3           # Free Llama 3 via Groq
transformers==4.40.0
sentence-transformers==3.0.1    # all-MiniLM-L6-v2 embeddings

# RAG + Vector DB
chromadb==0.5.0
faiss-cpu==1.8.0
ragas==0.1.10

# API + Frontend
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.1
streamlit==1.35.0
python-dotenv==1.0.1

# Data
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.4
seaborn==0.13.2

# Medical NLP (optional, for clinical NER)
spacy==3.7.4
# python -m spacy download en_core_sci_sm  (scispaCy medical model)
```

### .env.example
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=optional_if_using_gpt4
CHROMA_PERSIST_DIR=./chroma_db
MODEL_REGISTRY_DIR=./models
PUBMED_EMAIL=your@email.com
```

---

## 5. WEEK-BY-WEEK ROADMAP

### Phase 1: Foundation (Weeks 1–4)
- [ ] Create GitHub repo with full folder structure
- [ ] Set up virtual environment + install requirements
- [ ] Download all 3 Kaggle datasets
- [ ] Run EDA notebooks for each disease
- [ ] Understand class distributions and missing data patterns
- [ ] Document key clinical features per disease

### Phase 2: ML Core (Weeks 5–10)
- [ ] Train XGBoost model per disease (Kaggle GPU)
- [ ] Implement SHAP explainer for each model
- [ ] Build model registry + router (domain classifier → model selector)
- [ ] Build missing-data dialogue agent
- [ ] Validate: AUC > 0.85 on each disease
- [ ] Save models as .joblib, push to HuggingFace Hub

### Phase 3: RAG Pipeline (Weeks 11–16)
- [ ] Run `ingest_pubmed.py` to embed 500+ abstracts per disease
- [ ] Set up ChromaDB vector store
- [ ] Build semantic retriever with MMR (Maximal Marginal Relevance)
- [ ] Connect SHAP top-features → retrieval query terms
- [ ] Integrate Groq/Llama 3 for synthesis
- [ ] Add RAGAS faithfulness evaluation loop

### Phase 4: API + UI (Weeks 17–20)
- [ ] Build FastAPI backend with `/predict` endpoint
- [ ] Add Pydantic schemas for input validation
- [ ] Build Streamlit UI with severity gauge
- [ ] Show SHAP bar chart inline in UI
- [ ] Display literature citations

### Phase 5: Integration + Research (Weeks 21–40)
- [ ] End-to-end testing with synthetic patients
- [ ] RAGAS hallucination audit report
- [ ] Write research paper sections (novelty contributions)
- [ ] Deploy to Streamlit Cloud + Render (free)
- [ ] Use GCP free trial for demo deployment

---

## 6. THE SINGLE FILE TO CODE FIRST

**Start with `src/models/base_model.py`** — this is the architectural contract everything else depends on.

Then: `registry.py` → `diabetes_model.py` → `shap_explainer.py` → `vector_store.py` → `main.py`

See the accompanying `starter_code.py` file for the first 5 modules pre-written.

---

## 7. ACADEMIC NOVELTY — Track These As You Build

Your architecture has 6 genuine research contributions. Document evidence as you build:

1. **Dynamic domain routing** — measure routing accuracy (target >95%)
2. **Severity magnitude scoring** — compare vs binary classification on clinical utility
3. **SHAP-guided RAG** — ablation study: SHAP queries vs keyword queries (retrieval precision)
4. **RAGAS faithfulness loop** — report hallucination rate (target: 0%)
5. **Missing-data dialogue agent** — measure how many incomplete queries it salvages
6. **Modular model registry** — benchmark hot-swap latency (<50ms)

---

*Generated from your medical-ai-architecture.html design document*
