# 🏥 Medical AI Severity System — Complete Cursor Resource Guide
> **Purpose:** Every resource, model, dataset, code pattern, and implementation reference needed to build the full end-to-end pipeline in Cursor. Paste sections directly into Cursor chat or use as @docs context.

---

## 📁 REPO STRUCTURE (Copy on Day 1)

```
medical-ai-severity/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                          # gitignored
│   ├── processed/
│   ├── embeddings/                   # gitignored
│   └── imaging/
│       ├── xray/
│       ├── mri/
│       └── ecg/
├── notebooks/
│   ├── 01_eda_diabetes.ipynb
│   ├── 02_eda_heart.ipynb
│   ├── 03_eda_pneumonia.ipynb
│   ├── 04_eda_imaging.ipynb
│   ├── 05_eda_ecg.ipynb
│   └── 06_model_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── dicom_reader.py           # DICOM/NIfTI loader
│   │   ├── ecg_reader.py             # EDF/WFDB/XML loader
│   │   ├── lab_parser.py             # FHIR/HL7/CSV parser
│   │   └── multimodal_router.py      # Routes input type to loader
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clinical_ner.py           # Layer 2: BioBERT NER
│   │   ├── domain_classifier.py      # Layer 2: Disease routing
│   │   ├── normalizer.py             # Feature normalization
│   │   ├── ecg_processor.py          # Signal denoising + R-peak
│   │   └── image_preprocessor.py    # Resize/normalize images
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py             # Abstract base
│   │   ├── registry.py               # Model registry + router
│   │   ├── structured/
│   │   │   ├── diabetes_model.py
│   │   │   ├── heart_model.py
│   │   │   ├── pneumonia_model.py
│   │   │   └── ckd_model.py
│   │   └── imaging/
│   │       ├── imaging_extension.py  # ImageRouter (already built)
│   │       ├── ecg_model.py
│   │       ├── echo_model.py
│   │       └── us_model.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── ensemble.py               # Layer 4: Multi-model voting
│   │   ├── shap_explainer.py         # SHAP → RAG bridge
│   │   └── calibration.py            # Platt scaling
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py           # ChromaDB
│   │   ├── retriever.py              # MMR semantic search
│   │   ├── pubmed_ingestor.py        # PubMed → embeddings
│   │   └── hallucination_guard.py    # RAGAS faithfulness
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── report_generator.py       # LLM report (Groq/Llama3)
│   │   └── safety_rails.py           # Disclaimers + guardrails
│   └── api/
│       ├── __init__.py
│       ├── main.py                   # FastAPI app
│       └── schemas.py                # Pydantic models
├── frontend/
│   └── app.py                        # Streamlit UI
├── tests/
│   ├── test_models.py
│   ├── test_rag.py
│   ├── test_imaging.py
│   └── test_api.py
├── scripts/
│   ├── ingest_pubmed.py
│   ├── train_models.py
│   └── download_datasets.sh
├── models/                           # .joblib files — gitignored
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-imaging.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 📦 REQUIREMENTS FILES

### `requirements.txt` — Core Pipeline
```txt
# ── ML CORE ──────────────────────────────────────────────────
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.45.0
imbalanced-learn==0.12.3
joblib==1.4.2
lightgbm==4.3.0
catboost==1.2.5

# ── NLP + LLM ─────────────────────────────────────────────────
langchain==0.2.16
langchain-community==0.2.16
langchain-groq==0.1.9
langchain-openai==0.1.23
transformers==4.44.0
sentence-transformers==3.1.1
tokenizers==0.19.1
accelerate==0.33.0

# ── RAG + VECTOR DB ───────────────────────────────────────────
chromadb==0.5.5
faiss-cpu==1.8.0.post1
ragas==0.1.21
rank-bm25==0.2.2

# ── API + BACKEND ─────────────────────────────────────────────
fastapi==0.112.2
uvicorn[standard]==0.30.6
pydantic==2.8.2
python-multipart==0.0.9
httpx==0.27.2
python-dotenv==1.0.1

# ── FRONTEND ──────────────────────────────────────────────────
streamlit==1.38.0
plotly==5.23.0
streamlit-aggrid==1.0.5

# ── DATA ──────────────────────────────────────────────────────
pandas==2.2.2
numpy==1.26.4
scipy==1.14.0
matplotlib==3.9.2
seaborn==0.13.2
pyarrow==17.0.0

# ── MEDICAL NLP ───────────────────────────────────────────────
spacy==3.7.6
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
biopython==1.84
requests==2.32.3

# ── EVAL ──────────────────────────────────────────────────────
pytest==8.3.2
pytest-asyncio==0.23.8
```

### `requirements-imaging.txt` — Imaging & Signal Processing
```txt
# ── IMAGING CORE ──────────────────────────────────────────────
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
Pillow==10.4.0
timm==1.0.9
einops==0.8.0

# ── MEDICAL FORMATS ───────────────────────────────────────────
pydicom==2.4.4
pylibjpeg==2.0.0
pylibjpeg-libjpeg==2.1.0
nibabel==5.2.1
SimpleITK==2.4.0
highdicom==0.22.0

# ── SIGNAL PROCESSING ─────────────────────────────────────────
wfdb==4.1.2                           # PhysioNet WFDB format
mne==1.8.0                            # EEG/MEG processing
neurokit2==0.2.10                     # ECG/PPG/EDA analysis
biosppy==2.2.0                        # Biosignal processing
pyedflib==0.1.38                      # EDF file reading
scipy==1.14.0

# ── CLIP / ZERO-SHOT ──────────────────────────────────────────
open-clip-torch==2.26.1

# ── WSI / PATHOLOGY ───────────────────────────────────────────
openslide-python==4.3.1
tiatoolbox==1.5.1

# ── ULTRASOUND / ECHO ─────────────────────────────────────────
pyechoai==0.1.0                       # EchoNet wrapper
```

---

## 🤖 HUGGINGFACE MODELS — Complete Registry

### Structured Data Models (XGBoost wrappers → HF Hub)
```python
STRUCTURED_MODELS = {
    # ── DIABETES ──────────────────────────────────────────────
    "diabetes": {
        "hf_hub": "scikit-learn/diabetes-prediction",
        "local": "models/diabetes_xgb_v2.joblib",
        "dataset": "kaggle: uciml/pima-indians-diabetes-database",
        "features": ["pregnancies","glucose","blood_pressure","skin_thickness",
                     "insulin","bmi","diabetes_pedigree","age"],
        "auc_target": 0.87,
        "paper": "Pima Indians Diabetes Study (Smith et al., 1988)",
    },
    # ── HEART DISEASE ─────────────────────────────────────────
    "heart_disease": {
        "hf_hub": "assemblyai/heart-disease-prediction",
        "local": "models/heart_xgb_v2.joblib",
        "dataset": "kaggle: ronitf/heart-disease-uci",
        "features": ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"],
        "auc_target": 0.88,
        "paper": "Cleveland Heart Disease (Detrano et al., 1989)",
    },
    # ── CHRONIC KIDNEY DISEASE ────────────────────────────────
    "ckd": {
        "hf_hub": "None — train from UCI CKD dataset",
        "local": "models/ckd_xgb_v1.joblib",
        "dataset": "kaggle: mansoordaku/ckdisease",
        "features": ["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr",
                     "bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm",
                     "cad","appet","pe","ane"],
        "auc_target": 0.99,
    },
    # ── LIVER DISEASE ─────────────────────────────────────────
    "liver_disease": {
        "hf_hub": "None — train from ILPD dataset",
        "local": "models/liver_xgb_v1.joblib",
        "dataset": "kaggle: uciml/indian-liver-patient-records",
        "features": ["age","gender","total_bilirubin","direct_bilirubin",
                     "alkaline_phosphatase","alamine_aminotransferase",
                     "aspartate_aminotransferase","total_proteins","albumin",
                     "albumin_globulin_ratio"],
        "auc_target": 0.82,
    },
    # ── SEPSIS ────────────────────────────────────────────────
    "sepsis": {
        "hf_hub": "None — train from PhysioNet 2019",
        "local": "models/sepsis_lgbm_v1.joblib",
        "dataset": "physionet.org/content/challenge-2019/1.0.0/",
        "features": ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
                     "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2",
                     "WBC","Creatinine","Glucose","Lactate","Bilirubin_total",
                     "Age","Gender","HospAdmTime","ICULOS"],
        "auc_target": 0.85,
        "paper": "PhysioNet Challenge 2019",
    },
}
```

### Imaging Models (HuggingFace)
```python
IMAGING_MODELS = {
    # ── CHEST X-RAY ───────────────────────────────────────────
    "xray_pneumonia": "nickmuchi/vit-finetuned-chest-xray-pneumonia",
    "xray_chexpert": "google/vit-base-patch16-224",              # fine-tune on CheXpert
    "xray_14disease": "facebook/dinov2-base",                    # fine-tune on NIH ChestXray14
    
    # ── ZERO-SHOT (ALL MODALITIES) ────────────────────────────
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    
    # ── MRI ───────────────────────────────────────────────────
    "mri_brain_tumor": "Devarshi/Brain_Tumor_Classification",
    "mri_alzheimer": "Falah/Alzheimer_MRI_4_classes_classification",
    "mri_segmentation": "julien-c/segformer-b0-finetuned-ade-512-512",  # adapt
    
    # ── PATHOLOGY / HISTOLOGY ─────────────────────────────────
    "pathology_foundation": "mahmoodlab/uni",                    # UNI foundation model
    "pathology_conch": "mahmoodlab/conch",                       # Vision-language
    "pathology_phikon": "owkin/phikon",                          # Transfer learning
    
    # ── DERMATOLOGY ───────────────────────────────────────────
    "skin_lesion": "dima806/skin_lesions_image_detection",
    "skin_ham10000": "Anwarkh1/Skin_Cancer-Image_Classification",
    
    # ── RETINAL ───────────────────────────────────────────────
    "oct_retinal": "microsoft/swin-tiny-patch4-window7-224",    # fine-tune Kermany
    "fundus_diabetic_retinopathy": "microsoft/resnet-50",       # fine-tune APTOS
    
    # ── ECHOCARDIOGRAPHY ──────────────────────────────────────
    "echo_ef": "EchoNet-Dynamic/EchoNet-Dynamic",               # Stanford EchoNet
    # Install: pip install git+https://github.com/echonet/dynamic.git
    
    # ── GASTROENTEROLOGY ──────────────────────────────────────
    "polyp_segmentation": "reab/colonoscopy-polyp-segmentation",
    "gi_endoscopy": "SunshanHuang/polyp-detection",
}
```

### ECG / Signal Models
```python
ECG_MODELS = {
    # ── ECG CLASSIFICATION ────────────────────────────────────
    "ecg_ptbxl": {
        "hf_id": "Bangxi-Li/ECG-arrhythmia-classification",
        "dataset": "PTB-XL (21,837 records)",
        "classes": 10,   # Normal, LBBB, RBBB, PAC, PVC, STEMI, ...
        "paper": "Wagner et al., Nature Scientific Data 2020",
        "input": "12-lead ECG, 500Hz, 10 seconds = 5000 samples × 12 leads",
    },
    "ecg_af": {
        "hf_id": "Sharan185/ecg_classification",
        "dataset": "PhysioNet 2017 AF Challenge",
        "classes": 4,    # Normal, AF, Other, Noisy
        "paper": "Clifford et al., Physiol. Meas. 2017",
    },
    "ecg_heartbeat": {
        "hf_id": "Ibrahim-MH/ECG-Heartbeat-classification",
        "dataset": "MIT-BIH + PTB merged",
        "classes": 5,    # N, S, V, F, Q (AAMI classes)
        "paper": "Kachuee et al., IEEE EMBC 2018",
    },
    # ── FOUNDATION MODELS (ECG) ───────────────────────────────
    "ecg_foundation": {
        "hf_id": "Alibaba-DAMO/bge-m3-ecg",                    # check HF
        "note": "Or use: pip install ecg-foundation-model",
        "paper": "MERL — ECG Foundation (Chen et al., 2024)",
    },
}
```

### NLP / Clinical Text Models
```python
NLP_MODELS = {
    # ── CLINICAL BERT ─────────────────────────────────────────
    "clinical_bert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "scibert": "allenai/scibert_scivocab_uncased",
    
    # ── NER ───────────────────────────────────────────────────
    "ner_medical": "samrawal/bert-base-uncased_clinical-ner",
    "ner_disease": "allenai/scibert_scivocab_uncased",            # + NER head
    "ner_drugs": "blaze999/Medical-NER",
    
    # ── SUMMARIZATION ─────────────────────────────────────────
    "summarize_radiology": "Falconsai/medical_summarization",
    "summarize_clinical": "philschmid/distilbart-cnn-12-6-samsum",
    
    # ── GENERATION / LLM ──────────────────────────────────────
    "groq_llama3": "meta-llama/Meta-Llama-3-8B-Instruct",        # via Groq API
    "meditron": "epfl-llm/meditron-7b",                          # medical LLaMA
    "medalpaca": "medalpaca/medalpaca-13b",                       # medical instruction
    
    # ── EMBEDDINGS ────────────────────────────────────────────
    "embeddings_general": "sentence-transformers/all-MiniLM-L6-v2",
    "embeddings_medical": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "embeddings_biomed": "FremyCompany/BioLORD-2023",
}
```

### Genomics / Multi-Omics Models
```python
GENOMICS_MODELS = {
    "nucleotide_transformer": "InstaDeep/nucleotide-transformer-2.5b-multi-species",
    "dna_bert": "zhihan1996/DNABERT-2-117M",
    "biogpt": "microsoft/BioGPT-Large",
    "esm2_protein": "facebook/esm2_t33_650M_UR50D",              # protein sequences
    "pharmacogenomics": "Use CPIC API: api.cpicpgx.org",
}
```

---

## 📊 DATASETS — Complete Reference

### Structured Clinical Data
| Dataset | Size | Disease | Link | Kaggle Command |
|---------|------|---------|------|----------------|
| Pima Diabetes | 768 rows | Diabetes | UCI ML Repo | `kaggle datasets download -d uciml/pima-indians-diabetes-database` |
| Cleveland Heart | 303 rows | CAD | UCI ML Repo | `kaggle datasets download -d ronitf/heart-disease-uci` |
| CKD Dataset | 400 rows | Chronic Kidney | UCI ML Repo | `kaggle datasets download -d mansoordaku/ckdisease` |
| Liver Patient (ILPD) | 583 rows | Liver Disease | UCI ML Repo | `kaggle datasets download -d uciml/indian-liver-patient-records` |
| Heart Failure | 299 rows | Heart Failure | Kaggle | `kaggle datasets download -d andrewmvd/heart-failure-clinical-data` |
| MIMIC-IV | 40k+ ICU | Multi-disease | PhysioNet (credentialed) | `physionet.org/content/mimiciv/` |
| PhysioNet Sepsis | 40k patients | Sepsis | PhysioNet | `physionet.org/content/challenge-2019/` |
| CDC BRFSS | 440k rows | Multi-disease | CDC | `kaggle datasets download -d cdc/behavioral-risk-factor-surveillance-system` |

### Imaging Datasets
| Dataset | Size | Modality | Disease | Link |
|---------|------|----------|---------|------|
| CheXpert | 224k | Chest X-Ray | 14 conditions | `stanfordmlgroup.github.io/competitions/chexpert/` |
| NIH ChestX-ray14 | 112k | Chest X-Ray | 14 conditions | `kaggle datasets download -d nih-chest-xrays/data` |
| Kaggle Chest X-Ray | 5.8k | Chest X-Ray | Pneumonia | `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia` |
| BraTS 2023 | 1,251 | Brain MRI | Tumor (4 types) | `kaggle datasets download -d dschettler8845/brats-2021-task1` |
| Brain Tumor MRI | 3,064 | Brain MRI | Tumor | `kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection` |
| EchoNet-Dynamic | 10,030 | Echo video | EF / HFrEF | `echonet.github.io/dynamic/` |
| CAMUS Echo | 500 | Echo | LV function | `creatis.insa-lyon.fr/Challenge/camus/` |
| APTOS | 3,662 | Fundus | Diabetic Retinopathy | `kaggle datasets download -d mariaherrerot/aptos2019` |
| Kvasir-SEG | 1,000 | Colonoscopy | Polyps | `datasets.simula.no/kvasir-seg/` |
| HAM10000 | 10,015 | Dermoscopy | Skin Cancer (7 types) | `kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection` |
| ISIC 2019 | 25,331 | Dermoscopy | Skin Cancer | `kaggle datasets download -d andrewmvd/isic-2019` |
| Kermany OCT | 84,495 | Retinal OCT | CNV/DME/Drusen | `kaggle datasets download -d paultimothymooney/kermany2018` |
| TCGA | 33 cancers | WSI Pathology | Multi-cancer | `portal.gdc.cancer.gov/` |
| CAMELYON16 | 400 | WSI Pathology | Breast metastasis | `camelyon16.grand-challenge.org/` |

### ECG / Signal Datasets
| Dataset | Size | Signal | Disease | Link |
|---------|------|--------|---------|------|
| PTB-XL | 21,837 | 12-lead ECG | 10 conditions | `physionet.org/content/ptb-xl/` |
| MIT-BIH Arrhythmia | 48 records | 2-lead Holter | Arrhythmia | `physionet.org/content/mitdb/` |
| PhysioNet AF 2017 | 8,528 | 1-lead ECG | AF detection | `physionet.org/content/challenge-2017/` |
| ICBHI 2017 | 6,898 cycles | Lung sounds | Respiratory | `bhichallenge.med.auth.gr/` |
| SHHS (Sleep) | 5,804 | PSG (EEG+ECG) | Sleep apnea | `sleepdata.org/datasets/shhs` |
| CHB-MIT EEG | 844h | EEG | Epilepsy | `physionet.org/content/chbmit/` |

### PubMed / Literature (for RAG)
```bash
# Download abstracts per disease domain (use in ingest_pubmed.py)
# Free via Entrez API — no auth needed for <10k/day
# For bulk: download PMC Open Access subset

# Recommended search queries per domain:
PUBMED_QUERIES = {
    "diabetes": "diabetes mellitus severity clinical management[MeSH]",
    "heart_disease": "coronary artery disease risk stratification treatment[MeSH]",
    "pneumonia": "pneumonia diagnosis severity treatment antibiotics[MeSH]",
    "ckd": "chronic kidney disease staging management progression[MeSH]",
    "sepsis": "sepsis diagnosis management ICU outcomes[MeSH]",
    "ecg": "electrocardiogram arrhythmia diagnosis clinical significance[MeSH]",
    "imaging": "radiology findings clinical significance treatment[MeSH]",
}
# Target: 500+ abstracts per disease → embed with all-MiniLM-L6-v2
```

---

## 💻 CURSOR PROMPTS — Ready to Paste

### Prompt 1: Generate `base_model.py`
```
Create src/models/base_model.py with:
- Abstract BaseMedicalModel class with @abstractmethod for: domain, required_features, preprocess(raw_input: Dict) -> np.ndarray, predict(features: np.ndarray) -> PredictionResult
- PredictionResult dataclass with: disease_domain, severity_score (0-100), confidence (0-1), severity_label (Low/Moderate/High/Critical), shap_values Dict[str,float], top_features List[str], missing_features List[str], modality str, timestamp datetime
- score_to_label() method using thresholds: <25=Low, <50=Moderate, <75=High, >=75=Critical
- get_missing_features() method
- ImageAnalysisResult dataclass extending PredictionResult with: modality, model_used, findings List[ImagingFinding], rag_query str
- ImagingFinding dataclass: label, confidence, severity_contribution, region, is_critical bool
All type hints. Python 3.10+.
```

### Prompt 2: Generate `registry.py`
```
Create src/models/registry.py with:
- ModelRegistry class that:
  1. Stores models Dict[str, BaseMedicalModel]
  2. classify_domain(query_text: str) -> str using keyword scoring
  3. route_and_predict(query_text, patient_data) -> PredictionResult
  4. register(model, keywords=[]) method
  5. list_domains() method
  6. Hot-swap support: can replace a model at runtime without restart
- Domain keywords for: diabetes, heart_disease, pneumonia, ckd, liver_disease, sepsis, imaging, ecg, echo
- Raise ValueError with helpful message if domain unknown
- Log routing decisions to structured JSON logger
```

### Prompt 3: Generate `clinical_ner.py`
```
Create src/preprocessing/clinical_ner.py using:
- BioBERT (emilyalsentzer/Bio_ClinicalBERT) via HuggingFace pipeline
- Extract: diagnoses, medications (drug+dose+route+frequency), lab_values (name+value+unit), symptoms, body_parts, temporal expressions
- NegEx algorithm: detect negated entities ("no chest pain" → negated=True)
- Output ClinicalEntities dataclass: diagnoses List[str], medications List[dict], lab_values Dict[str,float], symptoms List[str], negated_diagnoses List[str]
- Fallback to regex patterns if BERT unavailable
- Unit: gram→g, milligram→mg, millimol→mmol/L normalization
```

### Prompt 4: Generate `ecg_processor.py`
```
Create src/preprocessing/ecg_processor.py with:
- ECGLoader class: read_wfdb(path), read_edf(path), read_xml_hl7(path), read_csv(path) → all return np.ndarray shape (n_leads, n_samples)
- ECGProcessor class:
  1. bandpass_filter(signal, low=0.5, high=40, fs=500) using scipy.signal
  2. detect_r_peaks(signal, fs=500) using neurokit2 or biosppy
  3. compute_hrv_features(rpeaks, fs) → dict with SDNN, RMSSD, pNN50, mean_hr
  4. extract_features(signal, fs=500) → ECGFeatures dataclass
- ECGFeatures: hr_mean, hr_std, sdnn, rmssd, pnn50, qrs_duration_ms, qt_interval_ms, qtc_ms, pr_interval_ms, st_deviation_mv
- Map QTc > 500ms → severity contribution 85; QTc 450–500 → 55; QTc < 450 → 10
```

### Prompt 5: Generate `shap_explainer.py`
```
Create src/inference/shap_explainer.py with:
- SHAPExplainer class:
  1. __init__(model, feature_names: List[str])
  2. explain(features: np.ndarray, top_k=5) → Tuple[Dict[str,float], List[str]]
  3. features_to_rag_query(shap_dict, domain) → str
  4. plot_waterfall(shap_dict) → matplotlib Figure
  5. plot_force(shap_dict) → matplotlib Figure
- features_to_rag_query logic:
  - top 3 features by |SHAP value|
  - direction: positive→"elevated", negative→"low"
  - readable label: snake_case → "blood glucose" 
  - output: "diabetes elevated blood glucose low insulin risk factors severity"
- Handle TreeExplainer for XGBoost/LightGBM and KernelExplainer for others
```

### Prompt 6: Generate `vector_store.py`
```
Create src/rag/vector_store.py with:
- MedicalVectorStore using ChromaDB PersistentClient
- Methods:
  1. ingest_documents(docs: List[Dict]) where doc = {id, text, metadata: {source, domain, pmid, year}}
  2. query(query_text, n_results=5, domain_filter=None, min_distance=0.8) → List[Dict]
  3. mmr_query(query_text, n_results=5, diversity=0.3) → List[Dict]  # Maximal Marginal Relevance
  4. ingest_from_pubmed(pmids: List[str]) async method
  5. get_collection_stats() → Dict
- Use FremyCompany/BioLORD-2023 embeddings for medical domain (better than MiniLM for clinical text)
- Fallback to all-MiniLM-L6-v2 if BioLORD unavailable
- Store: document text, metadata (source, domain, pmid, year, title), embedding
```

### Prompt 7: Generate `report_generator.py`
```
Create src/synthesis/report_generator.py with:
- ReportSynthesizer class using LangChain + Groq (Llama-3-8B free tier)
- System prompt template emphasizing:
  1. Evidence-based (only use retrieved literature)
  2. Structured output: SUMMARY, KEY FINDINGS, RISK FACTORS, RECOMMENDATIONS, DISCLAIMER
  3. Severity label + score prominently stated
  4. Always end with "This analysis is AI-assisted and must be reviewed by a qualified clinician"
- generate_report(prediction_result, rag_docs, patient_context) → str
- generate_report_streaming() → AsyncIterator[str] for real-time display
- Safety rails:
  1. Never provide specific drug doses
  2. Always recommend specialist review
  3. Flag if severity=Critical → prepend "⚠️ URGENT MEDICAL ATTENTION RECOMMENDED"
- Token budget: max 800 tokens output
```

### Prompt 8: Generate `main.py` FastAPI
```
Create src/api/main.py FastAPI app with:
Endpoints:
  POST /predict — main prediction endpoint
  POST /predict/image — image file upload prediction
  POST /predict/ecg — ECG file upload prediction
  GET /health — health check
  GET /models — list registered models
  WebSocket /predict/stream — streaming report generation

PatientInput schema:
  query_text: str
  structured_input: Optional[Dict[str, float]]
  image_file: Optional[UploadFile]
  ecg_file: Optional[UploadFile]
  patient_id: Optional[str]
  modality: Optional[str]

SeverityResponse schema:
  patient_id, disease_domain, severity_score (0-100), severity_label,
  confidence, top_risk_factors List[str], shap_values Dict,
  findings List[dict], literature_citations List[dict],
  report_text str, model_used str, timestamp str, disclaimer str

Include:
- CORS middleware (allow all origins for dev)
- Request logging with timing
- Error handling: return 422 with helpful message
- File size limit: 50MB for images, 10MB for ECG
- Background task: log prediction to JSONL file
```

### Prompt 9: Generate `app.py` Streamlit UI
```
Create frontend/app.py Streamlit app with:
Pages (use st.sidebar radio):
  1. "Patient Analysis" — main input form
  2. "History" — past predictions
  3. "Model Info" — registered models

Patient Analysis page:
  - Tab 1: Structured Input (sliders for lab values per disease domain)
  - Tab 2: Text Input (paste clinical notes)  
  - Tab 3: Image Upload (drag-drop for X-Ray/MRI/ECG)
  - Submit button → call FastAPI /predict
  - Results display:
    * Severity gauge (plotly) — 0-100 arc gauge with color zones
    * SHAP bar chart (horizontal, top 10 features)
    * Literature citations with expandable abstracts
    * Full report with copy button
    * Download PDF button (use fpdf2)

Color scheme: severity score < 25 = green, 25-50 = yellow, 50-75 = orange, > 75 = red
Show spinner during prediction. Handle connection errors gracefully.
```

---

## 🔧 KEY CODE PATTERNS

### Training Pattern (Kaggle Notebook)
```python
# Run this in Kaggle Notebook with GPU T4
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import shap
import joblib

def train_disease_model(csv_path: str, target_col: str, model_name: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Train with cross-validation
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="auc",
        early_stopping_rounds=50, random_state=42,
        tree_method="gpu_hist"  # GPU acceleration on Kaggle
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y_res, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Final fit
    model.fit(X_scaled, y_res, verbose=100)
    
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled[:100])
    
    # Save
    joblib.dump({"model": model, "scaler": scaler, "features": list(X.columns),
                 "explainer": explainer}, f"models/{model_name}.joblib")
    print(f"Saved to models/{model_name}.joblib")
    return model, scaler

# Usage:
train_disease_model("data/diabetes.csv", "Outcome", "diabetes_xgb_v2")
train_disease_model("data/heart.csv", "target", "heart_xgb_v2")
```

### Severity Score Calibration
```python
def calibrate_severity(
    probability: float,
    feature_dict: dict,
    domain: str,
    amplifiers: dict = None
) -> float:
    """
    Convert XGBoost probability → clinical severity score (0–100).
    Incorporates domain-specific amplifiers from SHAP top features.
    """
    base = probability * 100
    
    # Domain-specific amplifiers
    AMPLIFIERS = {
        "diabetes": {
            "glucose": lambda v: min(20, max(0, (v - 126) / 10)) if v > 126 else 0,
            "hba1c": lambda v: min(25, max(0, (v - 6.5) * 5)) if v > 6.5 else 0,
        },
        "heart_disease": {
            "trestbps": lambda v: min(15, max(0, (v - 140) / 5)) if v > 140 else 0,
            "chol": lambda v: min(10, max(0, (v - 240) / 20)) if v > 240 else 0,
        },
        "ckd": {
            "creatinine": lambda v: min(30, max(0, (v - 1.2) * 8)) if v > 1.2 else 0,
            "egfr": lambda v: max(0, (60 - v) * 0.5) if v < 60 else 0,
        },
    }
    
    boost = 0
    if domain in AMPLIFIERS and amplifiers:
        for feat, fn in AMPLIFIERS[domain].items():
            if feat in amplifiers:
                boost += fn(amplifiers[feat])
    
    return min(100.0, round(base + boost, 1))
```

### PubMed Ingestor
```python
# scripts/ingest_pubmed.py
import asyncio
import aiohttp
from Bio import Entrez
from src.rag.vector_store import MedicalVectorStore

Entrez.email = "your@email.com"  # Required by NCBI

async def fetch_abstracts(query: str, max_results: int = 500) -> list[dict]:
    """Fetch PubMed abstracts for a medical domain."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    pmids = record["IdList"]
    
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    
    docs = []
    for i, article in enumerate(records["PubmedArticle"]):
        try:
            abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
            pmid = str(article["MedlineCitation"]["PMID"])
            year = int(article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", 2000))
            
            docs.append({
                "id": f"pmid_{pmid}",
                "text": f"{title}\n\n{' '.join(str(a) for a in abstract)}",
                "metadata": {"source": "PubMed", "pmid": pmid, "year": year,
                             "title": title, "domain": "medical"}
            })
        except (KeyError, TypeError):
            continue
    
    return docs

async def ingest_all_domains():
    store = MedicalVectorStore()
    queries = {
        "diabetes": "diabetes mellitus treatment management outcomes",
        "heart_disease": "coronary artery disease treatment percutaneous coronary intervention",
        "pneumonia": "community acquired pneumonia treatment antibiotics severity",
        "ckd": "chronic kidney disease management dialysis outcomes",
        "sepsis": "sepsis management septic shock outcomes critical care",
    }
    for domain, query in queries.items():
        print(f"Ingesting {domain}...")
        docs = await fetch_abstracts(query, 500)
        for doc in docs:
            doc["metadata"]["domain"] = domain
        store.ingest_documents(docs)
        print(f"  Ingested {len(docs)} abstracts for {domain}")

if __name__ == "__main__":
    asyncio.run(ingest_all_domains())
```

### ECG Feature Extraction
```python
# src/preprocessing/ecg_processor.py
import numpy as np
import neurokit2 as nk
import wfdb
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ECGFeatures:
    hr_mean: float; hr_std: float; sdnn: float; rmssd: float; pnn50: float
    qrs_duration_ms: float; qt_interval_ms: float; qtc_ms: float
    pr_interval_ms: float; st_deviation_mv: float; af_probability: float

class ECGProcessor:
    def __init__(self, fs: int = 500):
        self.fs = fs

    def read_wfdb(self, path: str) -> np.ndarray:
        record = wfdb.rdrecord(path)
        return record.p_signal.T  # (n_leads, n_samples)

    def read_edf(self, path: str) -> np.ndarray:
        import mne
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return raw.get_data()

    def process_lead(self, signal_1d: np.ndarray) -> dict:
        """Process single ECG lead with NeuroKit2."""
        cleaned = nk.ecg_clean(signal_1d, sampling_rate=self.fs)
        signals, info = nk.ecg_process(cleaned, sampling_rate=self.fs)
        
        hrv_time = nk.hrv_time(signals, sampling_rate=self.fs, show=False)
        hrv_freq = nk.hrv_frequency(signals, sampling_rate=self.fs, show=False)
        
        return {
            "hr_mean": float(hrv_time.get("HRV_MeanNN", [70])[0]),
            "sdnn": float(hrv_time.get("HRV_SDNN", [50])[0]),
            "rmssd": float(hrv_time.get("HRV_RMSSD", [40])[0]),
            "pnn50": float(hrv_time.get("HRV_pNN50", [20])[0]),
        }

    def extract_features(self, signal_12lead: np.ndarray) -> ECGFeatures:
        """Extract features from 12-lead ECG."""
        lead_ii = signal_12lead[1] if signal_12lead.shape[0] >= 2 else signal_12lead[0]
        hrv = self.process_lead(lead_ii)
        
        # QTc (Bazett formula) — requires QT and RR interval detection
        # Simplified: use neurokit2 intervals
        intervals = nk.ecg_intervalrelated(
            nk.ecg_process(nk.ecg_clean(lead_ii, self.fs), self.fs)[0]
        )
        
        return ECGFeatures(
            hr_mean=hrv["hr_mean"],
            hr_std=0.0,
            sdnn=hrv["sdnn"],
            rmssd=hrv["rmssd"],
            pnn50=hrv["pnn50"],
            qrs_duration_ms=float(intervals.get("ECG_QRS_Duration", 100).values[0]) * 1000 / self.fs,
            qt_interval_ms=float(intervals.get("ECG_QT_Interval", 400).values[0]) * 1000 / self.fs,
            qtc_ms=0.0,  # compute: QT / sqrt(RR_in_seconds)
            pr_interval_ms=float(intervals.get("ECG_PR_Interval", 160).values[0]) * 1000 / self.fs,
            st_deviation_mv=0.0,
            af_probability=0.0,
        )

    def severity_from_ecg(self, features: ECGFeatures) -> float:
        """Map ECG features to severity score 0-100."""
        score = 5.0  # baseline

        # QTc prolongation
        if features.qtc_ms > 500: score += 40
        elif features.qtc_ms > 470: score += 20
        elif features.qtc_ms > 450: score += 10

        # Heart rate
        if features.hr_mean > 150 or features.hr_mean < 40: score += 35
        elif features.hr_mean > 120 or features.hr_mean < 50: score += 15

        # HRV (low HRV = poor autonomic function)
        if features.sdnn < 20: score += 20
        elif features.sdnn < 50: score += 8

        return min(100.0, score)
```

### RAGAS Hallucination Guard
```python
# src/rag/hallucination_guard.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

class HallucinationGuard:
    """Evaluate generated reports for faithfulness to retrieved literature."""
    
    def evaluate_report(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )
        
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "is_faithful": result["faithfulness"] > 0.85,
            "needs_review": result["faithfulness"] < 0.7,
        }
    
    def filter_unfaithful(self, report: str, score: float, threshold: float = 0.7) -> str:
        if score < threshold:
            return (
                f"[Faithfulness score {score:.2f} below threshold — report requires manual review]\n\n"
                + report
            )
        return report
```

---

## 🔬 RESEARCH NOVELTY TRACKING

### Ablation Study Template
```python
# Track these in wandb or MLflow as you build

NOVELTY_EXPERIMENTS = {
    "1_dynamic_routing": {
        "metric": "routing_accuracy_%",
        "target": 95.0,
        "baseline": "keyword_match_only",
        "ablation": "remove_semantic_similarity",
        "datasets": ["100 synthetic patient queries per domain"],
    },
    "2_severity_magnitude": {
        "metric": ["AUC", "clinical_utility_score", "calibration_ECE"],
        "target": "AUC > binary_classification + 0.05 on clinical utility",
        "baseline": "binary_classification_only",
        "datasets": ["Pima", "Cleveland Heart", "CheXpert"],
    },
    "3_shap_guided_rag": {
        "metric": ["retrieval_precision@5", "retrieval_recall@5", "NDCG@5"],
        "target": "SHAP_query > keyword_query by 15%",
        "baseline": "TF-IDF keyword query",
        "ablation": "replace SHAP query with generic domain query",
    },
    "4_ragas_faithfulness": {
        "metric": "hallucination_rate_%",
        "target": "< 5% unfaithful statements",
        "tool": "RAGAS faithfulness metric",
        "datasets": ["50 generated reports vs source documents"],
    },
    "5_missing_data_agent": {
        "metric": ["queries_salvaged_%", "imputation_accuracy"],
        "target": "Salvage > 70% of incomplete queries",
        "baseline": "reject_if_missing",
        "datasets": ["Pima with 30% random missing values"],
    },
    "6_model_registry_latency": {
        "metric": "hot_swap_latency_ms",
        "target": "< 50ms",
        "baseline": "cold_start_load",
    },
}
```

---

## 🐳 DOCKER & DEPLOYMENT

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for imaging
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./chroma_db:/app/chroma_db
      - ./data:/app/data
    env_file: .env
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

### `.env.example`
```bash
# LLM APIs
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxx  # optional

# Storage
CHROMA_PERSIST_DIR=./chroma_db
MODEL_REGISTRY_DIR=./models
DATA_DIR=./data

# PubMed
PUBMED_EMAIL=your@email.com

# App
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50

# HuggingFace
HF_TOKEN=hf_xxxxxxxxxxxx  # for gated models (UNI, CONCH)
```

---

## 📈 FREE COMPUTE RESOURCES

| Platform | Free Tier | GPU | Best For |
|----------|-----------|-----|----------|
| Kaggle Notebooks | Always free | T4 x2 (30h/week) | Model training |
| Google Colab | Free tier | T4 (limited) | Experimentation |
| HuggingFace Spaces | Always free | CPU (ZeroGPU for GPU) | Demo deployment |
| Render.com | Free tier | CPU only | FastAPI hosting |
| Streamlit Cloud | Always free | CPU | UI hosting |
| Groq API | 14,400 req/day free | n/a | Llama 3 inference |
| HuggingFace Hub | Always free | n/a | Model + dataset hosting |
| GitHub | Always free | n/a | Code + CI/CD |

---

## 🔗 KEY REFERENCES

### Papers
```
1. XGBoost: Chen & Guestrin, KDD 2016 — doi:10.1145/2939672.2939785
2. SHAP: Lundberg & Lee, NeurIPS 2017 — arxiv:1705.07874
3. BiomedCLIP: Zhang et al., Nature Methods 2023 — doi:10.1038/s41592-024-02297-2
4. CheXNet: Rajpurkar et al., 2017 — arxiv:1711.05225
5. EchoNet-Dynamic: Ouyang et al., Nature 2020 — doi:10.1038/s41586-020-2145-8
6. PTB-XL: Wagner et al., Scientific Data 2020 — doi:10.1038/s41597-020-0495-6
7. RAGAS: Es et al., 2023 — arxiv:2309.15217
8. LangChain: Chase, 2022 — github.com/langchain-ai/langchain
9. ChromaDB: Chroma, 2023 — docs.trychroma.com
10. UNI (PathFM): Chen et al., Nature Medicine 2024 — doi:10.1038/s41591-024-02857-3
11. CONCH: Lu et al., Nature Medicine 2024 — doi:10.1038/s41591-024-02856-4
12. NucleotideTransformer: Dalla-Torre et al., 2023 — arxiv:2306.15006
13. BioBERT: Lee et al., Bioinformatics 2020 — doi:10.1093/bioinformatics/btz682
14. BioGPT: Luo et al., Briefings in Bioinformatics 2022 — doi:10.1093/bib/bbac409
```

### Useful APIs
```
PhysioNet:        physionet.org/content/
NCBI Entrez:      biopython.org/docs/1.75/api/Bio.Entrez.html
ClinVar:          api.ncbi.nlm.nih.gov/variation/v0/
DrugBank:         go.drugbank.com/releases/latest
OMIM:             omim.org/api/
CPIC PGx:         api.cpicpgx.org/v1/
SNOMED CT:        browser.ihtsdotools.org/
ICD-10:           icd.who.int/browse10/
LOINC:            loinc.org/downloads/
OpenFDA:          open.fda.gov/apis/
```

### Cursor `.cursorrules` (place in project root)
```
You are an expert medical AI engineer building a clinical severity prediction system.
Stack: Python 3.11, FastAPI, XGBoost, SHAP, LangChain, ChromaDB, HuggingFace Transformers, Streamlit.
Architecture: 7-layer pipeline — Input → NER → Model Registry → Ensemble/SHAP → RAG → LLM Synthesis → API.
Severity scores: always 0-100 continuous scale. Labels: Low(<25), Moderate(<50), High(<75), Critical(>=75).
Always use type hints. Prefer dataclasses over dicts. Log decisions to structured JSON.
Medical safety: Never suggest specific drug doses. Always append clinical disclaimer.
Code style: Google style docstrings, black formatting, 88 char line limit.
Test every new model with the smoke test pattern from medical_ai_e2e.py.
```
