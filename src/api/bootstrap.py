import json
import os
from pathlib import Path

from src.models.ckd_model import CKDModel
from src.models.diabetes_model import DiabetesModel
from src.models.heart_model import HeartModel
from src.models.liver_model import LiverDiseaseModel
from src.models.pneumonia_model import PneumoniaModel
from src.models.registry import ModelRegistry
from src.models.sepsis_model import SepsisModel


def _model_dir() -> Path:
    return Path(os.getenv("MODEL_REGISTRY_DIR", "models"))


def build_registry() -> ModelRegistry:
    reg = ModelRegistry()
    mdir = _model_dir()

    dm = DiabetesModel()
    dpath = os.getenv("DIABETES_MODEL_PATH", str(mdir / "diabetes_v1.joblib"))
    if Path(dpath).exists():
        dm.load(dpath)
    else:
        dm.train_demo()
    reg.register(
        dm,
        keywords=[
            "diabetes",
            "glucose",
            "blood sugar",
            "insulin",
            "hba1c",
            "urination",
            "bmi",
        ],
    )

    hm = HeartModel()
    hpath = os.getenv("HEART_MODEL_PATH", str(mdir / "heart_v1.joblib"))
    if Path(hpath).exists():
        hm.load(hpath)
    else:
        hm.train_demo()
    reg.register(
        hm,
        keywords=[
            "heart",
            "heart disease",
            "cardiac",
            "chest pain",
            "cholesterol",
            "ecg",
            "troponin",
        ],
    )

    pm = PneumoniaModel()
    ppath = os.getenv("PNEUMONIA_MODEL_PATH", str(mdir / "pneumonia_v1.joblib"))
    if Path(ppath).exists():
        pm.load(ppath)
    else:
        pm.train_demo()
    reg.register(
        pm,
        keywords=[
            "pneumonia",
            "lung",
            "cough",
            "spo2",
            "oxygen",
            "respiratory",
            "crp",
            "fever",
        ],
    )

    ckm = CKDModel()
    ckpath = os.getenv("CKD_MODEL_PATH", str(mdir / "ckd_v1.joblib"))
    if Path(ckpath).exists():
        ckm.load(ckpath)
    else:
        ckm.train_demo()
    reg.register(
        ckm,
        keywords=[
            "ckd",
            "chronic kidney",
            "kidney disease",
            "creatinine",
            "egfr",
            "renal",
            "dialysis",
            "nephropathy",
        ],
    )

    sm = SepsisModel()
    spath = os.getenv("SEPSIS_MODEL_PATH", str(mdir / "sepsis_v1.joblib"))
    if Path(spath).exists():
        sm.load(spath)
    else:
        sm.train_demo()
    reg.register(
        sm,
        keywords=[
            "sepsis",
            "septic",
            "bacteremia",
            "sofa",
            "qsofa",
            "lactate",
            "infection shock",
            "systemic infection",
        ],
    )

    lm = LiverDiseaseModel()
    lpath = os.getenv("LIVER_MODEL_PATH", str(mdir / "liver_v1.joblib"))
    if Path(lpath).exists():
        lm.load(lpath)
    else:
        lm.train_demo()
    reg.register(
        lm,
        keywords=[
            "liver",
            "liver disease",
            "hepatitis",
            "cirrhosis",
            "bilirubin",
            "jaundice",
            "alt",
            "ast",
            "alkaline phosphatase",
            "albumin",
            "liver function",
        ],
    )

    return reg


def try_vector_store():
    if os.getenv("MEDICAL_AI_SKIP_RAG", "").lower() in ("1", "true", "yes"):
        return None
    try:
        from src.rag.vector_store import MedicalVectorStore

        return MedicalVectorStore()
    except Exception:
        return None


def seed_vector_store_if_empty(store) -> None:
    if store is None:
        return
    try:
        if store.count() > 0:
            return
    except Exception:
        return
    
    # Try loading real PubMed documents first
    pubmed_docs_path = Path("data/processed/pubmed_docs.json")
    if pubmed_docs_path.exists():
        try:
            with open(pubmed_docs_path, "r", encoding="utf-8") as f:
                pubmed_docs = json.load(f)
            if pubmed_docs:
                try:
                    store.ingest_documents(pubmed_docs)
                    return
                except Exception:
                    pass
        except Exception:
            pass
    
    # Fallback to synthetic seed documents
    docs = [
        {
            "id": "seed_diabetes_1",
            "text": (
                "Diabetes care: elevated fasting glucose and BMI are key risk factors; "
                "lifestyle and physician-guided therapy reduce complications."
            ),
            "metadata": {"source": "Synthetic seed (replace with PubMed)", "domain": "diabetes"},
        },
        {
            "id": "seed_heart_1",
            "text": (
                "Cardiovascular risk: hypertension and hyperlipidemia increase "
                "major adverse cardiac events; statins and BP control are standard."
            ),
            "metadata": {"source": "Synthetic seed (replace with PubMed)", "domain": "heart_disease"},
        },
        {
            "id": "seed_lung_1",
            "text": (
                "Pneumonia: hypoxemia and tachypnea warrant urgent evaluation; "
                "oxygenation and antimicrobial therapy per clinical guidelines."
            ),
            "metadata": {"source": "Synthetic seed (replace with PubMed)", "domain": "pneumonia"},
        },
    ]
    try:
        store.ingest_documents(docs)
    except Exception:
        pass
