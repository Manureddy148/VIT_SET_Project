import os
from pathlib import Path

from src.models.diabetes_model import DiabetesModel
from src.models.heart_model import HeartModel
from src.models.pneumonia_model import PneumoniaModel
from src.models.registry import ModelRegistry


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
