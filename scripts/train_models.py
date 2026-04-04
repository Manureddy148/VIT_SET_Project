"""
Train demo XGBoost models and save joblib bundles under models/.
Run: python scripts/train_models.py (from repo root)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.diabetes_model import DiabetesModel
from src.models.heart_model import HeartModel


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    mdir = Path(os.getenv("MODEL_REGISTRY_DIR", root / "models"))
    mdir.mkdir(parents=True, exist_ok=True)

    dm = DiabetesModel()
    dm.train_demo(n_samples=2000, random_state=42)
    dm.save(str(mdir / "diabetes_v1.joblib"))
    print("Saved", mdir / "diabetes_v1.joblib")

    hm = HeartModel()
    hm.train_demo(n_samples=1600, random_state=43)
    hm.save(str(mdir / "heart_v1.joblib"))
    print("Saved", mdir / "heart_v1.joblib")


if __name__ == "__main__":
    main()
