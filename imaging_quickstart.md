# Medical AI Imaging Extension — Quick Start

## What this adds to your project

Your existing system handles **structured lab data** (glucose, BMI, etc.)  
This extension adds **medical image analysis** — MRI, CT, X-Ray, and more —  
using **published open-source models from HuggingFace**. No training required.

---

## Step 1: Install dependencies

```bash
# Core (required)
pip install transformers torch torchvision Pillow numpy

# For BiomedCLIP zero-shot (CT, MRI, all modalities)
pip install open-clip-torch

# For DICOM files (.dcm) — clinical standard format
pip install pydicom pylibjpeg

# For NIfTI files (.nii) — MRI volumes
pip install nibabel
```

---

## Step 2: Models used (all open-source, no training)

| Model Key | HuggingFace ID | Modality | Disease | Paper |
|-----------|----------------|----------|---------|-------|
| `xray_pneumonia` | `nickmuchi/vit-finetuned-chest-xray-pneumonia` | X-Ray | Pneumonia | CheXNet (Rajpurkar 2017) |
| `mri_brain_tumor` | `Devarshi/Brain_Tumor_Classification` | MRI | Brain Tumor | BraTS dataset |
| `biomedclip` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | **All** | Zero-shot | Nature Methods 2023 |
| `oct_retinal` | `microsoft/swin-tiny-patch4-window7-224` | OCT | Retinal | Kermany et al. Cell 2018 |
| `skin_lesion` | `dima806/skin_lesions_image_detection` | Dermoscopy | Skin Cancer | ISIC (Tschandl 2018) |

---

## Step 3: Usage

### Basic — analyze a single image
```python
from imaging_extension import ImageRouter

router = ImageRouter()

# Chest X-Ray
result = router.analyze("patient_xray.jpg", modality="xray")
print(result.severity_score)   # 0-100, same scale as existing system
print(result.severity_label)   # "Low" / "Moderate" / "High" / "Critical"
print(result.findings)         # list of ImagingFinding objects
print(result.rag_query)        # auto-generated for MedicalRAG

# MRI
result = router.analyze("brain_mri.dcm", modality="mri")

# CT — uses BiomedCLIP zero-shot
result = router.analyze("thorax_ct.dcm", modality="ct")

# Auto-detect modality from DICOM metadata
result = router.analyze("unknown_scan.dcm")
```

### Plug into existing ModelRegistry
```python
from imaging_extension import ImagingPipelineBridge

bridge = ImagingPipelineBridge()

# Register alongside diabetes, heart_disease models
registry = ModelRegistry()
registry._models["imaging"] = bridge
registry._domain_keywords["imaging"] = [
    "xray", "x-ray", "mri", "ct scan", "chest", "scan",
    "imaging", "radiograph", "ultrasound", "brain scan"
]

# Now works like any other domain:
result = registry.route_and_predict(
    "Patient has chest x-ray showing bilateral opacity",
    {"image_path": "/data/chest.jpg"}
)
```

### Feed result into existing RAG + LLM pipeline
```python
# ImageAnalysisResult bridges to PredictionResult format
imaging_result = router.analyze("chest.jpg", modality="xray")

# Convert for existing pipeline
pred_dict = imaging_result.to_prediction_result_dict()
# {
#   "disease_domain": "imaging_xray",
#   "severity_score": 78.5,
#   "confidence": 0.92,
#   "severity_label": "High",
#   "shap_values": {"Pneumonia": 72.3},
#   "top_features": ["Pneumonia"],
#   ...
# }

# Use auto-generated RAG query
rag_results = vector_store.query(imaging_result.rag_query, n_results=5)
```

---

## Step 4: Add new models anytime

Just add an entry to `IMAGING_MODEL_REGISTRY`:

```python
IMAGING_MODEL_REGISTRY["my_new_model"] = {
    "hf_id": "username/my-medical-model",
    "task": "image-classification",
    "modalities": ["ct"],
    "diseases": ["lung_cancer"],
    "paper": "My Paper (2024)",
    "input_size": (224, 224),
    "severity_map": {"Cancer": 90.0, "Normal": 5.0},
    "description": "My custom model",
}
```

Then instantiate it:
```python
router = ImageRouter()
result = router.analyze("ct.dcm", modality="ct", model="my_new_model")
```

---

## Project structure update

```
medical-ai-severity/
├── src/
│   ├── models/
│   │   ├── imaging/                        ← NEW
│   │   │   ├── __init__.py
│   │   │   ├── imaging_extension.py        ← this file
│   │   │   ├── image_router.py
│   │   │   └── dicom_reader.py
│   │   ├── diabetes_model.py               (existing)
│   │   └── heart_model.py                  (existing)
│   └── ...
├── data/
│   └── imaging/                            ← NEW
│       ├── sample_xrays/
│       └── sample_mris/
```

---

## Kaggle datasets to download (free)

```bash
# Chest X-Ray (5,863 images)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/imaging/

# Brain MRI (3,064 images)
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection -p data/imaging/

# Retinal OCT (84,495 images)
kaggle datasets download -d paultimothymooney/kermany2018 -p data/imaging/

# Skin Lesions ISIC 2019
kaggle datasets download -d andrewmvd/isic-2019 -p data/imaging/
```

---

## Run smoke test (no model downloads needed)

```bash
python imaging_extension.py
```
