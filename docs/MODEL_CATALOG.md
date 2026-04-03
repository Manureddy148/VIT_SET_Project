# Model Catalog (Open-Source, No Custom Training)

This project uses **pre-trained** (open-source) models only. For each disease/domain we run **2–3 models** and fuse their outputs.

## Model-pack contract (what every domain must output)
- **domain**: string
- **models_used**: list of model identifiers
- **signals**: per-model labels/probabilities (or logits)
- **severity_score**: \(0–100\)
- **confidence**: \(0–1\)
- **disagreement**: \(0–1\) (how much models differ)
- **notes**: short, machine-readable flags (e.g., `needs_followup`, `escalate`)

---

## How to choose 2–3 models per disease
- Pick **different inductive biases**:
  - text classifier + retrieval-based judge
  - image classifier + zero-shot CLIP-like model
- Ensure at least one model is **lightweight** for latency.
- Prefer models with:
  - clear licensing
  - good documentation and reproducible checkpoints
  - clinically relevant validation (when available)

---

## Initial domains (example set)

### 1) Pneumonia (X-Ray)
- **Model A (specialist)**: `nickmuchi/vit-finetuned-chest-xray-pneumonia`
- **Model B (generalist, zero-shot)**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Model C (optional)**: add a second chest X-ray classifier if needed for robustness

Inputs
- Image file (jpg/png) or DICOM → converted to RGB

Outputs
- label probabilities → severity mapping (e.g., Normal=5, Pneumonia=80–95)

---

### 2) Brain tumor (MRI)
- **Model A (specialist)**: `Devarshi/Brain_Tumor_Classification`
- **Model B (generalist, zero-shot)**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Model C (optional)**: second MRI classifier (BraTS-style) if available/needed

Inputs
- DICOM/NIfTI (if supported) or PNG slices

---

### 3) Skin lesion (dermoscopy)
- **Model A (specialist)**: `dima806/skin_lesions_image_detection`
- **Model B (generalist, zero-shot)**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

---

## Notes on structured/tabular diseases (diabetes, heart disease)
If the project must remain **no-training**, use one of these approaches:
- **Rule + calibration baseline**: clinical score calculators (where appropriate) + confidence heuristics.
- **Public pre-trained tabular models** (if available/licensed): treat as inference-only.
- **LLM-structured reasoning + RAG**: extract values, compute risk categories, cite guidelines; do not claim “model-diagnosed” unless a validated model exists.

When we add a tabular domain, we must document:
- the exact model/checkpoint or score source
- what “severity score” means
- limitations and escalation logic

