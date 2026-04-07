# Pipeline Alignment (Phase 1)

This branch aligns the current codebase with the architecture/team-guide docs in a practical way.

## What Is Already Aligned

- Layer 1-3 core flow exists: text input -> preprocessing -> domain routing.
- Layer 4 exists: severity scoring + SHAP top feature extraction.
- Layer 5 exists: ChromaDB + SHAP-to-RAG query bridge.
- Layer 7 exists: FastAPI endpoint (`/predict`) + Streamlit UI.

## What This Branch Adds

- Layer 6 upgrade: optional LangChain + Groq (Llama) report synthesis.
- Safe fallback: if LLM is disabled/unavailable, template report generation still works.
- New endpoint: `/pipeline/status` for implemented vs planned capabilities.

## Important Gap Reality (Docs vs Code)

The HTML docs include full multimodal production scope (DICOM, ECG, audio, genomics, extra endpoints).
Those are larger Phase 2/3 tracks and are not realistic to fully "redo" in one pass.

Current high-impact gap list:

1. Multimodal ingestion endpoints (`/predict/image`, `/predict/ecg`) are not implemented.
2. PubMed-scale ingestion pipeline and faithfulness scoring are not implemented.
3. LLM synthesis is now integrated as optional, but needs API key and packages.

## How To Enable LangChain LLM Synthesis

1. Install optional packages:
   - `pip install -r requirements-rag.txt`
2. Set environment variables:
   - `MEDICAL_AI_USE_LLM=1`
   - `GROQ_API_KEY=<your_key>`
   - Optional: `MEDICAL_AI_LLM_MODEL=llama-3.1-8b-instant`
3. Start API and call `/predict`.

If disabled or missing keys, the API falls back to deterministic template reports.

## Recommended Next 3 Tasks (Team-Friendly)

1. Add `/predict/image` skeleton with validation + placeholder routing.
2. Build PubMed ingestion script to seed ChromaDB with real abstracts per domain.
3. Add faithfulness evaluator for citations (assert explanation cites retrieved sources only).
