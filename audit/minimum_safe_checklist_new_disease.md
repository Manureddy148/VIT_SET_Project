# Minimum Safe Checklist for Adding a New Disease Model

Use this checklist for every new disease domain before merging.

1. Implement model contract
- Create `src/models/<domain>_model.py` inheriting `BaseMedicalModel`.
- Define `domain`, `required_features`, `preprocess`, `predict`, `train`, `train_demo`, `save`, `load`.

2. Register in bootstrap
- Import model in `src/api/bootstrap.py`.
- Register with robust domain keywords and model path env var.

3. Structured fallback routing
- Ensure `required_features` is complete; fallback routing now uses feature-overlap scoring via registry.
- Validate routing works with sparse `query_text` and only structured labs.

4. Multimodal mapping (if relevant)
- For imaging, set env override when needed:
  - `MEDICAL_AI_MODALITY_DOMAIN_MAP_JSON={"xray":"pneumonia","derm":"skin_cancer","mri":"brain_tumor"}`
- Add modality-specific proxy feature mapping in `src/preprocessing/multimodal_router.py` if required.

5. RAG datasource setup
- Add PubMed query block to `scripts/ingest_pubmed.py` `PUBMED_SEARCH_CONFIG`.
- Add seed abstracts to `SAMPLE_ABSTRACTS` for offline/dev mode.
- Run ingestion:
  - `python scripts/ingest_pubmed.py --domains <new_domain> --limit 500`

6. Smoke tests
- Add model fixture + synthetic patient generator in `tests/test_e2e_smoke.py`.
- Add at least:
  - 10-patient validity test
  - high-severity edge case
  - normal/low-risk sanity case

7. RAGAS evaluation
- Add evaluation prompts in `scripts/eval_ragas.py` `DOMAIN_QUERIES`.
- Run:
  - `python scripts/eval_ragas.py --domains <new_domain>`
- Targets:
  - `faithfulness >= 0.90`
  - `answer_relevancy >= 0.80`

8. Final verification
- Run full tests:
  - `pytest tests/ -q`
- Confirm API route works for the new domain and returns citations + disclaimer.
