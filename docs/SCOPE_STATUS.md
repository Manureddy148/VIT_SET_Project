# Full roadmap coverage and remaining work

Snapshot: 2026-10-05. Source files: all five named `Create Roadmap` files, plus `medical-ai-architecture.html`. The folder also contains `medical_ai_team_reference_pipeline.html`, retained. Existing implementation is not proof of model validity. This is a milestone, NOT completion of all roadmap items.

## Foundation and primary models

| Item | State | Evidence / remaining |
|---|---|---|
| Package layout, env, smoke/API tests, CI | Implemented and run locally | 50 tests passed; new CI config not run on GitHub yet. Inherited legacy code remains. |
| Diabetes + heart datasets | Downloaded and full-data training run | CDC UCI fallback instead of gated Pima; Cleveland UCI. `docs/DATASETS.md`, hashes and reports. |
| Pneumonia images | Not downloaded/trained | Public Mendeley CC BY source verified; actual download endpoint not verified. Need documented patient mapping and manifest. |
| EDA | Two notebooks executed | Diabetes + heart summary tables; pneumonia manifest notebook built but unrun. No population/fairness analysis yet. |
| Groq key test | Not done | No scoped key; no paid LLM request made. |
| Diabetes model AUC > .85 | Run, target NOT met | AUC .825507, fixed full-data held-out split; do not tune on that test set. |
| Heart model AUC > .85 | Run, target met on one small split | AUC .948052, test n=61; not external validation. |
| Pneumonia true image model | Pipeline built, forward/leakage tests run | CNN, calibration, checkpoint/inference. No image-data AUC. Prior proxy inference disabled by default. |
| SHAP | Real tabular TreeSHAP run | 21 CDC / 13 Cleveland features. Underlying-tree log-odds contributions, not calibrated-probability SHAP. Image SHAP NOT done. |
| Platt + ECE | Both tabular models run | Separate calibration/test. ECE worsened for both; no improvement claim. |
| Severity bands | Implemented | A classification probability display only; datasets do NOT label validated severity. Clinical utility claim blocked. |
| Model artifacts | Saved locally, excluded from Git | Reproduce with commands; never publish patient datasets casually. |
| CKD / liver / sepsis | Inherited synthetic demos only | Not real-data trained, disabled outside explicit demo mode. Dataset-specific training/terms still needed. |

## Retrieval, safety, API and UI

| Item | State | Evidence / remaining |
|---|---|---|
| 500+ real PubMed abstracts per primary disease | Fetched + indexed locally | 500 each, 1,500 Chroma entries read back; publisher copyright respected by excluding corpus from Git. |
| BioLORD embedding | Not run | MiniLM used explicitly, no claim BioLORD is better based on this run. |
| SHAP-to-query + MMR | Implemented and run retrieval | 5 results/source URLs per domain. Relevance quality is not established by nonempty results. |
| P@5 / R@5 / nDCG@5 / 15% gain | Metric/evaluator built; no scientific scores | Requires independently judged relevant IDs and both rankings. No made-up ground truth. |
| RAGAS > .8 / .9, 50-report audit | Actual report-level evaluator added, unrun | Needs compatible evaluator LLM/key and reports. Snippet/source checks are NOT RAGAS. |
| Fabricated citation / no-evidence guard | Added and tested | No invented named guideline fallback; missing evidence fails, dose text fails; snippet guard cannot prove factual entailment. |
| /predict, /models, missing-data questions | Implemented, real tabular API calls run | Both real datasets work; unknown/no-match schemas reject rather than silently substitute. |
| Image API | Implemented; no-artifact path tested | 503 without trained checkpoint; never converts pixels to invented SpO2/CRP. |
| Stream + PDF | Inherited SSE and PDF, tests run | Not the roadmap's WebSocket protocol. PDF bytearray fix; UI/PDF pixel review separately. |
| UI gauge, SHAP, citations, disclaimer | Implemented | Current schemas displayed; citation links added. No claim clinician usability validation. |
| History / model info / copy UX | Partial | Model schema sidebar, inherited session behavior; dedicated history page and copy control not complete. |
| All three diseases E2E | Not complete | Real X-ray artifact missing. Demo-only tests are not clinical/model validation. |
| Audit log | Implemented JSONL | No PHI persistence permission; research deployment still needs access control/retention. |
| Missing-data LLM dialogue | Deterministic questions tested | LLM conversational loop, salvage/imputation benchmark not run. |

## Expanded modalities and full architecture

| Item | State | Evidence / remaining |
|---|---|---|
| 14-modality routing | Identifier routing tested | text, tabular, X-ray, CT, MRI, ultrasound, echo, derm, fundus, OCT, pathology, ECG, audio, genomics. No claim 14 trained models. |
| ECG filtering, R peaks, HRV | Features implemented/tested on constructed signal | Bandpass/peak/SDNN/RMSSD/pNN50; no rhythm classifier. QT/QTc/PR/QRS/ST delineation, WFDB/EDF/HL7 loaders and PTB-XL training not done. |
| Audio decoded features | Built | Real WAV decoder, RMS/ZCR. Respiratory/ASR model not trained; no diagnosis from compressed bytes. |
| Genomics VCF | Parser tested | No disease risk from variant keywords. DNA/ESM embeddings, CPIC/pharmacogenomics not implemented. |
| DICOM / NIfTI | Decoder built | pydicom/nibabel optional; no real volumes tested. No MRI tumor/Alzheimer classifier or segmentation training. |
| X-ray CheXpert / NIH 14 classes | Not done | Separate labels/data agreements, training and evaluation needed; not the binary Kermany model. |
| BiomedCLIP / DINO / ViT / Swin / ResNet catalog | Reference candidates only | IDs in roadmap are not treated as verified runnable/clinically validated models. Existing imaging extension retained, not trusted for production. |
| Dermatology / fundus / OCT | Not trained | Need datasets, model cards, licenses, splits and external tests. |
| Pathology UNI/CONCH/Phikon / WSI tiling | Not implemented | Gated model/data terms and native libraries; no pretending pretrained backbone is a task model. |
| Ultrasound / EchoNet / endoscopy | Not trained | Video loaders, task labels, EF/segmentation pipelines outstanding. |
| Clinical BERT/BioBERT/NegEx entities | Existing regex labs only | Full model NER, medications/negation/temporal extraction unrun/unimplemented. |
| Hot-swap | Atomic manager tested | Versions + timestamps; not integrated admin API; no claimed <50ms benchmark. |
| Dynamic semantic domain classifier | Not completed | Existing keywords/structured inference; no 95% routing/semantic benchmark. |
| Fusion / ensemble / novelty ablations | Existing scaffold only | No empirical clinical utility, multimodal fusion validation, missingness salvage claim. |
| FHIR / video / multi-provider models | Not implemented | No endpoint pretending these are supported. |

## Delivery and research

| Item | State | Remaining |
|---|---|---|
| Docker | Files retained, build unrun | No Docker executable available here. GPU/image dependencies separate. |
| Render / Streamlit Cloud | Not deployed | No account/deployment setup, no credentials requested for these. |
| README / paper outline | Added | Methods/results and limitations use actual run reports only. |
| Demo video | Not recorded | Depends on complete chosen demo path. |
| Team clone/push access | Not verified | Do not change collaborator audience without user confirmation. |
| Publish models/data / daily commits | Not performed | Remote push blocked on GitHub 2FA at this snapshot. Main and existing branches stay untouched. |
| Clinician validation | Not done | Requires actual reviewer protocol/ethics approval; no invented feedback or scientific outcomes. |
