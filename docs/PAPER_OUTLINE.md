# Paper outline (results are preliminary)

1. Research question: Does feature-attribution-guided retrieval improve evidence relevance compared with keyword retrieval for disease classifiers?
2. Data: CDC health-indicator survey labels and Cleveland angiographic presence labels, with source terms, population limits, missing values and hashes. Proposed pediatric Kermany X-rays must not be framed as adult severity labels.
3. Methods: stratified 60/20/20 train/calibration/test; train-only imputation; fixed XGBoost; independent sigmoid/Platt calibration; TreeSHAP log-odds explanations; query bridge; MiniLM + Chroma + MMR. Image CNN pipeline is not yet trained.
4. Results: cite `reports/diabetes_metrics.json` and `reports/heart_disease_metrics.json`. Diabetes AUC below target. Calibration worsened ECE/Brier. Small heart test sample has no external validation. No image or retrieval/RAGAS score exists yet.
5. Ablations: independently labeled retrieval judgments; SHAP vs keyword and MMR on/off; missingness masks fixed before evaluation; routing accuracy dataset; hot-swap latency measurements. These experiments remain pending.
6. Safety/ethics: research-only, no dosing, no synthetic evidence citations, no patient-data commits; probability score is not clinical severity, pediatric/population domain shift, privacy/access control, subgroup evaluation.
7. Limitations: dataset mismatch from Pima, survey diagnosis target, lack of prospective/clinician validation, no deployment or clinical utility claim.
8. Conclusion: only conclusions supported by measured outputs. Do not write target scores as achievements.
