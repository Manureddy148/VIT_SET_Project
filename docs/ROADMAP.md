# Medical AI Roadmap (Open-Source, No Custom Training)

## Goals and constraints
- **No custom training**: we only use pre-trained, open-source models (primarily Hugging Face).
- **Per disease/domain, use 2–3 models**: run in parallel and fuse results (robustness, uncertainty estimation).
- **Evidence-grounded answers**: RAG-backed citations from trusted sources (guidelines + papers).
- **Safety-first**: disclaimers, escalation, and hallucination guardrails.

---

## Phase 0 — Repo + Baseline (Week 1)
- Define a stable **domain list** (e.g., diabetes, heart disease, pneumonia, imaging-xray).
- Create a **model-pack contract** (common I/O):
  - Input: normalized patient context (text + optional files) + extracted entities.
  - Output: \(0–100\) severity score + confidence + key signals (labels/probs) + provenance.
- Add a **model registry** (config-driven): domains → list of model keys (2–3 each) + fusion policy.

Deliverables
- `docs/MODEL_CATALOG.md` drafted with initial model candidates per domain.
- `medical-ai-architecture.html` aligned with “open-source, no training” approach.

---

## Phase 1 — Orchestrator + Routing (Weeks 2–3)
- Build the LLM orchestrator prompt / schema:
  - intent + domain routing
  - entity extraction
  - missing-data question loop
  - tool/model selection
- Implement domain routing to:
  - **structured** (tabular) pack
  - **text-only** pack
  - **imaging** pack (optional)

Deliverables
- Deterministic JSON schema for orchestrator output.
- Model registry config with at least **3 domains** populated.

---

## Phase 2 — Per-domain model packs (Weeks 4–6)
For each disease/domain, implement a “pack” that runs **2–3 pre-trained models** and returns a fused severity.

Recommended pack design
- **Model A**: strong general model for the domain (high recall).
- **Model B**: specialist model (higher precision).
- **Model C (optional)**: lightweight fast model for latency + redundancy.

Fusion (start simple)
- Weighted average over calibrated probabilities → severity mapping.
- Track disagreement:
  - low disagreement → auto respond
  - high disagreement → ask clarifying question / escalate

Deliverables
- 3 domain packs integrated (example set: diabetes, heart disease, pneumonia or imaging-xray).
- A single end-to-end inference run producing:
  - severity score
  - confidence
  - key signals (labels/probabilities)
  - model provenance (which models ran)

---

## Phase 3 — RAG knowledge base (Weeks 7–8)
- Ingest trusted sources (start small, expand later):
  - WHO/CDC/NHS guideline pages (where permitted)
  - PubMed abstracts
  - curated PDFs (manual)
- Build retrieval with:
  - embeddings (sentence-transformers)
  - hybrid search (BM25 + vector)
  - MMR reranking
- Wire “signals” → retrieval queries:
  - entities (symptoms, labs)
  - predicted label(s)
  - model pack domain

Deliverables
- RAG returns **cited evidence** for each response.
- Guardrail: “no citation → do not assert as fact”.

---

## Phase 4 — Safety + evaluation (Weeks 9–10)
- Add guardrails:
  - medical disclaimer injection
  - urgent escalation triggers (severity threshold + red flag symptoms)
  - refuse illegal/dangerous requests
- Add evaluation:
  - RAG faithfulness checks (e.g., RAGAS where feasible)
  - scenario-based test set (synthetic patient cases)
  - logging + audit trail (inputs, routing, models used, citations)

Deliverables
- A measurable “hallucination budget” (target: near-zero in cited claims).
- Test suite with at least 20 curated cases per domain.

---

## Phase 5 — Product surface (Weeks 11–12)
- FastAPI endpoints:
  - `/analyze` (multi-modal request)
  - `/health`
  - `/models` (registry introspection)
- UI (Streamlit/Gradio):
  - severity gauge + confidence band
  - “why” section (signals + citations)
  - “what to do next” triage guidance

Deliverables
- Demo-ready application with 3 domains and citations.

---

## Phase 6 — Expand domains (Weeks 13+)
- Add more diseases/domains by:
  - updating registry (2–3 new models)
  - adding severity mapping + calibration
  - adding evidence sources + tests

Deliverables
- Repeatable playbook: add a new domain in < 1 week.

