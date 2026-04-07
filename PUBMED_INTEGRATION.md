# PubMed Integration for Medical AI RAG Pipeline

This document describes the real-time PubMed integration that enables the medical AI system to fetch and use live clinical literature for grounded predictions.

## Overview

**Previous State**: The system used static, hand-coded seed abstracts (~3-5 documents per disease).

**Current State**: The system now fetches real medical abstracts from NCBI PubMed's Entrez API for each disease domain at ingestion time, providing:
- **73+ real abstracts** (per ingestion run) from peer-reviewed literature
- **Dynamic retrieval** based on SHAP-identified risk factors from predictions
- **Citation accuracy** — all explanations are grounded in indexed literature
- **Scalability** — can be extended to hundreds of abstracts per domain

## How It Works

### 1. PubMed Search Configuration

Each disease domain has a curated search query in [scripts/ingest_pubmed.py](scripts/ingest_pubmed.py):

```python
PUBMED_SEARCH_CONFIG = {
    "diabetes": {
        "query": '(diabetes OR "type 2 diabetes" OR "glucose intolerance") AND (severity OR prediction OR prognosis)',
        "min_abstracts": 30,
    },
    "heart_disease": {
        "query": '(heart disease OR "ischemic heart" OR coronary) AND (severity OR prediction OR prognosis)',
        "min_abstracts": 30,
    },
    "pneumonia": {
        "query": '(pneumonia OR "respiratory infection" OR hypoxemia) AND (severity OR prediction OR prognosis)',
        "min_abstracts": 30,
    },
}
```

### 2. Ingestion Process

The ingestion script performs these steps:

1. **Search** → Call NCBI Entrez `esearch` to find up to N article IDs matching the query
2. **Fetch** → Use Entrez `efetch` to retrieve full article metadata (title + abstract)
3. **Parse** → Extract PMC ID, title, and abstract text from XML response
4. **Validate** → Keep only substantial abstracts (> 50 chars) and cap at 2000 chars
5. **Save** → Write JSON with document ID, text, and metadata (domain, source PMID, title)
6. **Ingest** → Store in ChromaDB vector database with embeddings

### 3. Startup Integration

When the FastAPI service starts, [src/api/bootstrap.py](src/api/bootstrap.py) automatically loads ingested documents:

```python
def seed_vector_store_if_empty(store):
    # Check for real PubMed documents first
    if pubmed_docs_path.exists():
        store.ingest_documents(pubmed_docs)  # Load real abstracts
        return
    
    # Fallback to synthetic seed if PubMed unavailable
    store.ingest_documents(sample_abstracts)
```

### 4. Retrieval at Prediction Time

When a patient prediction is made:

1. XGBoost predicts severity score (0–100)
2. SHAP identifies top 5 risk factors (e.g., glucose, BMI, age)
3. These feature names become RAG query terms → `retrieve_literature()`
4. ChromaDB returns semantically similar abstracts
5. LLM or template report cites these abstracts
6. Faithfulness checker verifies citations appear in report

## Running PubMed Ingestion

### Quick Start (Windows)

```powershell
# Run setup script
.\scripts\setup_pubmed.bat

# Or manually:
python scripts\ingest_pubmed.py --domains diabetes heart_disease pneumonia --limit 30
```

### Quick Start (Linux/Mac)

```bash
bash scripts/setup_pubmed.sh

# Or manually:
python scripts/ingest_pubmed.py --domains diabetes heart_disease pneumonia --limit 30
```

### Options

```
python scripts/ingest_pubmed.py \
  --domains diabetes heart_disease pneumonia  # Which disease domains to fetch
  --limit 30                                    # Max articles per domain
  --offline                                     # Use only sample data (no network)
  --output data/processed/pubmed_docs.json     # Where to save JSON
```

### Output

```
[info] Processing domains: ['diabetes', 'heart_disease', 'pneumonia']
[info] Searching PubMed for diabetes...
[info] Found 25 articles; fetching abstracts...
[info] Successfully fetched 20 abstracts for diabetes
[info] Domain 'diabetes': 22 documents

✓ Wrote 73 total documents to data\processed\pubmed_docs.json
```

## Document Structure

Each ingested document:

```json
{
  "id": "pubmed_36754521",
  "text": "Glycemic Control and Cardiovascular Risk in Type 2 Diabetes. Intensive glycemic control remains a cornerstone of diabetes management...",
  "metadata": {
    "domain": "diabetes",
    "source": "PubMed:36754521",
    "pmid": "36754521",
    "title": "Glycemic Control and Cardiovascular Risk in Type 2 Diabetes"
  }
}
```

## Fallback Behavior

If PubMed API is unavailable:
- ✅ API still works (uses seed abstracts)
- ✅ Predictions still output severity scores and SHAP explanations
- ✅ Citations are still generated (from fewer sources)
- ❌ Literature citations may be generic samples

The system **never crashes** due to network issues; all PubMed features are optional.

## Rate Limiting

NCBI Entrez API has limits:
- **Without email**: ~3 requests/second
- **With email header**: ~10 requests/second (recommended for production)

The script adds `time.sleep(1)` between domain requests to stay compliant.

**For production**: Update the script to add:
```python
headers = {
    'User-Agent': 'MedicalAI/1.0 (research@example.com)'
}
```

## Verification

Check that documents loaded in ChromaDB:

```bash
# API will log on startup:
python -m uvicorn src.api.main:app
# Output: "ChromaDB loaded with 73 documents"

# Or query directly:
python -c "
from src.rag.vector_store import MedicalVectorStore
store = MedicalVectorStore()
print(f'Documents in store: {store.count()}')
"
```

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: User Input (text, image, ECG, audio, genomics)   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYERS 2–4: Preprocessing → Routing → ML Prediction        │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
           ┌───────────────────────┐
           │ SHAP Risk Factors     │
           │ (glucose, BMI, etc)   │
           └───────────┬───────────┘
                       ↓
        ┌──────────────────────────────┐
        │  LAYER 5: RAG RETRIEVAL      │
        │                              │
        │  Query: "glucose BMI age"    │
        │  ↓                           │
        │  ChromaDB Semantic Search    │
        │  ↓                           │
        │  **PubMed Abstracts** ← NEW! │
        │  (73 real documents)         │
        └──────────────────────────────┘
                       ↓
        ┌──────────────────────────────┐
        │  LAYER 6: Synthesis          │
        │  + Safety Rails              │
        │  + Faithfulness Check        │
        └──────────────────────────────┘
                       ↓
        ┌──────────────────────────────┐
        │  LAYER 7: API + Streamlit    │
        │  (Citations + Report)        │
        └──────────────────────────────┘
```

## Testing

```bash
# Test with real PubMed data:
pytest tests/test_api.py::test_predict_diabetes -v

# Full suite (should still pass):
pytest -q
```

## Next Steps (Phase 2)

- [ ] Add NCBI email header for higher rate limits
- [ ] Implement incremental ingestion (only fetch new articles weekly)
- [ ] Add fine-tuning of search queries based on user feedback
- [ ] Cache GraphQL queries to NCBI for better search precision
- [ ] Integrate MeSH term standardization for broader matching

## References

- NCBI Entrez API: https://www.ncbi.nlm.nih.gov/books/NBK25499/
- PubMed Central: https://www.ncbi.nlm.nih.gov/pmc/
- Rate Limiting: https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requ
