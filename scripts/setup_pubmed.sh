#!/bin/bash
# Setup script for PubMed ingestion
# Downloads real abstracts from NCBI PubMed via Entrez API and seeds ChromaDB

set -e

echo "🔬 Medical AI — PubMed Ingestion Setup"
echo "======================================"
echo ""

# Check Python available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

echo "✓ Python found"
echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed
echo "✓ data/raw, data/processed created"
echo ""

# Ingest PubMed abstracts (online)
echo "📡 Fetching real abstracts from PubMed (NCBI Entrez API)..."
echo "   This will query for relevant articles on:"
echo "   - Diabetes severity & prediction"
echo "   - Heart disease severity & prediction"
echo "   - Pneumonia severity & prediction"
echo ""

python scripts/ingest_pubmed.py \
    --domains diabetes heart_disease pneumonia \
    --limit 30 \
    --output data/processed/pubmed_docs.json

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ PubMed ingestion completed"
    docs_count=$(python -c "import json; data=json.load(open('data/processed/pubmed_docs.json')); print(len(data))")
    echo "  Total documents: $docs_count"
else
    echo "⚠ PubMed ingestion failed (check internet/API limits)"
    echo "  Falling back to sample data on API startup"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start the API: python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
echo "  2. Or start the frontend: streamlit run frontend/app.py"
echo ""
echo "The ChromaDB vector store will auto-load these documents on first API startup."
