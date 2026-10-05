import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rag.vector_store import MedicalVectorStore
if __name__=='__main__':
    docs=json.loads(Path('data/processed/pubmed_verified.json').read_text())
    store=MedicalVectorStore()
    if store.count(): raise SystemExit('Index already contains documents; use a fresh CHROMA_PERSIST_DIR to avoid duplicates.')
    store.ingest_documents(docs)
    print(json.dumps({'indexed':store.count(),'embedding':store._embed_model_name}))
