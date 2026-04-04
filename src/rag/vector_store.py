import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class MedicalVectorStore:
    """ChromaDB-backed vector store (Layer 5)."""

    COLLECTION_NAME = "medical_literature"

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]
        embeddings = self._embed_model.encode(texts).tolist()
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self._embed_model.encode(query_text).tolist()
        where_filter = {"domain": domain_filter} if domain_filter else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(
                {
                    "text": text,
                    "source": meta.get("source", "Unknown"),
                    "domain": meta.get("domain", "general"),
                    "distance": round(float(dist), 4),
                }
            )
        return docs

    def count(self) -> int:
        return int(self._collection.count())
