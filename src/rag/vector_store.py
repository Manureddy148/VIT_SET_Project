import os
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Doc-specified embedding: FremyCompany/BioLORD-2023
# Fallback: all-MiniLM-L6-v2 (fast, always available)
_DEFAULT_EMBED_MODEL = os.getenv(
    "MEDICAL_AI_EMBED_MODEL", "FremyCompany/BioLORD-2023"
)
_FALLBACK_EMBED_MODEL = "all-MiniLM-L6-v2"
# MMR diversity parameter (0 = pure similarity, 1 = maximum diversity)
_MMR_DIVERSITY = float(os.getenv("MEDICAL_AI_MMR_DIVERSITY", "0.3"))


def _mmr_select(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_data: List[Dict[str, Any]],
    k: int,
    diversity: float,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance selection (doc-specified algorithm).

    Balances relevance to the query with diversity among selected documents.
    diversity=0 → pure similarity; diversity=1 → maximum spread.
    """
    if len(candidate_data) <= k:
        return candidate_data

    # cosine similarity helpers
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

    query_sims = np.array([_cos(query_emb, e) for e in candidate_embs])
    selected_indices: List[int] = []
    remaining = list(range(len(candidate_data)))

    for _ in range(k):
        if not remaining:
            break
        if not selected_indices:
            # First pick: highest similarity to query
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            sel_embs = candidate_embs[selected_indices]
            scores = []
            for i in remaining:
                rel = query_sims[i]
                red = max(_cos(candidate_embs[i], s) for s in sel_embs)
                scores.append((1 - diversity) * rel - diversity * red)
            best = remaining[int(np.argmax(scores))]
        selected_indices.append(best)
        remaining.remove(best)

    return [candidate_data[i] for i in selected_indices]


class MedicalVectorStore:
    """ChromaDB-backed vector store (Layer 5).

    Embedding: BioLORD-2023 (medical-domain sentence embeddings, per docs).
    Retrieval:  MMR (Maximal Marginal Relevance) for diversified top-K results.
    """

    COLLECTION_NAME = "medical_literature"

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        # Try BioLORD-2023 first; fall back to MiniLM if unavailable
        try:
            self._embed_model = SentenceTransformer(_DEFAULT_EMBED_MODEL)
            self._embed_model_name = _DEFAULT_EMBED_MODEL
        except Exception:
            self._embed_model = SentenceTransformer(_FALLBACK_EMBED_MODEL)
            self._embed_model_name = _FALLBACK_EMBED_MODEL

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
        use_mmr: bool = True,
        mmr_diversity: float = _MMR_DIVERSITY,
    ) -> List[Dict[str, Any]]:
        """Query with optional MMR diversification (doc-specified, diversity=0.3)."""
        query_embedding = self._embed_model.encode(query_text)
        where_filter = {"domain": domain_filter} if domain_filter else None

        # Fetch more candidates for MMR (3× to allow diversity selection)
        fetch_k = n_results * 3 if use_mmr else n_results
        fetch_k = max(fetch_k, n_results)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        docs: List[Dict[str, Any]] = []
        candidate_embs_list = []
        for text, meta, dist, emb in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["embeddings"][0],
        ):
            docs.append(
                {
                    "url": meta.get("url", ""),
                    "pmid": meta.get("pmid", ""),
                    "text": text,
                    "source": meta.get("source", "Unknown"),
                    "domain": meta.get("domain", "general"),
                    "distance": round(float(dist), 4),
                }
            )
            candidate_embs_list.append(np.array(emb, dtype=np.float32))

        if not docs:
            return []

        if use_mmr and len(docs) > n_results:
            candidate_embs = np.array(candidate_embs_list, dtype=np.float32)
            query_emb = query_embedding.astype(np.float32)
            docs = _mmr_select(query_emb, candidate_embs, docs, n_results, mmr_diversity)
        else:
            docs = docs[:n_results]

        return docs

    def count(self) -> int:
        return int(self._collection.count())
