from typing import Any, Dict, List, Optional


class FallbackMedicalVectorStore:
    """Lightweight in-memory retrieval store used when ChromaDB stack is unavailable."""

    def __init__(self) -> None:
        self._docs: List[Dict[str, Any]] = []

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        for doc in documents:
            text = str(doc.get("text", "")).strip()
            if not text:
                continue
            metadata = doc.get("metadata") or {}
            self._docs.append(
                {
                    "id": doc.get("id", f"doc_{len(self._docs)+1}"),
                    "text": text,
                    "url": metadata.get("url", ""),
                    "pmid": metadata.get("pmid", ""),
                    "source": metadata.get("source", "Synthetic fallback"),
                    "domain": metadata.get("domain", "general"),
                }
            )

    def count(self) -> int:
        return len(self._docs)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        tokens = {t for t in query_text.lower().replace("_", " ").split() if len(t) > 2}
        candidates = self._docs
        if domain_filter:
            candidates = [d for d in candidates if d.get("domain") == domain_filter]

        scored: List[tuple[float, Dict[str, Any]]] = []
        for doc in candidates:
            text_tokens = set(str(doc.get("text", "")).lower().split())
            overlap = len(tokens.intersection(text_tokens))
            score = overlap / max(1, len(tokens))
            scored.append((score, doc))

        if not scored:
            return []

        scored.sort(key=lambda it: it[0], reverse=True)
        top = scored[:n_results]
        return [
            {
                "id": d["id"],
                "url": d.get("url", ""),
                "pmid": d.get("pmid", ""),
                "text": d["text"],
                "source": d.get("source", "Synthetic fallback"),
                "domain": d.get("domain", "general"),
                "distance": round(1.0 - s, 4),
            }
            for s, d in top
        ]
