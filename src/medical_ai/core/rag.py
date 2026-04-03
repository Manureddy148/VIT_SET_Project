from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EvidenceDoc:
    id: str
    source: str
    title: str
    domain_tags: List[str]
    text: str


@dataclass(frozen=True)
class EvidenceHit:
    doc_id: str
    source: str
    title: str
    score: float
    snippet: str


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    parts = [p for p in text.split() if len(p) >= 3]
    return parts


class LocalKnowledgeBase:
    """
    Lightweight, dependency-free retrieval over a small curated corpus.
    Uses simple token overlap scoring (good enough to keep E2E working offline).
    """

    def __init__(self, docs: Sequence[EvidenceDoc]):
        self.docs = list(docs)
        self._doc_tokens: Dict[str, set[str]] = {d.id: set(_tokenize(d.text)) for d in self.docs}

    @classmethod
    def from_json(cls, path: str) -> "LocalKnowledgeBase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = []
        for d in raw.get("documents", []) or []:
            docs.append(
                EvidenceDoc(
                    id=str(d["id"]),
                    source=str(d.get("source", "Unknown")),
                    title=str(d.get("title", d["id"])),
                    domain_tags=list(d.get("domain_tags", []) or []),
                    text=str(d.get("text", "")),
                )
            )
        return cls(docs)

    def query(
        self,
        query_text: str,
        domain: Optional[str] = None,
        k: int = 3,
    ) -> List[EvidenceHit]:
        q_tokens = set(_tokenize(query_text))
        if not q_tokens:
            return []

        scored: List[Tuple[float, EvidenceDoc]] = []
        for d in self.docs:
            if domain and domain not in d.domain_tags:
                continue
            overlap = len(q_tokens & self._doc_tokens.get(d.id, set()))
            if overlap == 0:
                continue
            # normalize by query length
            score = overlap / max(1, len(q_tokens))
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits: List[EvidenceHit] = []
        for score, d in scored[:k]:
            hits.append(
                EvidenceHit(
                    doc_id=d.id,
                    source=d.source,
                    title=d.title,
                    score=round(float(score), 3),
                    snippet=(d.text[:220] + ("..." if len(d.text) > 220 else "")),
                )
            )
        return hits

