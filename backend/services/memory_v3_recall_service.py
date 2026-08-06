"""Memory v3 recall service — composes Graph FTS (subject) + Map vector
(contents) retrieval into one recall result.

The two raw primitives live on the models (model-owns-SQL rule):
``MemoryGraphRow.fts_subject_search`` (FTS5 over ``subject``) and
``MemoryMapRow.vec_knn`` (vec0 KNN over contents). This service owns keyword
extraction, the query embedding, searchable-pool restriction, iteration
ranking, and the fused result shape.

Per the Memory v3 design: "keyword extraction -> FTS5 over graph subject (top 3)
plus whole-prompt vector over the map (top 3, higher iteration ranks higher)".
Never raises — a recall miss returns empty lists, never breaks a turn.
"""
from __future__ import annotations

import logging
import re

from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
        "for", "with", "is", "are", "was", "were", "be", "been", "i", "you", "he",
        "she", "it", "we", "they", "my", "your", "his", "her", "its", "our",
        "their", "me", "him", "them", "this", "that", "do", "does", "did",
        "have", "has", "had", "what", "where", "when", "who", "why", "how",
    }
)


def _extract_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _fts_query(keywords: list[str]) -> str:
    # FTS5: OR terms so any matching subject surfaces (lenient recall).
    return " OR ".join(keywords)


class MemoryV3RecallService:
    """Fuses Graph (living facts) + Map (episodic lineage) recall."""

    def recall(
        self, query: str, *, k_graph: int = 3, k_map: int = 3
    ) -> dict[str, list[dict[str, object]]]:
        """Top ``k_graph`` Graph facts (FTS over subject) plus top ``k_map`` Map
        episodes (vector over contents, iteration-ranked, retired rows excluded).
        Empty lists on miss; never raises."""
        return {
            "graph": self._recall_graph(query, k_graph),
            "map": self._recall_map(query, k_map),
        }

    def _recall_graph(self, query: str, k: int) -> list[dict[str, object]]:
        terms = _fts_query(_extract_keywords(query))
        if not terms:
            return []
        rows = MemoryGraphRow.fts_subject_search(terms, k)
        return [
            {
                "subject": row.subject,
                "contents": row.contents,
                "last_updated_at": row.last_updated_at,
            }
            for row in rows
        ]

    def _recall_map(self, query: str, k: int) -> list[dict[str, object]]:
        pool_ids = {
            row.id for row in MemoryMapRow.searchable_pool() if row.id is not None
        }
        if not pool_ids:
            return []
        embedding = get_embedding_service().generate_embedding(query)
        if not embedding:
            return []
        # KNN over the whole vec table with a generous window, then restrict to
        # the searchable pool and re-rank by iteration DESC, distance ASC.
        knn = MemoryMapRow.vec_knn(embedding, k=max(k * 10, 20))
        ranked: list[tuple[int, float, MemoryMapRow]] = []
        for rowid, distance in knn.items():
            if rowid not in pool_ids:
                continue
            row = MemoryMapRow.by_id(rowid)
            if row is not None:
                ranked.append((row.iteration, distance, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "contents": row.contents,
                "iteration": row.iteration,
                "generated_at": row.generated_at,
            }
            for _, _, row in ranked[:k]
        ]
