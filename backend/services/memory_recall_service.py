"""Memory recall — composes Graph FTS (subject) + Map vector (cues) retrieval
into one recall result.

The two raw primitives live on the models (model-owns-SQL rule):
``MemoryGraphRow.fts_subject_search`` (FTS5 over ``subject``) and
``MemoryMapRow.vec_knn`` (vec0 KNN over cues). This service owns keyword
extraction, the query embedding, searchable-pool restriction, ranking, and
the fused result shape.

Design: "keyword extraction -> FTS5 over graph subject (top 3) plus whole-prompt
vector over the map's situational cues (top 3, most-consolidated first)".
Matching the situation the user is in against the situations a memory is about
is what makes a consolidated episode surface at the moment it helps — matching
two narratives never did. Cues pick the candidates; ``iteration`` orders them,
so the memory that absorbed the most history speaks before any single moment.
A recall miss returns empty lists; an oversized query (more than 500 characters
after stop-word removal) raises ValueError — callers surface it as their native
error, never a silent trim.
"""
from __future__ import annotations

import logging
import re

from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

#: A query whose stop-word-scrubbed form exceeds this many characters is
#: rejected outright — a recall of a 2000-character "query" is the model
#: dumping a paragraph instead of asking, and the FTS/vector lanes are not
#: built to be meaningful at that size. Rejection, never a silent trim.
_MAX_QUERY_CHARS = 500

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


def scrubbed_query(query: str) -> str:
    """The query after stop-word removal, single-space joined — the exact
    string the length gate measures, so callers (e.g. the ``recall`` ability)
    always agree with the service on what counts toward the cap."""
    return " ".join(_extract_keywords(query))


def _fts_query(keywords: list[str]) -> str:
    # FTS5: OR terms so any matching subject surfaces (lenient recall).
    return " OR ".join(keywords)


class MemoryRecallService:
    """Fuses Graph (living facts) + Map (episodic lineage) recall."""

    def recall(
        self, query: str, *, k_graph: int = 3, k_map: int = 3
    ) -> dict[str, list[dict[str, object]]]:
        """Top ``k_graph`` Graph facts (FTS over subject) plus top ``k_map`` Map
        episodes (vector over cues, most-consolidated first, retired rows
        excluded). Empty lists on a miss; a query whose stop-word-scrubbed
        form exceeds ``_MAX_QUERY_CHARS`` raises ValueError before any
        retrieval — the query is rejected, never trimmed."""
        scrubbed = scrubbed_query(query)
        if len(scrubbed) > _MAX_QUERY_CHARS:
            raise ValueError(
                "Recall query is too long, maximum of 500 characters allowed. "
                "Use more fine-grained queries to load relevent memories"
            )
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
        # the searchable pool and re-rank by iteration DESC, distance ASC. The
        # cues decide WHICH episodes are candidates; iteration decides which of
        # them speaks first, because a consolidated memory carries everything its
        # parents said and outranks any single moment it was distilled from.
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
                "source": row.source,
                "contents": row.contents,
                "iteration": row.iteration,
                "generated_at": row.generated_at,
            }
            for _, _, row in ranked[:k]
        ]
