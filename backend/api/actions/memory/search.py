"""Memory search action — read-only search across episodic memory + the data graph.

A single retrieval route fans the query out to two independent stores, merges the
heterogeneous rows into ranked hit DTOs, and returns partial results when one
store fails (intended resilience, not a shim). No service layer is touched
beyond the two stores queried here.
"""

from __future__ import annotations

import logging
from typing import cast

from flask import request
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, EndpointError
from api.response.memory import MemoryHitResponse
from models.episode import Episode
from services.episodic_service import EpisodicService
from services.memory_recall_service import recall
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# Kinds recalled from the data graph alongside episodic memory (user facts + system facts).
_RECALL_KINDS = ["user_specific", "system"]
_RECALL_LIMIT = 5


def _episode_hit(ep: Episode) -> MemoryHitResponse:
    """Build a hit DTO from an episodic-retrieval :class:`Episode`."""
    return MemoryHitResponse(
        type="episode",
        content=ep.gist,
        score=getattr(ep, "composite_score", 0.0),
        created_at=parse_utc(ep.created_at or ""),
    )


def _concept_hit(row: dict[str, object]) -> MemoryHitResponse:
    """Build a hit DTO from a data-graph concept row."""
    return MemoryHitResponse(
        type="concept",
        content=f"{cast(str, row.get('key', ''))}: {cast(str, row.get('value', ''))}",
        score=cast(float, row.get("composite_score", row.get("retrieval_weight", 0))),
        confidence=cast(float, row.get("retrieval_weight", 0)),
    )


class MemorySearch(Action):
    """Action searching episodic memory + the data graph and returning ranked, merged hits."""

    # id is ignored — there is no per-id resource here, only a query-driven
    # search, so this handler can never 404.
    response_dto = {"get": DocumentedResponse(MemoryHitResponse, listing=True, not_found=False)}

    def slug(self) -> str:
        return "memory"

    def verb(self) -> str:
        return "search"

    def get(self, id: int | str) -> ResponseReturnValue:
        """Search episodic memory + the data graph and return ranked, merged hits.

        Accepts a required `q` query parameter (non-empty after stripping
        whitespace), matching the legacy MemorySearchQuery DTO validation.
        """
        raw_q = request.args.get("q")
        if raw_q is None:
            raise EndpointError("q is required")
        q = raw_q.strip()
        if not q:
            raise EndpointError("q must not be empty")

        results: list[MemoryHitResponse] = []

        try:
            results.extend(
                _episode_hit(ep)
                for ep in cast(
                    "list[Episode]",
                    EpisodicService().retrieve(query_text=q, channel=None),
                )
            )
        except Exception as exc:
            logger.warning("[Memory] Episode search failed: %s", exc)

        try:
            results.extend(
                _concept_hit(c) for c in recall(q, kinds=_RECALL_KINDS, limit=_RECALL_LIMIT)
            )
        except Exception as exc:
            logger.warning("[Memory] Data graph search failed: %s", exc)

        results.sort(key=lambda hit: hit.score, reverse=True)
        total = len(results)
        return MemoryHitResponse.listing(results, page=1, limit=max(total, 1), total=total)
