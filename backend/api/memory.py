"""Memory namespace — read-only search across episodic memory + the data graph.

A single retrieval route fans the query out to two independent stores, merges the
heterogeneous rows into ranked :class:`MemoryHit` results, and returns partial
results when one store fails (intended resilience, not a shim). Validation and
serialization run through the DTO boundary decorators; no service layer is
touched by this route.
"""

from __future__ import annotations

import logging
from typing import cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from models.episode import Episode
from services.time_utils import parse_utc
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.memory import MemoryHit, MemorySearchQuery, MemorySearchResponse

logger = logging.getLogger(__name__)

_ERR_SEARCH_FAILED = "Failed to search memory"

# Kinds recalled from the data graph alongside episodic memory (user facts + system facts).
_RECALL_KINDS = ["user_specific", "system"]
_RECALL_LIMIT = 5

memory_ns = Namespace("memory", description="Memory search", path="/api/memory")

register_dto(memory_ns, MemorySearchQuery, MemoryHit, MemorySearchResponse, Error)

_M = memory_ns.models


def _error(message: str, status: int) -> ResponseReturnValue:
    """Build a uniform non-2xx ``Error`` body carrying its own status code."""
    return Error(error=message).model_dump(mode="json"), status


def _episode_hit(ep: Episode) -> MemoryHit:
    """Build a ``MemoryHit`` from an episodic-retrieval :class:`Episode`."""
    return MemoryHit(
        type="episode",
        content=ep.gist,
        score=getattr(ep, "composite_score", 0.0),
        created_at=parse_utc(ep.created_at or ""),
    )


def _concept_hit(row: dict[str, object]) -> MemoryHit:
    """Build a ``MemoryHit`` from a data-graph concept row."""
    return MemoryHit(
        type="concept",
        content=f"{cast(str, row.get('key', ''))}: {cast(str, row.get('value', ''))}",
        score=cast(float, row.get("composite_score", row.get("retrieval_weight", 0))),
        confidence=cast(float, row.get("retrieval_weight", 0)),
    )


@memory_ns.route("/search")
class MemorySearchResource(Resource):
    @require_session
    @memory_ns.response(200, "Search results", model=_M["MemorySearchResponse"])
    @memory_ns.response(422, "Validation failed", model=_M["Error"])
    @memory_ns.response(500, _ERR_SEARCH_FAILED, model=_M["Error"])
    @responds(MemorySearchResponse, code=200)
    @expects(MemorySearchQuery, source="args")
    def get(self, dto: MemorySearchQuery) -> MemorySearchResponse | ResponseReturnValue:
        """Search episodic memory + the data graph and return ranked, merged hits."""
        results: list[MemoryHit] = []
        try:
            try:
                from services.episodic_service import EpisodicService

                results.extend(
                    _episode_hit(ep)
                    for ep in cast(
                        "list[Episode]",
                        EpisodicService().retrieve(query_text=dto.q, channel=None),
                    )
                )
            except Exception as exc:
                logger.warning("[Memory] Episode search failed: %s", exc)

            try:
                from services.memory_recall_service import recall
                results.extend(
                    _concept_hit(c)
                    for c in recall(dto.q, kinds=_RECALL_KINDS, limit=_RECALL_LIMIT)
                )
            except Exception as exc:
                logger.warning("[Memory] Data graph search failed: %s", exc)

            results.sort(key=lambda hit: hit.score, reverse=True)
            return MemorySearchResponse(results=results)
        except Exception as exc:
            logger.exception("[REST API] memory/search error: %s", exc)
            return _error(_ERR_SEARCH_FAILED, 500)
