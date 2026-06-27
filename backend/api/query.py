"""
Query API — Cognitive state query endpoints for external wrappers.

Lets wrappers query Chalie's cognitive state to make local decisions:
when to show notifications, how to present information, what context
is relevant.

Routes:
  GET  /api/query/situation          — current situation assessment
  GET  /api/query/relevance?q=<text> — relevance ranking for a topic
  GET  /api/query/memory?q=<query>&k=5 — semantic memory search
  POST /api/query/composite          — multiple slices in one call

Permission model:
  - Bearer-authenticated requests require the query type in
    ``permissions.query`` via WrapperAuthService.check_permission().
  - Cookie-authenticated requests (chat UI) are always permitted.
"""

import logging
from typing import TYPE_CHECKING, cast

from flask import g, request
from flask_restx import Namespace, Resource

from .auth import require_auth
from services.log_utils import safe

if TYPE_CHECKING:
    from typing import Protocol
    from services.wrapper_auth_service import WrapperAuthService

    class _ERS(Protocol):
        def retrieve(self, query_text: str, channel: "str | None") -> "list[dict[str, object]]": ...

logger = logging.getLogger(__name__)

query_bp = Namespace("query", description="Cognitive state query endpoints", path="/api/query")


# ---------------------------------------------------------------------------
# Lazy service factories — each returns the class/callable.
# Defined at module level so tests can patch them by name.
# ---------------------------------------------------------------------------

def _get_retrieval_module() -> object:
    from services import episodic_retrieval_service
    return episodic_retrieval_service


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _get_wrapper_service() -> "WrapperAuthService":
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


def _check_query_permission(slice_name: str) -> bool:
    """Cookie-authenticated requests are always permitted. Bearer-authenticated
    requests must have the slice listed in the wrapper's ``permissions.query`` array.
    """
    wrapper_id = getattr(g, "wrapper_id", None)
    if wrapper_id is None:
        # Cookie auth — full access
        return True

    try:
        svc = _get_wrapper_service()
        return svc.check_permission(wrapper_id, "query", slice_name)
    except Exception as e:
        logger.debug("[Query API] Permission check failed: %s", e)
        return False


def _permission_denied(slice_name: str) -> tuple[dict[str, object], int]:
    return {"error": f"Not permitted to query '{slice_name}'"}, 403


# ---------------------------------------------------------------------------
# Slice handlers (each returns a plain dict, never a Flask response)
# ---------------------------------------------------------------------------

def _slice_relevance(query: str) -> "dict[str, object]":
    """Uses episodic_retrieval_service for semantic similarity.
    Falls back to neutral defaults on any error.
    """
    if not query or not query.strip():
        return {
            "relevance": 0.0,
            "related_traits": [],
            "recommendation": "no_query",
        }

    relevance = 0.0
    related_traits: list[object] = []

    # Semantic similarity via episodic retrieval
    try:
        episodes = cast("_ERS", _get_retrieval_module()).retrieve(query_text=query, channel=None)

        if episodes:
            top = episodes[0]
            raw_score = float(cast(float, top.get("composite_score") or top.get("score") or 0.0))
            # Sigmoid normalization: midpoint ~50, steepness 0.05
            # Maps composite scores (typical range 0-150) to [0,1]
            import math
            relevance = 1.0 / (1.0 + math.exp(-0.05 * (raw_score - 50)))
    except Exception as e:
        logger.debug("[Query API] relevance episodic lookup failed: %s", e)

    # Derive recommendation
    if relevance >= 0.7:
        recommendation = "surface_now"
    elif relevance >= 0.4:
        recommendation = "surface_soon"
    else:
        recommendation = "defer"

    return {
        "relevance": round(relevance, 3),
        "related_traits": related_traits,
        "recommendation": recommendation,
    }


def _slice_memory(query: str, _k: int) -> "dict[str, object]":
    """Returns dict with key ``results`` — list of episode dicts with ``id``,
    ``summary``, ``score``, and ``created_at``.
    """
    if not query or not query.strip():
        return {"results": []}

    try:
        episodes = cast("_ERS", _get_retrieval_module()).retrieve(
            query_text=query, channel=None
        )

        results: list[dict[str, object]] = []
        for ep in episodes:
            results.append({
                "id": ep.get("id"),
                "summary": ep.get("summary") or ep.get("content") or "",
                "score": round(
                    float(cast(float, ep.get("composite_score") or ep.get("score") or 0.0)), 4
                ),
                "created_at": ep.get("created_at"),
            })

        return {"results": results}
    except Exception as e:
        logger.debug("[Query API] memory slice failed: %s", e)
        return {"results": []}


# ---------------------------------------------------------------------------
# Slice dispatch table (used by composite endpoint)
# ---------------------------------------------------------------------------

def _dispatch_slice(slice_name: str) -> "dict[str, object]":
    """Handles parameterised slices like ``"relevance:auth tests"`` by splitting
    on the first colon.
    """
    # Split parameterised slices, e.g. "relevance:auth tests" → name="relevance", param="auth tests"
    name, _, param = slice_name.partition(":")
    name = name.strip().lower()

    if name == "relevance":
        return _slice_relevance(param.strip())
    elif name == "memory":
        parts = param.split("&k=", 1)
        q = parts[0].strip()
        k = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 5
        return _slice_memory(q, k)
    else:
        return {"error": f"Unknown slice: {slice_name!r}"}


# ---------------------------------------------------------------------------
# GET /api/query/relevance
# ---------------------------------------------------------------------------

@query_bp.route("/relevance")
class RelevanceResource(Resource):
    @require_auth
    @query_bp.response(200, "OK")
    def get(self):
        """Returns 403 if bearer token lacks ``"relevance"`` query permission."""
        if not _check_query_permission("relevance"):
            return _permission_denied("relevance")

        q = (request.args.get("q") or "").strip()
        return _slice_relevance(q), 200


# ---------------------------------------------------------------------------
# GET /api/query/memory
# ---------------------------------------------------------------------------

@query_bp.route("/memory")
class MemoryResource(Resource):
    @require_auth
    @query_bp.response(200, "OK")
    def get(self):
        """Returns 403 if bearer token lacks ``"memory"`` query permission."""
        if not _check_query_permission("memory"):
            return _permission_denied("memory")

        q = (request.args.get("q") or "").strip()
        try:
            k = int(request.args.get("k", 5))
            k = max(1, min(k, 20))
        except (TypeError, ValueError):
            k = 5

        return _slice_memory(q, k), 200


# ---------------------------------------------------------------------------
# POST /api/query/composite
# ---------------------------------------------------------------------------

@query_bp.route("/composite")
class CompositeResource(Resource):
    @require_auth
    @query_bp.response(200, "OK")
    def post(self):
        """Each slice is checked independently against the bearer's query permissions.
        Denied slices are listed in the ``denied`` array. Cookie-authenticated
        requests are always fully permitted. Returns 200 even if all slices are denied.
        """
        body = request.get_json(silent=True) or {}
        slices = body.get("slices")
        if slices is None or not isinstance(slices, list):
            return {"error": "'slices' must be a JSON array"}, 400

        results: dict[str, object] = {}
        denied: list[str] = []
        errors: list[str] = []

        for raw_slice in slices:
            if not isinstance(raw_slice, str) or not raw_slice.strip():
                continue

            # Derive the base name for permission checking (before any ":" param)
            base_name = raw_slice.partition(":")[0].strip().lower()

            if not _check_query_permission(base_name):
                denied.append(raw_slice)
                continue

            try:
                payload = _dispatch_slice(raw_slice)
                results[raw_slice] = payload
            except Exception as e:
                logger.warning("[Query API] composite slice '%s' failed: %s", safe(raw_slice), e)
                errors.append(raw_slice)

        return {"results": results, "denied": denied, "errors": errors}, 200