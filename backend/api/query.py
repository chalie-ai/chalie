"""
Query API — Cognitive state query endpoints for external wrappers.

Lets wrappers query Chalie's cognitive state to make local decisions:
when to show notifications, how to present information, what context
is relevant.

Routes:
  GET  /api/query/situation          — current situation assessment
  GET  /api/query/relevance?q=<text> — relevance ranking for a topic
  GET  /api/query/world-state        — salience-scored world model items
  GET  /api/query/identity           — identity vectors + style metrics
  GET  /api/query/memory?q=<query>&k=5 — semantic memory search
  POST /api/query/composite          — multiple slices in one call

Permission model:
  - Bearer-authenticated requests require the query type in
    ``permissions.query`` via WrapperAuthService.check_permission().
  - Cookie-authenticated requests (chat UI) are always permitted.
"""

import json
import logging

from flask import Blueprint, g, jsonify, request

from .auth import require_auth

logger = logging.getLogger(__name__)

query_bp = Blueprint("query", __name__, url_prefix="/api/query")


# ---------------------------------------------------------------------------
# Lazy service factories — each returns the class/callable.
# Defined at module level so tests can patch them by name.
# ---------------------------------------------------------------------------

def _get_situation_model_service():
    """Return SituationModelService class (lazy, fail-open)."""
    from services.situation_model_service import SituationModelService
    return SituationModelService


def _get_world_state_service():
    """Return WorldStateService class (lazy, fail-open)."""
    from services.world_state_service import WorldStateService
    return WorldStateService


def _get_episodic_retrieval_service():
    """Return EpisodicRetrievalService class (lazy, fail-open)."""
    from services.episodic_retrieval_service import EpisodicRetrievalService
    return EpisodicRetrievalService


def _get_identity_service():
    """Return IdentityService class (lazy, fail-open)."""
    from services.identity_service import IdentityService
    return IdentityService


def _get_memory_client():
    """Return MemoryClientService class (lazy, fail-open)."""
    from services.memory_client import MemoryClientService
    return MemoryClientService


def _get_db_service():
    """Return the shared DatabaseService instance (lazy, fail-open)."""
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _get_wrapper_service():
    """Return a WrapperAuthService using the shared DatabaseService."""
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


def _check_query_permission(slice_name: str) -> bool:
    """Return True if the current request may query the given slice.

    Cookie-authenticated requests (g.wrapper_id is None) are always
    permitted.  Bearer-authenticated requests must have the slice listed
    in the wrapper's ``permissions.query`` array.

    Args:
        slice_name: The query slice name, e.g. ``"situation"``.

    Returns:
        True if the request is permitted, False otherwise.
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


def _permission_denied(slice_name: str):
    """Return a 403 response for a denied query slice.

    Args:
        slice_name: The slice that was denied.

    Returns:
        Flask response tuple (json, 403).
    """
    return jsonify({"error": f"Not permitted to query '{slice_name}'"}), 403


# ---------------------------------------------------------------------------
# Slice handlers (each returns a plain dict, never a Flask response)
# ---------------------------------------------------------------------------

def _slice_situation() -> dict:
    """Collect current situation assessment from SituationModelService.

    Returns:
        dict with keys ``scores``, ``directive``, ``phase``.
        Falls back to empty defaults if the service is unavailable.
    """
    try:
        SituationModelService = _get_situation_model_service()
        sms = SituationModelService()
        state = sms.get_current()
        directive = sms.get_directive()

        # Flatten scores: return just the ``value`` float per dimension
        raw_scores = state.get("scores", {})
        scores = {k: v.get("value") for k, v in raw_scores.items() if isinstance(v, dict)}

        phase = state.get("phase", {})
        phase_out = {
            "current": phase.get("current"),
            "momentum": phase.get("momentum"),
        }

        return {
            "scores": scores,
            "directive": directive,
            "phase": phase_out,
        }
    except Exception as e:
        logger.debug("[Query API] situation slice failed: %s", e)
        return {"scores": {}, "directive": None, "phase": {}}


def _slice_relevance(query: str) -> dict:
    """Score relevance of a text snippet against the current cognitive context.

    Uses EpisodicRetrievalService for semantic similarity and the cached
    world model for goal/task matching.

    Args:
        query: Text to score relevance for.

    Returns:
        dict with keys ``relevance``, ``related_goals``, ``related_traits``,
        ``recommendation``.  Falls back to neutral defaults on any error.
    """
    if not query or not query.strip():
        return {
            "relevance": 0.0,
            "related_goals": [],
            "related_traits": [],
            "recommendation": "no_query",
        }

    relevance = 0.0
    related_goals: list = []
    related_traits: list = []

    # Semantic similarity via episodic retrieval
    try:
        EpisodicRetrievalService = _get_episodic_retrieval_service()
        db = _get_db_service()
        svc = EpisodicRetrievalService(db)
        episodes = svc.retrieve_episodes(query_text=query, limit=3)

        if episodes:
            top = episodes[0]
            raw_score = top.get("composite_score") or top.get("score") or 0.0
            relevance = max(0.0, min(1.0, float(raw_score)))
    except Exception as e:
        logger.debug("[Query API] relevance episodic lookup failed: %s", e)

    # Goal matching via world model cache
    try:
        MemoryClientService = _get_memory_client()
        store = MemoryClientService.create_connection()
        raw = store.get("world_model:items")
        if raw:
            payload = json.loads(raw)
            for task in payload.get("persistent_tasks", []):
                goal = task.get("goal", "")
                if goal and query.lower() in goal.lower():
                    related_goals.append(goal[:120])
    except Exception as e:
        logger.debug("[Query API] relevance world model lookup failed: %s", e)

    # Derive recommendation
    if relevance >= 0.7:
        recommendation = "surface_now"
    elif relevance >= 0.4:
        recommendation = "surface_soon"
    else:
        recommendation = "defer"

    return {
        "relevance": round(relevance, 3),
        "related_goals": related_goals[:5],
        "related_traits": related_traits,
        "recommendation": recommendation,
    }


def _slice_world_state(query: str = "") -> dict:
    """Return salience-scored world model items.

    Delegates to WorldStateService.get_world_model_summary() for the
    structured cache read.

    Args:
        query: Optional topic hint (reserved for future semantic filtering).

    Returns:
        dict with key ``items`` — list of dicts with ``type`` and ``label``.
    """
    try:
        WorldStateService = _get_world_state_service()
        svc = WorldStateService()
        summary = svc.get_world_model_summary()

        items = []
        for label in summary.get("scheduled", []):
            items.append({"type": "scheduled", "label": label})
        for label in summary.get("tasks", []):
            items.append({"type": "task", "label": label})
        for label in summary.get("lists", []):
            items.append({"type": "list", "label": label})
        for label in summary.get("ambient", []):
            items.append({"type": "ambient", "label": label})
        for label in summary.get("topics", []):
            items.append({"type": "topic", "label": label})

        return {"items": items}
    except Exception as e:
        logger.debug("[Query API] world-state slice failed: %s", e)
        return {"items": []}


def _slice_identity() -> dict:
    """Return identity vectors and current style metrics.

    Delegates to IdentityService (vectors) and reads last-measured style
    metrics from MemoryStore key ``style_metrics:last``.

    Returns:
        dict with keys ``vectors`` (identity dimensions keyed by name, each
        with ``baseline`` and ``activation`` floats) and ``style`` (measured
        style metrics dict, empty when not yet recorded).
    """
    vectors: dict = {}
    style: dict = {}

    # Identity vectors
    try:
        IdentityService = _get_identity_service()
        db = _get_db_service()
        svc = IdentityService(db)
        raw_vectors = svc.get_vectors()

        for name, v in raw_vectors.items():
            vectors[name] = {
                "baseline": v.get("baseline_weight"),
                "activation": v.get("current_activation"),
            }
    except Exception as e:
        logger.debug("[Query API] identity vectors failed: %s", e)

    # Style metrics — read last measured value from MemoryStore
    try:
        MemoryClientService = _get_memory_client()
        store = MemoryClientService.create_connection()
        raw = store.get("style_metrics:last")
        if raw:
            style = json.loads(raw)
    except Exception as e:
        logger.debug("[Query API] style metrics lookup failed: %s", e)

    return {"vectors": vectors, "style": style}


def _slice_memory(query: str, k: int) -> dict:
    """Search episodic memory semantically.

    Args:
        query: Search text.
        k: Maximum number of results (clamped to 1–20).

    Returns:
        dict with key ``results`` — list of episode dicts with ``id``,
        ``summary``, ``score``, and ``created_at``.
    """
    if not query or not query.strip():
        return {"results": []}

    try:
        EpisodicRetrievalService = _get_episodic_retrieval_service()
        db = _get_db_service()
        svc = EpisodicRetrievalService(db)
        episodes = svc.retrieve_episodes(query_text=query, limit=max(1, min(k, 20)))

        results = []
        for ep in episodes:
            results.append({
                "id": ep.get("id"),
                "summary": ep.get("summary") or ep.get("content") or "",
                "score": round(
                    float(ep.get("composite_score") or ep.get("score") or 0.0), 4
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

def _dispatch_slice(slice_name: str) -> dict:
    """Dispatch a named slice and return its payload dict.

    Handles parameterised slices like ``"relevance:auth tests"`` by splitting
    on the first colon.

    Args:
        slice_name: Slice name, optionally with a colon-separated parameter.

    Returns:
        The slice payload dict, or ``{"error": "Unknown slice: <name>"}`` for
        unknown names.
    """
    # Split parameterised slices, e.g. "relevance:auth tests" → name="relevance", param="auth tests"
    name, _, param = slice_name.partition(":")
    name = name.strip().lower()

    if name == "situation":
        return _slice_situation()
    elif name == "relevance":
        return _slice_relevance(param.strip())
    elif name in ("world-state", "world_state"):
        return _slice_world_state(param.strip())
    elif name == "identity":
        return _slice_identity()
    elif name == "memory":
        parts = param.split("&k=", 1)
        q = parts[0].strip()
        k = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 5
        return _slice_memory(q, k)
    else:
        return {"error": f"Unknown slice: {slice_name!r}"}


# ---------------------------------------------------------------------------
# GET /api/query/situation
# ---------------------------------------------------------------------------

@query_bp.route("/situation", methods=["GET"])
@require_auth
def get_situation():
    """Return the current situation assessment.

    Response JSON:
        scores (dict): Composite scores — float per dimension.
        directive (str|None): Plain-language behavioral directive.
        phase (dict): Conversation phase with ``current`` and ``momentum``.

    Returns:
        200 with situation dict.
        403 if the bearer token does not include ``"situation"`` in query permissions.
    """
    if not _check_query_permission("situation"):
        return _permission_denied("situation")

    return jsonify(_slice_situation()), 200


# ---------------------------------------------------------------------------
# GET /api/query/relevance
# ---------------------------------------------------------------------------

@query_bp.route("/relevance", methods=["GET"])
@require_auth
def get_relevance():
    """Score the relevance of a text query against the current cognitive context.

    Query parameters:
        q (str): The text to score.

    Response JSON:
        relevance (float): 0.0–1.0 relevance score.
        related_goals (list[str]): Matching goals from the world model.
        related_traits (list): Matching user traits (reserved; always empty for now).
        recommendation (str): ``"surface_now"``, ``"surface_soon"``,
            ``"defer"``, or ``"no_query"``.

    Returns:
        200 with relevance dict.
        403 if bearer token lacks ``"relevance"`` query permission.
    """
    if not _check_query_permission("relevance"):
        return _permission_denied("relevance")

    q = (request.args.get("q") or "").strip()
    return jsonify(_slice_relevance(q)), 200


# ---------------------------------------------------------------------------
# GET /api/query/world-state
# ---------------------------------------------------------------------------

@query_bp.route("/world-state", methods=["GET"])
@require_auth
def get_world_state():
    """Return salience-scored world model items.

    Query parameters:
        q (str, optional): Topic hint for future semantic filtering.

    Response JSON:
        items (list[dict]): Each item has ``type`` and ``label`` fields.

    Returns:
        200 with world-state dict.
        403 if bearer token lacks ``"world-state"`` query permission.
    """
    if not _check_query_permission("world-state"):
        return _permission_denied("world-state")

    q = (request.args.get("q") or "").strip()
    return jsonify(_slice_world_state(q)), 200


# ---------------------------------------------------------------------------
# GET /api/query/identity
# ---------------------------------------------------------------------------

@query_bp.route("/identity", methods=["GET"])
@require_auth
def get_identity():
    """Return identity vectors and current style metrics.

    Response JSON:
        vectors (dict): Identity dimensions with ``baseline`` and ``activation``
            per entry.
        style (dict): Most recently measured style metrics (may be empty).

    Returns:
        200 with identity dict.
        403 if bearer token lacks ``"identity"`` query permission.
    """
    if not _check_query_permission("identity"):
        return _permission_denied("identity")

    return jsonify(_slice_identity()), 200


# ---------------------------------------------------------------------------
# GET /api/query/memory
# ---------------------------------------------------------------------------

@query_bp.route("/memory", methods=["GET"])
@require_auth
def get_memory():
    """Search episodic memory semantically.

    Query parameters:
        q (str): Search text.
        k (int, optional): Maximum results (1–20, default 5).

    Response JSON:
        results (list[dict]): Each result has ``id``, ``summary``, ``score``,
            and ``created_at`` fields.

    Returns:
        200 with memory dict.
        403 if bearer token lacks ``"memory"`` query permission.
    """
    if not _check_query_permission("memory"):
        return _permission_denied("memory")

    q = (request.args.get("q") or "").strip()
    try:
        k = int(request.args.get("k", 5))
        k = max(1, min(k, 20))
    except (TypeError, ValueError):
        k = 5

    return jsonify(_slice_memory(q, k)), 200


# ---------------------------------------------------------------------------
# POST /api/query/composite
# ---------------------------------------------------------------------------

@query_bp.route("/composite", methods=["POST"])
@require_auth
def get_composite():
    """Request multiple cognitive state slices in a single call.

    Body (JSON):
        slices (list[str]): Named slices to include.  Each element is either
            a plain name (e.g. ``"situation"``) or a parameterised name
            (e.g. ``"relevance:auth tests"``).

    Supported slice names:
        ``situation``, ``relevance``, ``world-state``, ``identity``, ``memory``,
        ``world_state`` (alias for ``world-state``).

    Permission model:
        Each slice is checked independently against the bearer's query
        permissions.  Denied slices are listed in the ``denied`` array.
        Cookie-authenticated requests are always fully permitted.

    Response JSON:
        results (dict): Slice name → payload dict for permitted slices.
        denied (list[str]): Slice names that were denied by permissions.
        errors (list[str]): Slice names that failed at the service layer.

    Returns:
        200 with composite dict (even if all slices are denied).
        400 if the body is missing or ``slices`` is not a list.
    """
    body = request.get_json(silent=True) or {}
    slices = body.get("slices")
    if slices is None or not isinstance(slices, list):
        return jsonify({"error": "'slices' must be a JSON array"}), 400

    results: dict = {}
    denied: list = []
    errors: list = []

    for raw_slice in slices:
        if not isinstance(raw_slice, str) or not raw_slice.strip():
            continue

        # Derive the base name for permission checking (before any ":" param)
        base_name = raw_slice.partition(":")[0].strip().lower()
        # Normalise world_state alias for permission key
        perm_key = "world-state" if base_name == "world_state" else base_name

        if not _check_query_permission(perm_key):
            denied.append(raw_slice)
            continue

        try:
            payload = _dispatch_slice(raw_slice)
            results[raw_slice] = payload
        except Exception as e:
            logger.warning("[Query API] composite slice '%s' failed: %s", raw_slice, e)
            errors.append(raw_slice)

    return jsonify({"results": results, "denied": denied, "errors": errors}), 200
