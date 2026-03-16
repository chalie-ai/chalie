"""
Signal Ingestion API — external wrappers push signals into the reasoning loop.

Routes:
  POST /api/signals          — ingest a single signal
  POST /api/signals/batch    — ingest up to 50 signals in one request

Authentication:
  Cookie session (chat UI):  wrapper_id is set to ``'__chat_ui__'``
  Bearer token (wrappers):   wrapper_id comes from ``g.wrapper_id``; the
                             signal_type must appear in the wrapper's declared
                             ``capabilities.signals`` list.

Rate limiting:
  100 signals/minute per wrapper_id via :class:`WrapperRateLimiter`.

Signal schema (per item):
  signal_type       str, required
  content           str, required
  source            str, optional (defaults to wrapper_id or '__chat_ui__')
  topic             str | null, optional
  activation_energy float 0–1, optional (default 0.5)
  metadata          dict | null, optional
"""

import logging
import uuid

from flask import Blueprint, g, jsonify, request

from .auth import require_auth

logger = logging.getLogger(__name__)

signals_bp = Blueprint("signals", __name__, url_prefix="/api/signals")

# Maximum signals accepted in a single batch request
_BATCH_MAX = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rate_limiter():
    """Return the process-singleton WrapperRateLimiter."""
    from services.wrapper_rate_limiter import WrapperRateLimiter
    return WrapperRateLimiter()


def _get_wrapper_service():
    """Return a WrapperAuthService using the shared DatabaseService."""
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


def _effective_wrapper_id() -> str:
    """Return the caller's wrapper_id, falling back to '__chat_ui__' for cookie auth."""
    wid = getattr(g, "wrapper_id", None)
    return wid if wid else "__chat_ui__"


def _check_signal_capability(wrapper_id: str, signal_type: str) -> bool:
    """Return True if the wrapper may emit ``signal_type``.

    Cookie-authenticated callers (wrapper_id == '__chat_ui__') are always
    allowed — they are the native UI, not an external wrapper.

    Bearer-authenticated callers must have ``signal_type`` listed in
    ``capabilities.signals``.  An absent or empty ``signals`` list means the
    wrapper may not emit *any* signals.

    Args:
        wrapper_id: The resolved wrapper identifier.
        signal_type: The signal type string the caller wants to emit.

    Returns:
        ``True`` if the caller is permitted, ``False`` otherwise.
    """
    if wrapper_id == "__chat_ui__":
        return True

    svc = _get_wrapper_service()
    wrapper = svc.get_wrapper(wrapper_id)
    if wrapper is None:
        return False

    allowed_signals = wrapper.get("capabilities", {}).get("signals", [])
    return signal_type in allowed_signals


def _validate_signal(body: dict) -> tuple[dict | None, str | None]:
    """Validate a single signal payload.

    Args:
        body: Raw dict from the request JSON.

    Returns:
        ``(cleaned_dict, None)`` on success, or ``(None, error_message)`` on
        failure.  The cleaned dict contains only the known fields with defaults
        applied.
    """
    if not isinstance(body, dict):
        return None, "signal must be a JSON object"

    signal_type = (body.get("signal_type") or "").strip()
    if not signal_type:
        return None, "signal_type is required"

    content = body.get("content")
    if content is None or str(content).strip() == "":
        return None, "content is required"

    activation_energy = body.get("activation_energy", 0.5)
    try:
        activation_energy = float(activation_energy)
    except (TypeError, ValueError):
        return None, "activation_energy must be a number"
    if not (0.0 <= activation_energy <= 1.0):
        return None, "activation_energy must be between 0 and 1"

    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return None, "metadata must be a JSON object or null"

    return {
        "signal_type": signal_type,
        "source": (body.get("source") or "").strip() or None,
        "content": str(content),
        "topic": body.get("topic") or None,
        "activation_energy": activation_energy,
        "metadata": metadata,
    }, None


def _build_and_emit(validated: dict, wrapper_id: str) -> str:
    """Construct a ReasoningSignal from validated data and emit it.

    The ``activation_energy`` value drives priority:
    - >= 0.8 → pushed to ``reasoning:priority`` queue by the caller
    - < 0.8  → pushed to ``reasoning:signals`` (normal queue)
    Both queues are already handled by :func:`emit_reasoning_signal` when the
    correct ``activation_energy`` is set — but that function always uses the
    normal queue.  High-priority signals are pushed directly here to match the
    pattern used by the WebSocket chat handler.

    Args:
        validated: Dict produced by :func:`_validate_signal`.
        wrapper_id: The effective wrapper identifier for this signal.

    Returns:
        A new UUID string identifying the emitted signal.
    """
    from services.reasoning_loop_service import ReasoningSignal, emit_reasoning_signal
    from services.memory_client import MemoryClientService

    signal_id = str(uuid.uuid4())
    source = validated["source"] or wrapper_id

    signal = ReasoningSignal(
        signal_type=validated["signal_type"],
        source=source,
        content=validated["content"],
        topic=validated["topic"],
        activation_energy=validated["activation_energy"],
        metadata=validated["metadata"],
        wrapper_id=wrapper_id,
    )

    # High-priority signals go straight to the priority queue (same pattern as
    # WebSocket chat handler for user_message signals).
    if validated["activation_energy"] >= 0.8:
        store = MemoryClientService.create_connection()
        store.rpush("reasoning:priority", signal.to_json())
        logger.debug(
            "[Signals API] High-priority signal %s from %s → reasoning:priority",
            validated["signal_type"], wrapper_id,
        )
    else:
        emit_reasoning_signal(signal)
        logger.debug(
            "[Signals API] Signal %s from %s → reasoning:signals",
            validated["signal_type"], wrapper_id,
        )

    return signal_id


# ---------------------------------------------------------------------------
# POST /api/signals — single signal
# ---------------------------------------------------------------------------

@signals_bp.route("", methods=["POST"])
@require_auth
def ingest_signal():
    """Ingest a single signal into the reasoning loop.

    Body (JSON):
        signal_type (str, required): Identifies the type of signal.
        content (str, required): Human-readable signal content / description.
        source (str, optional): Identifies the originating system.  Defaults
            to the wrapper_id (or ``'__chat_ui__'`` for cookie auth).
        topic (str | null, optional): Conversation topic hint.
        activation_energy (float 0–1, optional): Urgency / priority weight.
            Values >= 0.8 go to the priority queue.  Defaults to ``0.5``.
        metadata (dict | null, optional): Arbitrary key-value context.

    Returns:
        202 ``{"ok": true, "signal_id": "<uuid>"}`` on success.
        400 if validation fails.
        403 if the wrapper is not permitted to emit this signal_type.
        429 if the rate limit is exceeded.
    """
    wrapper_id = _effective_wrapper_id()
    body = request.get_json(silent=True) or {}

    validated, err = _validate_signal(body)
    if err:
        return jsonify({"error": err}), 400

    # Capability check for bearer-authenticated callers
    if not _check_signal_capability(wrapper_id, validated["signal_type"]):
        return jsonify({
            "error": f"wrapper is not permitted to emit signal type '{validated['signal_type']}'"
        }), 403

    # Rate limit check
    limiter = _get_rate_limiter()
    if not limiter.is_allowed(wrapper_id):
        return jsonify({"error": "rate limit exceeded"}), 429

    signal_id = _build_and_emit(validated, wrapper_id)
    return jsonify({"ok": True, "signal_id": signal_id}), 202


# ---------------------------------------------------------------------------
# POST /api/signals/batch — batch ingest
# ---------------------------------------------------------------------------

@signals_bp.route("/batch", methods=["POST"])
@require_auth
def ingest_signals_batch():
    """Ingest up to 50 signals in a single request.

    Body (JSON):
        Array of signal objects (same schema as the single-signal endpoint).

    Each signal is validated and capability-checked independently.  Valid
    signals are emitted even when others in the batch fail.  Rate limit
    checking applies per-signal; once the limit is hit, remaining signals in
    the batch are rejected with a rate-limit error.

    Returns:
        200 ``{"accepted": N, "rejected": M, "errors": [...]}``
        ``errors`` is a list of ``{"index": I, "error": "..."}`` objects.
        400 if the request body is not a JSON array.
    """
    wrapper_id = _effective_wrapper_id()
    body = request.get_json(silent=True)

    if not isinstance(body, list):
        return jsonify({"error": "request body must be a JSON array"}), 400

    if len(body) > _BATCH_MAX:
        return jsonify({
            "error": f"batch exceeds maximum size of {_BATCH_MAX} signals"
        }), 400

    limiter = _get_rate_limiter()
    accepted = 0
    rejected = 0
    errors = []

    for idx, item in enumerate(body):
        # Validate
        validated, err = _validate_signal(item)
        if err:
            rejected += 1
            errors.append({"index": idx, "error": err})
            continue

        # Capability check
        if not _check_signal_capability(wrapper_id, validated["signal_type"]):
            rejected += 1
            errors.append({
                "index": idx,
                "error": f"wrapper is not permitted to emit signal type '{validated['signal_type']}'",
            })
            continue

        # Rate limit
        if not limiter.is_allowed(wrapper_id):
            rejected += 1
            errors.append({"index": idx, "error": "rate limit exceeded"})
            continue

        # Emit
        try:
            _build_and_emit(validated, wrapper_id)
            accepted += 1
        except Exception as exc:
            logger.error("[Signals API] Batch emit error at index %d: %s", idx, exc, exc_info=True)
            rejected += 1
            errors.append({"index": idx, "error": "internal error during signal emission"})

    return jsonify({"accepted": accepted, "rejected": rejected, "errors": errors}), 200
