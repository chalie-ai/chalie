"""
Signal Ingestion API — external signals update world state directly (zero LLM).

Signals represent passive world knowledge — things Chalie overheard, not
things directed at Chalie.  They bypass the reasoning loop entirely and
write to the world state singleton's signals slot.  The reasoning
loop picks them up naturally during idle cycles via world state context.

For direct communication (messages that should trigger reasoning), use
the ``/api/messages`` endpoint or WebSocket chat instead.

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
  activation_energy float 0–1, optional (default 0.5, affects salience weight)
  metadata          dict | null, optional
"""

import logging
import uuid
from typing import TYPE_CHECKING, cast

from flask import g, request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_auth

if TYPE_CHECKING:
    from services.wrapper_auth_service import WrapperAuthService
    from services.wrapper_rate_limiter import WrapperRateLimiter

logger = logging.getLogger(__name__)

signals_bp = Namespace("signals", description="External signal ingestion", path="/api/signals")

# Maximum signals accepted in a single batch request
_BATCH_MAX = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rate_limiter() -> "WrapperRateLimiter":
    from services.wrapper_rate_limiter import WrapperRateLimiter
    return WrapperRateLimiter()


def _get_wrapper_service() -> "WrapperAuthService":
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


def _effective_wrapper_id() -> str:
    wid = getattr(g, "wrapper_id", None)
    return wid if wid else "__chat_ui__"


def _check_signal_capability(wrapper_id: str, signal_type: str) -> bool:
    """Cookie callers are always allowed; bearer callers need ``signal_type`` in
    ``capabilities.signals`` (absent/empty list means no signals permitted)."""
    if wrapper_id == "__chat_ui__":
        return True

    svc = _get_wrapper_service()
    wrapper = svc.get_wrapper(wrapper_id)
    if wrapper is None:
        return False

    allowed_signals = cast("list[str]", cast("dict[str, object]", wrapper.get("capabilities", {})).get("signals", []))
    return "*" in allowed_signals or signal_type in allowed_signals


def _validate_signal(body: "dict[str, object] | None") -> "tuple[dict[str, object] | None, str | None]":
    if not isinstance(body, dict):
        return None, "signal must be a JSON object"

    signal_type = (cast(str, body.get("signal_type")) or "").strip()
    if not signal_type:
        return None, "signal_type is required"

    content = body.get("content")
    if content is None or str(content).strip() == "":
        return None, "content is required"

    activation_energy = body.get("activation_energy", 0.5)
    try:
        activation_energy = float(cast(float, activation_energy))
    except (TypeError, ValueError):
        return None, "activation_energy must be a number"
    if not (0.0 <= activation_energy <= 1.0):
        return None, "activation_energy must be between 0 and 1"

    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return None, "metadata must be a JSON object or null"

    return {
        "signal_type": signal_type,
        "source": (cast(str, body.get("source")) or "").strip() or None,
        "content": str(content),
        "topic": body.get("topic") or None,
        "activation_energy": activation_energy,
        "metadata": metadata,
    }, None


def _build_and_emit(validated: "dict[str, object]", wrapper_id: str) -> str:
    """Write an external signal directly to world state (zero LLM).

    External signals update Chalie's world model without entering the
    reasoning loop.  The reasoning loop picks up world state changes
    during its normal idle cycle.  ``activation_energy`` is preserved
    as a salience weight for temporal scoring — higher energy signals
    stay visible longer in the world state.

    Args:
        validated: Dict produced by :func:`_validate_signal`.
        wrapper_id: The effective wrapper identifier for this signal.

    Returns:
        A new UUID string identifying the stored signal.
    """
    from services.world_state import world_state

    signal_id = str(uuid.uuid4())
    source = cast(str, validated["source"]) or wrapper_id

    world_state.push_signal(source, cast(str, validated["content"]), ttl=3600)

    logger.debug(
        "[Signals API] External signal %s from %s → world_state",
        validated["signal_type"], wrapper_id,
    )

    return signal_id


# ---------------------------------------------------------------------------
# POST /api/signals — single signal
# ---------------------------------------------------------------------------

@signals_bp.route("")
class SignalResource(Resource):
    @require_auth
    @signals_bp.response(202, "Accepted")
    @signals_bp.response(400, "Bad request")
    @signals_bp.response(403, "Forbidden")
    @signals_bp.response(429, "Rate limit exceeded")
    def post(self) -> ResponseReturnValue:
        """Ingest a single signal into world state (zero LLM).

        Body (JSON):
            signal_type (str, required): Identifies the type of signal.
            content (str, required): Human-readable signal content / description.
            source (str, optional): Identifies the originating system.  Defaults
                to the wrapper_id (or ``'__chat_ui__'`` for cookie auth).
            topic (str | null, optional): Conversation topic hint.
            activation_energy (float 0–1, optional): Salience weight.  Higher
                values make the signal persist longer in world state.  Default 0.5.
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
            return {"error": err}, 400

        # Capability check for bearer-authenticated callers
        if not _check_signal_capability(wrapper_id, cast(str, cast("dict[str, object]", validated)["signal_type"])):
            return {
                "error": f"wrapper is not permitted to emit signal type '{cast('dict[str, object]', validated)['signal_type']}'"
            }, 403

        # Rate limit check
        limiter = _get_rate_limiter()
        if not limiter.is_allowed(wrapper_id):
            return {"error": "rate limit exceeded"}, 429

        signal_id = _build_and_emit(cast("dict[str, object]", validated), wrapper_id)
        return {"ok": True, "signal_id": signal_id}, 202


# ---------------------------------------------------------------------------
# POST /api/signals/batch — batch ingest
# ---------------------------------------------------------------------------

@signals_bp.route("/batch")
class SignalBatchResource(Resource):
    @require_auth
    @signals_bp.response(200, "Success")
    @signals_bp.response(400, "Bad request")
    def post(self) -> ResponseReturnValue:
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
            return {"error": "request body must be a JSON array"}, 400

        if len(body) > _BATCH_MAX:
            return {
                "error": f"batch exceeds maximum size of {_BATCH_MAX} signals"
            }, 400

        limiter = _get_rate_limiter()
        accepted = 0
        rejected = 0
        errors: list[dict[str, object]] = []

        for idx, item in enumerate(body):
            # Validate
            validated, err = _validate_signal(item)
            if err:
                rejected += 1
                errors.append({"index": idx, "error": err})
                continue

            # Capability check
            if not _check_signal_capability(wrapper_id, cast(str, cast("dict[str, object]", validated)["signal_type"])):
                rejected += 1
                errors.append({
                    "index": idx,
                    "error": f"wrapper is not permitted to emit signal type '{cast('dict[str, object]', validated)['signal_type']}'",
                })
                continue

            # Rate limit
            if not limiter.is_allowed(wrapper_id):
                rejected += 1
                errors.append({"index": idx, "error": "rate limit exceeded"})
                continue

            # Emit
            try:
                _build_and_emit(cast("dict[str, object]", validated), wrapper_id)
                accepted += 1
            except Exception as exc:
                logger.exception("[Signals API] Batch emit error at index %d: %s", idx, exc)
                rejected += 1
                errors.append({"index": idx, "error": "internal error during signal emission"})

        return {"accepted": accepted, "rejected": rejected, "errors": errors}, 200