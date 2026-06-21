"""Intent Service — structured intents from Chalie to external wrappers.

Chalie emits CognitiveIntents when it needs a wrapper to execute something on its behalf. Intents
are transient coordination messages stored in MemoryStore. Broadcast intents (target_wrapper=None)
land in ``intents:__broadcast__``; individual ones are also keyed by ID for O(1) lookup.
"""

import dataclasses
import json
import logging
from typing import Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from services.memory_store import MemoryStore

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

_BROADCAST_KEY = "__broadcast__"
_INTENT_TTL_SECONDS = 24 * 60 * 60  # 24 hours


@dataclasses.dataclass
class CognitiveIntent:
    """Structured intent emitted by the cognitive runtime to a wrapper."""

    intent_id: str
    intent_type: str
    target_wrapper: Optional[str]
    payload: dict[str, object]
    urgency: str = "normal"
    confidence: float = 0.5
    expires_at: Optional[str] = None
    requires_ack: bool = False
    created_at: str = dataclasses.field(
        default_factory=lambda: utc_now().isoformat()
    )
    status: str = "pending"

    def to_dict(self) -> dict[str, object]:
        """Serialize the intent to a plain dict for JSON storage."""
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """Serialize the intent to a JSON string."""
        return json.dumps(self.to_dict())


class IntentService:
    """Broker for cognitive intents between Chalie and external wrappers."""

    def __init__(self, store: "Optional[MemoryStore]" = None) -> None:
        """Initialize with an optional MemoryStore (uses shared singleton when None)."""
        if store is None:
            from services.memory_client import MemoryClientService
            self._store = MemoryClientService.create_connection()
        else:
            self._store = store

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, intent: CognitiveIntent) -> None:
        """Store the intent and notify the target wrapper channel via pub/sub broadcast."""
        target = intent.target_wrapper or _BROADCAST_KEY
        list_key = f"intents:{target}"
        id_key = f"intent:{intent.intent_id}"

        serialized = intent.to_json()

        # Append to the pending-intents list for this wrapper
        self._store.rpush(list_key, serialized)
        self._store.expire(list_key, _INTENT_TTL_SECONDS)

        # Store individually for O(1) lookup (24-hour TTL)
        self._store.setex(id_key, _INTENT_TTL_SECONDS, serialized)

        # Push intent to UI so streaming consumers see it immediately
        try:
            from services.websocket_broker import WebSocketBroker
            WebSocketBroker().broadcast({
                "type": "intent",
                "intent_id": intent.intent_id,
                "intent_type": intent.intent_type,
                "target": target,
            })
        except Exception:
            pass

        logger.debug(
            "[IntentService] Emitted intent %s (type=%s target=%s)",
            intent.intent_id,
            intent.intent_type,
            target,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_pending(self, wrapper_id: str, limit: int = 10) -> list[dict[str, object]]:
        """Fetch undelivered pending intents for ``wrapper_id`` from both its list and the broadcast list, mark them delivered."""
        results = []
        seen_ids: set[str] = set()

        for key in (f"intents:{wrapper_id}", f"intents:{_BROADCAST_KEY}"):
            raw_items = self._store.lrange(key, 0, -1)
            for raw in raw_items:
                try:
                    intent_dict = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                intent_id = intent_dict.get("intent_id")
                if not intent_id or intent_id in seen_ids:
                    continue
                seen_ids.add(intent_id)

                # Read canonical status from intent:{id} key (list may be stale)
                canonical_raw = self._store.get(f"intent:{intent_id}")
                if canonical_raw:
                    try:
                        intent_dict = json.loads(canonical_raw)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if intent_dict.get("status") != "pending":
                    continue

                # Check expiry
                if self._is_expired(intent_dict):
                    intent_dict["status"] = "expired"
                    self._persist_intent(intent_dict)
                    continue

                # Mark as delivered
                intent_dict["status"] = "delivered"
                self._persist_intent(intent_dict)

                results.append(intent_dict)
                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        return results

    def get_intent(self, intent_id: str) -> Optional[dict[str, object]]:
        """Retrieve a single intent by its ID; ``None`` when not found or expired."""
        raw = self._store.get(f"intent:{intent_id}")
        if not raw:
            return None
        try:
            return cast(dict[str, object], json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Lifecycle mutations
    # ------------------------------------------------------------------

    def acknowledge(self, intent_id: str, wrapper_id: str) -> bool:
        """Mark an intent as acknowledged; returns False when it doesn't exist."""
        intent = self.get_intent(intent_id)
        if intent is None:
            return False

        intent["status"] = "acknowledged"
        self._persist_intent(intent)
        logger.debug(
            "[IntentService] Intent %s acknowledged by wrapper %s",
            intent_id, wrapper_id,
        )
        return True

    def resolve(self, intent_id: str, result: dict[str, object]) -> bool:
        """Record execution status reported by a wrapper. The ``result`` dict should contain
        ``status`` (``"executed"``, ``"failed"``, or ``"skipped"``) plus any type-specific fields."""
        intent = self.get_intent(intent_id)
        if intent is None:
            return False

        new_status = result.get("status", "executed")
        if new_status not in ("executed", "failed", "skipped"):
            new_status = "executed"

        intent["status"] = new_status
        intent["execution_result"] = result
        self._persist_intent(intent)
        logger.debug(
            "[IntentService] Intent %s resolved with status=%s",
            intent_id, new_status,
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_intent(self, intent_dict: dict[str, object]) -> None:
        """Persist an intent to its ``intent:{id}`` key."""
        intent_id = intent_dict.get("intent_id")
        if not intent_id:
            return
        serialized = json.dumps(intent_dict)
        self._store.setex(f"intent:{intent_id}", _INTENT_TTL_SECONDS, serialized)

    @staticmethod
    def _is_expired(intent_dict: dict[str, object]) -> bool:
        """Check whether an intent has passed its ``expires_at`` deadline."""
        expires_at = intent_dict.get("expires_at")
        if not expires_at:
            return False
        try:
            expiry_dt = parse_utc(cast(str, expires_at))
            return utc_now() > expiry_dt
        except Exception:
            return False
