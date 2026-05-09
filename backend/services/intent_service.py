"""
Intent Service — structured intents from Chalie to external wrappers.

Chalie emits CognitiveIntents when it needs a wrapper to execute something
on its behalf (e.g. open a PR, run a command, display a notification).
The intent is stored in MemoryStore and delivered via the REST API when
the wrapper polls or receives a push notification.

Key design choices:
- Storage is ephemeral (MemoryStore) — intents are transient coordination
  messages, not permanent records.
- Broadcast intents (target_wrapper=None) land in ``intents:__broadcast__``.
- Individual intents are also stored by ID for O(1) lookup.
- Pub/sub channel ``intents:{wrapper_id}`` lets connected wrappers receive
  intents via a streaming endpoint in the future.
"""

import dataclasses
import json
import logging
import uuid
from typing import Optional

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

_BROADCAST_KEY = "__broadcast__"
_INTENT_TTL_SECONDS = 24 * 60 * 60  # 24 hours


@dataclasses.dataclass
class CognitiveIntent:
    """Structured intent emitted by the cognitive runtime to a wrapper.

    Attributes:
        intent_id: UUID string uniquely identifying this intent.
        intent_type: Semantic category — ``"suggest"``, ``"notify"``,
            ``"request_input"``, or ``"execute"``.
        target_wrapper: The ``wrapper_id`` of the intended recipient, or
            ``None`` to broadcast to all capable wrappers.
        payload: Type-specific structured data the wrapper acts on.
        urgency: Priority hint — ``"critical"``, ``"normal"``, or ``"low"``.
        confidence: Chalie's confidence in this intent (0.0–1.0).
        expires_at: ISO-8601 UTC timestamp after which the intent expires,
            or ``None`` for no expiry.
        requires_ack: When ``True``, the wrapper must call the ``/ack``
            endpoint after receiving the intent.
        created_at: ISO-8601 UTC timestamp set at creation time.
        status: Lifecycle state — ``"pending"`` | ``"delivered"`` |
            ``"acknowledged"`` | ``"executed"`` | ``"failed"`` | ``"expired"``.
    """

    intent_id: str
    intent_type: str
    target_wrapper: Optional[str]
    payload: dict
    urgency: str = "normal"
    confidence: float = 0.5
    expires_at: Optional[str] = None
    requires_ack: bool = False
    created_at: str = dataclasses.field(
        default_factory=lambda: utc_now().isoformat()
    )
    status: str = "pending"

    def to_dict(self) -> dict:
        """Serialize the intent to a plain dict for JSON storage.

        Returns:
            Dict representation of all fields.
        """
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """Serialize the intent to a JSON string.

        Returns:
            JSON-encoded string.
        """
        return json.dumps(self.to_dict())


def _make_intent_id() -> str:
    """Generate a new UUID string for an intent.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


class IntentService:
    """Broker for cognitive intents between Chalie and external wrappers.

    Intents are stored in the shared MemoryStore.  The service is stateless
    between calls — all state lives in MemoryStore keys.
    """

    def __init__(self, store=None):
        """Initialize the service, optionally injecting a MemoryStore.

        Args:
            store: A MemoryStore instance.  When ``None``, the shared
                singleton is used via ``MemoryClientService.create_connection()``.
        """
        if store is None:
            from services.memory_client import MemoryClientService
            self._store = MemoryClientService.create_connection()
        else:
            self._store = store

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, intent: CognitiveIntent) -> None:
        """Store the intent and notify the target wrapper channel.

        The intent JSON is appended to the ``intents:{target}`` list and
        also stored under ``intent:{intent_id}`` for O(1) retrieval.  A
        pub/sub message is published to ``intents:{target}`` so any
        streaming subscriber can react immediately.

        Args:
            intent: The intent to emit.
        """
        target = intent.target_wrapper or _BROADCAST_KEY
        list_key = f"intents:{target}"
        id_key = f"intent:{intent.intent_id}"
        channel = f"intents:{target}"

        serialized = intent.to_json()

        # Append to the pending-intents list for this wrapper
        self._store.rpush(list_key, serialized)
        self._store.expire(list_key, _INTENT_TTL_SECONDS)

        # Store individually for O(1) lookup (24-hour TTL)
        self._store.setex(id_key, _INTENT_TTL_SECONDS, serialized)

        # Notify any streaming subscriber
        try:
            self._store.publish(channel, serialized)
        except Exception:
            pass  # Pub/sub failure must never block emission

        logger.debug(
            "[IntentService] Emitted intent %s (type=%s target=%s)",
            intent.intent_id,
            intent.intent_type,
            target,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_pending(self, wrapper_id: str, limit: int = 10) -> list[dict]:
        """Return undelivered intents for ``wrapper_id`` and mark them delivered.

        Fetches from both the wrapper-specific list and the broadcast list,
        filters to ``status == "pending"`` intents, marks each as
        ``"delivered"``, and returns them (most-recent-last, capped at
        ``limit``).

        Args:
            wrapper_id: The wrapper polling for intents.
            limit: Maximum number of intents to return.

        Returns:
            List of intent dicts (may be empty).
        """
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

    def get_intent(self, intent_id: str) -> Optional[dict]:
        """Retrieve a single intent by its ID.

        Args:
            intent_id: UUID string of the intent.

        Returns:
            Intent dict, or ``None`` if not found or expired.
        """
        raw = self._store.get(f"intent:{intent_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Lifecycle mutations
    # ------------------------------------------------------------------

    def acknowledge(self, intent_id: str, wrapper_id: str) -> bool:
        """Mark an intent as acknowledged by a wrapper.

        Args:
            intent_id: UUID string of the intent.
            wrapper_id: The wrapper acknowledging the intent (used for logging).

        Returns:
            ``True`` if the intent was found and updated, ``False`` otherwise.
        """
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

    def resolve(self, intent_id: str, result: dict) -> bool:
        """Record the execution result reported by a wrapper.

        The ``result`` dict should contain a ``status`` key
        (``"executed"``, ``"failed"``, or ``"skipped"``) plus any
        type-specific fields (e.g. ``result``, ``error``, ``reason``).

        Args:
            intent_id: UUID string of the intent.
            result: Execution result dict from the wrapper.

        Returns:
            ``True`` if the intent was found and updated, ``False`` otherwise.
        """
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

    def expire_stale(self) -> int:
        """Expire all intents that have passed their ``expires_at`` timestamp.

        Scans all ``intent:*`` keys in MemoryStore and transitions any
        ``pending``/``delivered`` intent past its TTL to ``"expired"``.

        Returns:
            Number of intents expired.
        """
        expired_count = 0
        # MemoryStore exposes _strings directly for key scanning (no SCAN command)
        try:
            with self._store._str_lock:
                all_keys = list(self._store._strings.keys())
        except AttributeError:
            return 0

        for key in all_keys:
            if not key.startswith("intent:"):
                continue
            raw = self._store.get(key)
            if not raw:
                continue
            try:
                intent = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            if intent.get("status") not in ("pending", "delivered"):
                continue

            if self._is_expired(intent):
                intent["status"] = "expired"
                self._persist_intent(intent)
                expired_count += 1

        logger.debug("[IntentService] expire_stale: expired %d intents", expired_count)
        return expired_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_intent(self, intent_dict: dict) -> None:
        """Update the ``intent:{id}`` key with the current intent state.

        Args:
            intent_dict: The intent dict (must contain ``intent_id``).
        """
        intent_id = intent_dict.get("intent_id")
        if not intent_id:
            return
        serialized = json.dumps(intent_dict)
        self._store.setex(f"intent:{intent_id}", _INTENT_TTL_SECONDS, serialized)

    @staticmethod
    def _is_expired(intent_dict: dict) -> bool:
        """Check whether an intent has passed its ``expires_at`` deadline.

        Args:
            intent_dict: The intent dict to check.

        Returns:
            ``True`` if ``expires_at`` is set and has passed, ``False`` otherwise.
        """
        expires_at = intent_dict.get("expires_at")
        if not expires_at:
            return False
        try:
            expiry_dt = parse_utc(expires_at)
            return utc_now() > expiry_dt
        except Exception:
            return False
