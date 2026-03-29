"""
Capability base — abstract base class for all capability plugins.

Every capability plugin must subclass :class:`AbstractCapability` and implement
all abstract methods defining the four-phase cognitive pipeline
(ingest/understand/monitor/act) plus lifecycle methods.  The concrete helper
methods handle credential encryption / storage so individual capabilities don't
need to deal with cryptographic details.

Credential storage
------------------
Credentials are stored in the ``tool_configs`` table via
:class:`~services.tool_config_service.ToolConfigService`.  Values are encrypted
with AES-256-GCM via :class:`~services.vault_service.VaultService` before being
written; the vault must be unlocked (via
:meth:`~services.vault_service.VaultService.unlock`) before any credential
operations can succeed.

Lazy imports
------------
All service imports are performed *inside* method bodies rather than at module
level to prevent circular-import issues during early application boot.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proactive hook registry — hooks self-register at import time
# ---------------------------------------------------------------------------

_hook_registry: list = []
_hooks_discovered = False


def register_hook(fn) -> None:
    """Register a proactive hook to fire after each monitor cycle.

    Hook modules call this at module level so they are automatically
    discovered without naming them in infrastructure code.
    """
    if fn not in _hook_registry:
        _hook_registry.append(fn)


def _discover_hooks() -> None:
    """Import capability-level modules once to trigger hook self-registration."""
    global _hooks_discovered
    if _hooks_discovered:
        return
    _hooks_discovered = True

    import importlib
    from pathlib import Path

    cap_dir = Path(__file__).parent
    for py_file in sorted(cap_dir.glob("[!_]*.py")):
        name = py_file.stem
        if name == "base":
            continue
        try:
            importlib.import_module(f"capabilities.{name}")
        except Exception as exc:
            logger.debug("hook discovery: skip %s: %s", name, exc)


def _drain_prompt_queue(max_items: int = 50) -> int:
    """Drain the MemoryStore ``prompt-queue`` list and dispatch via PromptQueue.

    Hooks push JSON items to the MemoryStore list with ``store.rpush``.
    This function pops them and dispatches each to :func:`digest_worker`
    through the working :class:`PromptQueue` thread mechanism.

    Returns:
        int: Number of items dispatched.
    """
    try:
        import json
        from services.memory_client import MemoryClientService
        from services.prompt_queue import PromptQueue
        from workers.digest_worker import digest_worker

        store = MemoryClientService.create_connection()
        queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
        dispatched = 0

        for _ in range(max_items):
            raw = store.lpop("prompt-queue")
            if raw is None:
                break
            try:
                item = json.loads(raw)
                prompt = item.get("prompt", "")
                metadata = item.get("metadata", {})
                if prompt:
                    queue.enqueue(prompt, metadata=metadata)
                    dispatched += 1
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.debug("prompt-queue drain: bad item: %s", exc)

        if dispatched:
            logger.info("[prompt-queue] Drained %d item(s)", dispatched)
        return dispatched
    except Exception as exc:
        logger.debug("prompt-queue drain failed: %s", exc)
        return 0


def _get_tool_config_service():
    """Return a :class:`~services.tool_config_service.ToolConfigService` instance.

    Uses the shared database service singleton.  Deferred import prevents
    circular imports during early boot.

    Returns:
        services.tool_config_service.ToolConfigService: Configured service
        bound to the shared database connection.
    """
    from services.database_service import get_shared_db_service
    from services.tool_config_service import ToolConfigService

    return ToolConfigService(get_shared_db_service())


class AbstractCapability(ABC):
    """Abstract base class that every capability plugin must subclass.

    Subclasses must implement all abstract methods that define the four-phase
    cognitive pipeline (ingest → understand → monitor → act) plus lifecycle
    and identity methods:

    **Identity & lifecycle:**
    * :meth:`get_id`
    * :meth:`get_manifest`
    * :meth:`configure`
    * :meth:`connect`
    * :meth:`disconnect`
    * :meth:`health`

    **Cognitive pipeline:**
    * :meth:`ingest` — pull raw data from the external source
    * :meth:`understand` — extract meaning from raw items via LLM or parsing
    * :meth:`monitor` — detect changes, emit signals (called by scheduler)
    * :meth:`act` — perform a write action on the external source
    * :meth:`get_tools` — return tool definitions for dynamic registration

    Concrete helper methods for credential management, connection-state
    tracking, and health monitoring are provided by this base class and
    should not be overridden unless there is a specific need.
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self) -> None:
        """Initialise base state.

        Sets the internal ``_connected`` flag to ``False`` and health
        tracking counters to their defaults.  Subclasses should call
        ``super().__init__()`` if they define their own ``__init__``.
        """
        self._connected: bool = False
        self._error_count: int = 0
        self._last_error: str | None = None
        self._failure_alerted: bool = False
        self._first_monitor: bool = True
        self._backoff_secs: int = 0
        self._next_retry_at = None

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def get_id(self) -> str:
        """Return the unique capability identifier (must match ``manifest.yaml``).

        Returns:
            str: Lowercase, hyphen-separated identifier, e.g. ``"caldav"``.
        """

    @abstractmethod
    def get_manifest(self) -> dict:
        """Return the parsed ``manifest.yaml`` contents as a dict.

        Returns:
            dict: Full manifest, including at minimum ``id``, ``name``,
            ``version``, and ``entry_class`` keys.
        """

    @abstractmethod
    def configure(self, credentials: dict) -> None:
        """Accept, validate, and persist credentials for this capability.

        Implementations should call :meth:`store_credential` to persist each
        field and raise :exc:`ValueError` with a descriptive message if the
        credentials are rejected (e.g. failed auth test against the remote
        server).

        Args:
            credentials: Provider-specific mapping, e.g.
                ``{"provider": "google", "username": "...", "password": "..."}``.

        Raises:
            ValueError: If credentials are invalid or the remote server rejects
                them.
        """

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the capability's data source.

        Should load stored credentials via :meth:`load_credential`, attempt to
        reach the remote service, update ``self._connected``, and return the
        result without raising on transient errors.

        Returns:
            bool: ``True`` if the connection was established successfully,
            ``False`` otherwise.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the active connection and clear any cached state.

        Must set ``self._connected = False``.  Should not raise.
        """

    @abstractmethod
    def ingest(self) -> list:
        """Fetch and return structured data from the capability's data source.

        This method is called periodically by the scheduler via system
        handler dispatch and should be idempotent.

        Returns:
            list[dict]: A list of structured data dicts whose schema is
            defined by the concrete capability.
        """

    @abstractmethod
    def understand(self, items: list) -> list:
        """Extract meaning from raw ingested items.

        Called after :meth:`ingest` with the returned items. Implementations
        should extract entities, dates, people, commitments, and patterns
        via LLM prompts or deterministic parsing.

        Args:
            items: Raw data dicts returned by :meth:`ingest`.

        Returns:
            list[dict]: Enriched/extracted knowledge dicts ready for storage
            via KnowledgeService.
        """

    @abstractmethod
    def _do_monitor(self) -> None:
        """Detect changes and emit signals (subclass implementation).

        Called periodically by the capability scheduler via :meth:`monitor`
        or :meth:`run_monitor`. Should compare current state with previous
        state, detect new/changed/deleted items, and emit signals via
        EventBusService.
        """

    @abstractmethod
    def act(self, action: str, params: dict) -> dict:
        """Perform a write action on the external data source.

        Args:
            action: Action name matching one of the manifest's declared actions,
                e.g. ``"create_event"``, ``"send_email"``.
            params: Action-specific parameters.

        Returns:
            dict: Result of the action, including at minimum a ``success``
            boolean key.
        """

    def monitor(self) -> None:
        """Run the monitor cycle and dispatch proactive hooks.

        Calls :meth:`_do_monitor` (the subclass implementation) followed by
        proactive hooks.  This is the public entry point that ensures hooks
        fire regardless of whether the caller uses ``monitor()`` or
        ``run_monitor()``.

        On the **first** call (initial sync after connect), hooks fire
        *before* ``_do_monitor()`` as well.  This lets time-sensitive hooks
        like ``first_look`` use data already available from other
        capabilities without waiting for this capability's slow initial
        fetch.  Hook dedup flags prevent double-firing.
        """
        if self._first_monitor:
            self._first_monitor = False
            self._dispatch_hooks()
        self._do_monitor()
        self._dispatch_hooks()

    def _dispatch_hooks(self) -> None:
        """Fire all registered proactive hooks after a successful monitor."""
        _discover_hooks()
        for fn in _hook_registry:
            try:
                fn()
            except Exception as exc:
                logger.debug("%s hook: %s", fn.__name__, exc)
        _drain_prompt_queue()

    def health(self) -> bool:
        """Quick connectivity check.

        Default implementation returns the current ``_connected`` state.
        Subclasses may override to perform an active probe (e.g. a lightweight
        server request).

        Returns:
            bool: ``True`` if the capability's data source is reachable.
        """
        return self._connected

    def health_details(self) -> dict:
        """Return detailed health information including error tracking.

        Returns:
            dict: ``{"connected": bool, "error_count": int,
            "last_error": str | None}``.
        """
        return {
            "connected": self._connected,
            "error_count": self._error_count,
            "last_error": self._last_error,
        }

    def run_monitor(self) -> None:
        """Execute :meth:`monitor` with health tracking and circuit breaker.

        Resets error count on success, increments on failure, auto-disconnects
        after :attr:`MAX_CONSECUTIVE_FAILURES`.  When disconnected, applies
        exponential backoff so dead servers are not hammered every cycle.
        Sends a recovery alert when a degraded capability comes back online.
        """
        if self._next_retry_at:
            from services.time_utils import utc_now
            if utc_now() < self._next_retry_at:
                return

        try:
            self.monitor()
            if self._backoff_secs and self._failure_alerted:
                self._send_recovery_alert()
            self._error_count = 0
            self._last_error = None
            self._failure_alerted = False
            self._backoff_secs = 0
            self._next_retry_at = None
        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)
            logger.warning("[%s] monitor() failed (%d/%d): %s",
                           self.get_id(), self._error_count,
                           self.MAX_CONSECUTIVE_FAILURES, exc)
            if self._error_count >= self.MAX_CONSECUTIVE_FAILURES:
                self._connected = False
                self._maybe_send_failure_alert()
                self._activate_backoff()
        self._persist_health()

    def _persist_health(self) -> None:
        """Persist current error state to ``tool_configs``."""
        try:
            svc = _get_tool_config_service()
            cap_id = self.get_id()
            svc.set_tool_config(cap_id, {
                f"{cap_id}:error_count": str(self._error_count),
                f"{cap_id}:last_error": self._last_error or "",
            })
        except Exception:
            pass

    def _maybe_send_failure_alert(self) -> None:
        """Push a user-facing alert when a capability disconnects.

        Fires once per disconnection event.  Reset when monitor()
        succeeds again (``_failure_alerted`` cleared on success path).
        """
        if self._failure_alerted:
            return
        self._failure_alerted = True
        cap_id = self.get_id()
        cap_name = self.get_manifest().get("name", cap_id)
        err = self._last_error or "unknown error"
        msg = (
            f"{cap_name} disconnected after "
            f"{self.MAX_CONSECUTIVE_FAILURES} failures."
            f" Last error: {err}"
        )
        try:
            from capabilities.signal_bridge import (
                emit_capability_signal,
            )
            emit_capability_signal(
                cap_id, "capability_failure", msg,
                source=f"{cap_id}:health",
            )
        except Exception as exc:
            logger.debug("failure signal emit: %s", exc)
        try:
            import json
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.rpush("prompt-queue", json.dumps({
                "prompt": (
                    f"[CAPABILITY ALERT] {cap_name} has "
                    f"disconnected — {err}. "
                    "Let the user know and suggest "
                    "reconnecting via settings."
                ),
                "metadata": {
                    "type": "proactive_drift",
                    "source": f"capability_health:{cap_id}",
                    "topic": "proactive",
                },
            }))
        except Exception as exc:
            logger.debug("failure alert push: %s", exc)

    def _send_recovery_alert(self) -> None:
        """Push a user-facing message when a degraded capability recovers."""
        cap_id = self.get_id()
        cap_name = self.get_manifest().get("name", cap_id)
        try:
            import json
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.rpush("prompt-queue", json.dumps({
                "prompt": (
                    f"[CAPABILITY RECOVERED] {cap_name} is back "
                    "online. Sync has resumed normally."
                ),
                "metadata": {
                    "type": "proactive_drift",
                    "source": f"capability_health:{cap_id}",
                    "topic": "proactive",
                },
            }))
        except Exception as exc:
            logger.debug("recovery alert push: %s", exc)

    INITIAL_BACKOFF_SECS = 60
    MAX_BACKOFF_SECS = 1800

    def _activate_backoff(self) -> None:
        """Set or double the exponential backoff timer."""
        from datetime import timedelta
        from services.time_utils import utc_now

        self._backoff_secs = min(
            self._backoff_secs * 2 if self._backoff_secs else self.INITIAL_BACKOFF_SECS,
            self.MAX_BACKOFF_SECS,
        )
        self._next_retry_at = utc_now() + timedelta(seconds=self._backoff_secs)
        logger.info("[%s] backoff %ds", self.get_id(), self._backoff_secs)

    @abstractmethod
    def get_tools(self) -> list:
        """Return tool definitions for dynamic registration.

        Each entry in the returned list must be a dict with at minimum the
        keys ``name`` (str), ``handler`` (callable), and ``metadata`` (dict).

        Returns:
            list[dict]: Tool definitions suitable for passing to
            :func:`~services.tool_library_service.register_tool`.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — credential management & connection state
    # ------------------------------------------------------------------

    def store_credential(self, key: str, value: str) -> None:
        """Encrypt *value* with AES-256-GCM via VaultService and persist
        it in ``tool_configs``.

        The credential is stored under ``tool_name = self.get_id()`` and
        ``config_key = key``.  The encrypted blob is base64-encoded before
        being written so it can be safely stored as a text column.

        The vault must be unlocked before calling this method; if it is sealed
        a :exc:`~services.vault_service.VaultLockedError` is raised and
        propagated to the caller.

        Args:
            key:   Config key, e.g. ``"caldav:password"``.
            value: Plaintext credential value to encrypt and store.

        Returns:
            None

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the vault has
                not yet been unlocked via
                :meth:`~services.vault_service.VaultService.unlock`.
            Exception: Any other cryptography or database error is re-raised
                after logging.
        """
        try:
            import base64
            from services.vault_service import get_vault_service
            blob = get_vault_service().encrypt_str(value)
            encrypted = base64.b64encode(blob).decode()
            svc = _get_tool_config_service()
            svc.set_tool_config(self.get_id(), {key: encrypted})
        except Exception as exc:
            logger.error(
                "[%s] store_credential(%r) failed: %s",
                self.get_id(), key, exc,
                exc_info=True,
            )
            raise

    def load_credential(self, key: str) -> str | None:
        """Load and AES-256-GCM–decrypt a stored credential via VaultService.

        Returns ``None`` if the key is absent, the vault is locked, or
        decryption fails (e.g. the vault has been re-initialised and the DEK
        has rotated).

        The stored value is expected to be a base64-encoded blob as written by
        :meth:`store_credential`.

        Args:
            key: Config key, e.g. ``"caldav:password"``.

        Returns:
            str | None: Decrypted plaintext value, or ``None`` on any failure.
        """
        try:
            svc = _get_tool_config_service()
            config = svc.get_tool_config(self.get_id())
            encrypted = config.get(key)
            if encrypted is None:
                return None
            import base64
            from services.vault_service import get_vault_service, VaultLockedError
            raw = base64.b64decode(encrypted.encode())
            return get_vault_service().decrypt_str(raw)
        except VaultLockedError as exc:
            logger.warning(
                "[%s] load_credential(%r) failed — vault is "
                "locked (returning None): %s",
                self.get_id(), key, exc,
            )
            return None
        except Exception as exc:
            logger.warning(
                "[%s] load_credential(%r) failed (returning None): %s",
                self.get_id(), key, exc,
            )
            return None

    def delete_credentials(self) -> None:
        """Remove ALL ``tool_configs`` rows associated with this capability.

        This is equivalent to calling
        :meth:`~services.tool_config_service.ToolConfigService.delete_tool_config`
        with ``self.get_id()`` as the tool name.

        Returns:
            None
        """
        try:
            svc = _get_tool_config_service()
            svc.delete_tool_config(self.get_id())
        except Exception as exc:
            logger.error(
                "[%s] delete_credentials() failed: %s",
                self.get_id(), exc,
                exc_info=True,
            )

    def is_connected(self) -> bool:
        """Return the current in-memory connection state.

        This reflects the most recent call to :meth:`connect` or
        :meth:`disconnect` and is *not* persisted across restarts.

        Returns:
            bool: ``True`` if the capability currently has an active
            connection, ``False`` otherwise.
        """
        return self._connected
