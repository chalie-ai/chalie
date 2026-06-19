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

def _get_tool_config_service():
    """Return a :class:`~services.tool_config_service.ToolConfigService` instance via deferred import."""
    from services.database_service import get_shared_db_service
    from services.tool_config_service import ToolConfigService

    return ToolConfigService(get_shared_db_service())


class AbstractCapability(ABC):
    """Abstract base class that every capability plugin must subclass."""

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self) -> None:
        self._connected: bool = False
        self._error_count: int = 0
        self._last_error: str | None = None
        self._failure_alerted: bool = False
        self._backoff_secs: int = 0
        self._next_retry_at = None

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def get_id(self) -> str:
        ...

    @abstractmethod
    def get_manifest(self) -> dict:
        ...

    @abstractmethod
    def configure(self, credentials: dict) -> None:
        """Accept, validate, and persist credentials for this capability.

        Implementations should call :meth:`store_credential` to persist each field
        and raise :exc:`ValueError` with a descriptive message if the credentials are
        rejected (e.g. failed auth test against the remote server).
        """

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the capability's data source.

        Should load stored credentials via :meth:`load_credential`, attempt to reach
        the remote service, update ``self._connected``, and return the result without
        raising on transient errors.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the active connection and clear any cached state."""

    @abstractmethod
    def ingest(self) -> list:
        ...

    @abstractmethod
    def understand(self, items: list) -> list:
        """Extract meaning from raw ingested items."""

    @abstractmethod
    def _do_monitor(self) -> None:
        ...

    @abstractmethod
    def act(self, action: str, params: dict) -> dict:
        ...

    def monitor(self) -> None:
        self._do_monitor()

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
            was_degraded = self._backoff_secs and self._failure_alerted
            if was_degraded:
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
        """Send a WebSocket status-bar alert when a capability disconnects.

        Fires once per disconnection event.  Reset when monitor()
        succeeds again (``_failure_alerted`` cleared on success path).
        """
        if self._failure_alerted:
            return
        self._failure_alerted = True
        cap_id = self.get_id()
        cap_name = self.get_manifest().get("name", cap_id)
        err = self._last_error or "unknown error"
        try:
            import json
            from services.memory_client import MemoryClientService
            from services.websocket_broker import WebSocketBroker
            store = MemoryClientService.create_connection()
            payload = {
                "type": "capability_alert",
                "cap_id": cap_id,
                "cap_name": cap_name,
                "error": err,
                "recovered": False,
            }
            WebSocketBroker().broadcast(payload)
            store.setex(
                f"capability:alert:{cap_id}",
                1800,
                json.dumps(payload),
            )
        except Exception as exc:
            logger.debug("failure alert push: %s", exc)

    def _send_recovery_alert(self) -> None:
        """Send a WebSocket status-bar recovery notification."""
        cap_id = self.get_id()
        cap_name = self.get_manifest().get("name", cap_id)
        try:
            from services.memory_client import MemoryClientService
            from services.websocket_broker import WebSocketBroker
            store = MemoryClientService.create_connection()
            payload = {
                "type": "capability_alert",
                "cap_id": cap_id,
                "cap_name": cap_name,
                "recovered": True,
            }
            WebSocketBroker().broadcast(payload)
            store.delete(f"capability:alert:{cap_id}")
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
        """Return tool definitions exposed by this capability.

        Each entry in the returned list must be a dict with at minimum the
        keys ``name`` (str), ``handler`` (callable), and ``parameters`` (dict).
        Handlers are closures that check connection state and dispatch to the
        appropriate protocol handler. Ability subclasses (email, calendar,
        contacts) call ``get_tools()`` to obtain and invoke these handlers.

        Returns:
            list[dict]: Tool definition dicts with ``name``, ``handler``,
            ``parameters``, ``description``, and ``timeout`` keys.
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
        except VaultLockedError:
            raise
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
