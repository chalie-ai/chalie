"""
Capability base — abstract base class for all capability plugins.

Every capability plugin must subclass :class:`AbstractCapability` and implement
all abstract methods defining the four-phase cognitive pipeline
(ingest/understand/monitor/act) plus lifecycle methods.  The concrete helper methods handle credential
encryption / storage so individual capabilities don't need to deal with
cryptographic details.

Credential storage
------------------
Credentials are stored in the ``tool_configs`` table via
:class:`~services.tool_config_service.ToolConfigService`.  Values are encrypted
with `Fernet <https://cryptography.io/en/latest/fernet/>`_ symmetric encryption
before being written; the Fernet key is derived from the application-wide
encryption key returned by
:func:`~services.encryption_key_service.get_encryption_key`.

Lazy imports
------------
All service imports are performed *inside* method bodies rather than at module
level to prevent circular-import issues during early application boot.
"""

import base64
import hashlib
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def _make_fernet():
    """Build and return a :class:`cryptography.fernet.Fernet` instance.

    The Fernet key is derived by taking the first 32 bytes of the application
    encryption key (UTF-8 encoded), left-padding with ``b'='`` if shorter than
    32 bytes, then base64-url-encoding the result to produce the 44-byte key
    string that Fernet expects.

    Imports are intentionally deferred to avoid circular dependencies during
    application boot.

    Returns:
        cryptography.fernet.Fernet: A ready-to-use Fernet cipher instance.
    """
    from cryptography.fernet import Fernet
    from services.encryption_key_service import get_encryption_key

    key_bytes = hashlib.sha256(get_encryption_key().encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


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
    def monitor(self) -> None:
        """Detect changes and emit signals.

        Called periodically by the capability scheduler. Should compare
        current state with previous state, detect new/changed/deleted items,
        and emit signals via EventBusService. This is the primary entry point
        the scheduler calls each cycle.
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
        """Execute :meth:`monitor` with health tracking.

        Resets error count on success, increments on failure, auto-disconnects
        after :attr:`MAX_CONSECUTIVE_FAILURES`.  Persists state to tool_configs.
        """
        try:
            self.monitor()
            self._error_count = 0
            self._last_error = None
            self._failure_alerted = False
            try:
                from capabilities.first_look import maybe_send_first_look
                maybe_send_first_look()
            except Exception:
                pass
            try:
                from capabilities.meeting_prep import maybe_send_meeting_prep
                maybe_send_meeting_prep()
            except Exception as exc:
                logger.debug("meeting_prep hook: %s", exc)
        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)
            logger.warning("[%s] monitor() failed (%d/%d): %s",
                           self.get_id(), self._error_count,
                           self.MAX_CONSECUTIVE_FAILURES, exc)
            if self._error_count >= self.MAX_CONSECUTIVE_FAILURES:
                self._connected = False
                self._maybe_send_failure_alert()
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
        """Encrypt *value* with Fernet and persist it in ``tool_configs``.

        The credential is stored under ``tool_name = self.get_id()`` and
        ``config_key = key``.

        Args:
            key:   Config key, e.g. ``"caldav:password"``.
            value: Plaintext credential value to encrypt and store.

        Returns:
            None
        """
        try:
            fernet = _make_fernet()
            encrypted = fernet.encrypt(value.encode()).decode()
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
        """Load and decrypt a stored credential.

        Returns ``None`` if the key is absent or decryption fails (e.g. key
        rotation has invalidated the ciphertext).

        Args:
            key: Config key, e.g. ``"caldav:password"``.

        Returns:
            str | None: Decrypted plaintext value, or ``None``.
        """
        try:
            svc = _get_tool_config_service()
            config = svc.get_tool_config(self.get_id())
            encrypted = config.get(key)
            if encrypted is None:
                return None
            fernet = _make_fernet()
            return fernet.decrypt(encrypted.encode()).decode()
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
