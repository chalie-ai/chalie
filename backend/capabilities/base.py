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
:class:`~models.tool_config.ToolConfig`.  Values are encrypted
with AES-256-GCM via :class:`~services.vault_service.VaultService` before being
written; the vault must be unlocked (via
:meth:`~services.vault_service.VaultService.unlock`) before any credential
operations can succeed.
"""

import logging
from abc import ABC, abstractmethod

from models.tool_config import ToolConfig
from services.vault_service import VaultLockedError

logger = logging.getLogger(__name__)


class AbstractCapability(ABC):
    """Abstract base class that every capability plugin must subclass."""

    def __init__(self) -> None:
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def get_id(self) -> str:
        ...

    @abstractmethod
    def get_manifest(self) -> dict[str, object]:
        ...

    @abstractmethod
    def configure(self, credentials: dict[str, object]) -> None:
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
    def ingest(self) -> list[object]:
        ...

    @abstractmethod
    def understand(self, items: list[object]) -> list[object]:
        """Extract meaning from raw ingested items."""

    @abstractmethod
    def _do_monitor(self) -> None:
        ...

    def monitor(self) -> None:
        self._do_monitor()

    @abstractmethod
    def get_tools(self) -> list[dict[str, object]]:
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
            ToolConfig.set_encrypted(self.get_id(), key, value)
        except Exception as exc:
            logger.error(
                "[%s] store_credential(%r) failed: %s",
                self.get_id(), key, exc,
                exc_info=True,
            )
            raise

    def load_credential(self, key: str) -> str | None:
        """Load and AES-256-GCM–decrypt a stored credential via VaultService.

        Returns ``None`` if the key is absent or decryption fails (e.g. the
        vault has been re-initialised and the DEK has rotated). A *locked*
        vault is different: :exc:`~services.vault_service.VaultLockedError` is
        re-raised so the caller (e.g. capability setup) can surface a clear
        "unlock the vault" response rather than treating the credential as
        merely absent.

        The stored value is expected to be a base64-encoded blob as written by
        :meth:`store_credential`.

        Args:
            key: Config key, e.g. ``"caldav:password"``.

        Returns:
            str | None: Decrypted plaintext value, or ``None`` on absent/failed
            decryption.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the vault has
                not yet been unlocked.
        """
        try:
            return ToolConfig.get_encrypted(self.get_id(), key)
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
        :meth:`~models.tool_config.ToolConfig.delete_all`
        with ``self.get_id()`` as the tool name.

        Returns:
            None
        """
        try:
            ToolConfig.delete_all(self.get_id())
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
