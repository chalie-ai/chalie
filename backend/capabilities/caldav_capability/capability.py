"""
CaldavCapability — concrete CalDAV calendar integration.

This module provides :class:`CaldavCapability`, a concrete implementation of
:class:`~capabilities.base.AbstractCapability` that connects to CalDAV-compatible
calendar providers (Google Calendar, Apple iCloud, Fastmail, Nextcloud, Synology,
Radicale) and exposes calendar data as knowledge facts and action tools.

Graceful degradation
--------------------
The ``caldav`` third-party library is imported inside a ``try/except`` block so
that the module can be imported even when the package is not installed.  Any
method that actually *uses* the library will raise :exc:`RuntimeError` in that
case.

Credential storage
------------------
Credentials are persisted under ``tool_name='caldav'`` in the ``tool_configs``
table.  Config keys follow the ``caldav:{field}`` convention:

- ``caldav:provider``  — provider identifier, e.g. ``"google"``
- ``caldav:username``  — account username / e-mail address
- ``caldav:password``  — app password (encrypted at rest)
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import yaml

from capabilities.base import AbstractCapability
from capabilities.caldav_capability.providers import resolve_provider
from services.time_utils import utc_now, parse_utc  # noqa: F401 — available for subclasses

# ---------------------------------------------------------------------------
# Optional caldav import — graceful degradation when package is absent
# ---------------------------------------------------------------------------

try:
    import caldav as _caldav_lib  # type: ignore
    _CALDAV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _caldav_lib = None  # type: ignore
    _CALDAV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

# Credential config keys
_KEY_PROVIDER = "caldav:provider"
_KEY_USERNAME = "caldav:username"
_KEY_PASSWORD = "caldav:password"

# CalDAV connection timeout in seconds
_CONNECT_TIMEOUT = 10


class CaldavCapability(AbstractCapability):
    """CalDAV calendar capability.

    Implements the :class:`~capabilities.base.AbstractCapability` interface for
    CalDAV-compatible calendar providers.  The 5 structural methods
    (:meth:`get_id`, :meth:`get_manifest`, :meth:`configure`, :meth:`connect`,
    :meth:`disconnect`) are fully implemented here.

    :meth:`ingest` and :meth:`get_tools` are concrete stubs that raise
    :exc:`NotImplementedError` — they will be implemented in subsequent tasks.

    Attributes:
        _connected (bool): Inherited from :class:`~capabilities.base.AbstractCapability`.
            ``True`` when a successful connection has been established.
    """

    def __init__(self) -> None:
        """Initialise the capability, setting connection state to ``False``."""
        super().__init__()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_id(self) -> str:
        """Return the unique capability identifier.

        Returns:
            str: Always ``"caldav"``.
        """
        return "caldav"

    def get_manifest(self) -> dict:
        """Load and return the parsed ``manifest.yaml`` for this capability.

        The manifest is read from disk on every call so that changes take effect
        without requiring a process restart.

        Returns:
            dict: Parsed YAML contents of ``manifest.yaml``, including at
            minimum ``id``, ``name``, ``version``, and ``entry_class``.

        Raises:
            FileNotFoundError: If ``manifest.yaml`` does not exist at the
                expected path alongside this module.
            yaml.YAMLError: If the manifest is not valid YAML.
        """
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Credential management & connection lifecycle
    # ------------------------------------------------------------------

    def configure(self, credentials: dict) -> None:
        """Validate, store, and test CalDAV credentials.

        Expects *credentials* to contain all three of ``provider``,
        ``username``, and ``password``.  The password is encrypted at rest via
        :meth:`~capabilities.base.AbstractCapability.store_credential`.  After
        storing, a live connection test is performed by calling
        :meth:`connect`; if the test fails a :exc:`ValueError` is raised and
        the stored credentials are removed.

        Args:
            credentials: Mapping with the following required keys:

                - ``provider`` (str): A supported provider name, e.g.
                  ``"google"``, ``"apple"``, ``"fastmail"``, ``"nextcloud"``,
                  ``"synology"``, or ``"radicale"``.
                - ``username`` (str): Account username or e-mail address.
                - ``password`` (str): App-specific or account password.

        Raises:
            ValueError: If any required key is missing, the provider is not
                recognised, or the connection test against the remote server
                fails.
        """
        # --- Validate required keys ---
        missing = [k for k in ("provider", "username", "password") if k not in credentials]
        if missing:
            raise ValueError(
                f"[caldav] configure() missing required credential fields: {missing}"
            )

        provider_name: str = credentials["provider"]
        username: str = credentials["username"]
        password: str = credentials["password"]

        # --- Validate provider is known before storing anything ---
        if resolve_provider(provider_name) is None:
            raise ValueError(
                f"[caldav] Unknown provider '{provider_name}'.  "
                f"Supported providers: google, apple, fastmail, nextcloud, synology, radicale."
            )

        # --- Persist credentials (password encrypted at rest) ---
        self.store_credential(_KEY_PROVIDER, provider_name)
        self.store_credential(_KEY_USERNAME, username)
        self.store_credential(_KEY_PASSWORD, password)

        # --- Test connectivity; roll back on failure ---
        if not self.connect():
            self.delete_credentials()
            raise ValueError(
                f"[caldav] Could not connect to provider '{provider_name}' "
                f"for user '{username}'.  Check credentials and try again."
            )

    def connect(self) -> bool:
        """Establish a connection to the CalDAV server.

        Loads stored credentials, resolves the provider's server URL, and
        performs a ``PROPFIND`` request (via :meth:`caldav.DAVClient.principal`)
        to verify that the credentials are valid.  The test uses a
        ``{_CONNECT_TIMEOUT}``-second timeout.

        Returns:
            bool: ``True`` if the connection was established successfully;
            ``False`` on any connection or authentication failure (the error is
            logged but not re-raised).
        """
        if not _CALDAV_AVAILABLE:
            logger.error(
                "[caldav] connect() called but 'caldav' package is not installed."
            )
            return False

        # --- Load stored credentials ---
        provider_name = self.load_credential(_KEY_PROVIDER)
        username = self.load_credential(_KEY_USERNAME)
        password = self.load_credential(_KEY_PASSWORD)

        if not provider_name or not username or not password:
            logger.warning(
                "[caldav] connect() aborted: one or more credentials are missing "
                "(provider=%r, username=%r, password=%s).",
                provider_name,
                username,
                "<set>" if password else "<missing>",
            )
            return False

        # --- Resolve provider config ---
        provider_config = resolve_provider(provider_name)
        if provider_config is None:
            logger.error(
                "[caldav] connect() failed: unknown provider '%s'.", provider_name
            )
            return False

        url: str = provider_config["url"]

        # --- Attempt connection ---
        try:
            client = _caldav_lib.DAVClient(
                url=url,
                username=username,
                password=password,
                timeout=_CONNECT_TIMEOUT,
            )
            client.principal()  # Performs a PROPFIND — raises on auth/network failure
            self._connected = True
            logger.info(
                "[caldav] Connected successfully (provider=%s, username=%s).",
                provider_name,
                username,
            )
            return True
        except Exception as exc:
            logger.error(
                "[caldav] connect() failed for provider=%s, username=%s: %s",
                provider_name,
                username,
                exc,
            )
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Tear down the active connection and delete stored credentials.

        Sets ``self._connected`` to ``False`` and removes all stored
        credentials from ``tool_configs``.  Does not raise.

        Returns:
            None
        """
        self._connected = False
        self.delete_credentials()
        logger.info("[caldav] Disconnected and credentials removed.")

    # ------------------------------------------------------------------
    # Stubs — implemented in subsequent tasks
    # ------------------------------------------------------------------

    def ingest(self) -> list:
        """Fetch and return calendar events from the CalDAV server.

        .. note::
            Not yet implemented.  Will be implemented in a subsequent task.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "[caldav] ingest() is not yet implemented."
        )

    def get_tools(self) -> list:
        """Return CalDAV tool definitions for dynamic registration.

        .. note::
            Not yet implemented.  Will be implemented in a subsequent task.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "[caldav] get_tools() is not yet implemented."
        )
