"""MailCapability — unified IMAP/SMTP + CalDAV + CardDAV capability.

:class:`MailCapability` ties :class:`~capabilities.mail_capability.imap_handler.ImapHandler`,
:class:`~capabilities.mail_capability.caldav_handler.CaldavHandler`, and
:class:`~capabilities.mail_capability.carddav_handler.CarddavHandler` together
into a single :class:`~capabilities.base.AbstractCapability` subclass.

Protocol probing
----------------
:meth:`MailCapability.configure` probes each protocol independently; any subset
may succeed.  The set of active protocols is persisted as ``mail:protocols`` so
that subsequent :meth:`MailCapability.connect` calls never attempt a protocol
that is unsupported by the provider (e.g. Outlook has no CalDAV/CardDAV).

Monitor cadence
---------------
The base cycle is 5 minutes (``interval:5``).  IMAP runs every cycle;
CalDAV every 3rd cycle (~15 min); CardDAV every 12th cycle (~60 min).

Credential storage
------------------
All credentials are persisted under ``tool_name='mail'`` in ``tool_configs``.
Config keys follow the ``mail:{field}`` convention:

- ``mail:email``          — account email address
- ``mail:password``       — app password (encrypted at rest)
- ``mail:provider_name``  — resolved provider name (e.g. "Google")
- ``mail:imap_host``      — IMAP server hostname
- ``mail:imap_port``      — IMAP port number
- ``mail:imap_tls``       — "1" if TLS, "0" otherwise
- ``mail:smtp_host``      — SMTP server hostname
- ``mail:smtp_port``      — SMTP port number
- ``mail:smtp_tls``       — "1" if TLS, "0" otherwise
- ``mail:caldav_url``     — CalDAV endpoint URL (absent if provider has none)
- ``mail:carddav_url``    — CardDAV endpoint URL (absent if provider has none)
- ``mail:protocols``      — JSON list of active protocols, e.g. ["imap","caldav"]
- ``mail:imap_watermark`` — last seen IMAP UID
"""

from __future__ import annotations

import json
import logging
import pathlib

import yaml

from capabilities.base import AbstractCapability
from capabilities.mail_capability.caldav_handler import CaldavHandler
from capabilities.mail_capability.carddav_handler import CarddavHandler
from capabilities.mail_capability.imap_handler import ImapHandler
from capabilities.mail_capability.providers import discover_provider
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

# ---------------------------------------------------------------------------
# Credential key names (unified mail namespace)
# ---------------------------------------------------------------------------

_PLACEHOLDER_USERNAME = "{username}"
_ERR_IMAP_NOT_CONNECTED = "Mail (IMAP) not connected."
_ERR_CALDAV_NOT_CONNECTED = "Mail (CalDAV) not connected."
_ERR_CALDAV_OPEN_FAILED = "Failed to open CalDAV connection."
_DESC_CALDAV_UID = "CalDAV event UID"

_K_EMAIL = "mail:email"
_K_PASSWORD = "mail:password"
_K_PROVIDER = "mail:provider_name"
_K_IMAP_HOST = "mail:imap_host"
_K_IMAP_PORT = "mail:imap_port"
_K_IMAP_TLS = "mail:imap_tls"
_K_SMTP_HOST = "mail:smtp_host"
_K_SMTP_PORT = "mail:smtp_port"
_K_SMTP_TLS = "mail:smtp_tls"
_K_CALDAV_URL = "mail:caldav_url"
_K_CARDDAV_URL = "mail:carddav_url"
_K_PROTOCOLS = "mail:protocols"
_K_IMAP_WATERMARK = "mail:imap_watermark"
_K_WATERMARK = _K_IMAP_WATERMARK  # alias used by MailCapability


# ---------------------------------------------------------------------------
# MailCapability


class MailCapability(AbstractCapability):
    """Unified email + calendar + contacts capability.

    Orchestrates :class:`ImapHandler`, :class:`CaldavHandler`, and
    :class:`CarddavHandler`.  Each protocol is probed independently during
    :meth:`configure`; only the successful subset is activated.

    Attributes:
        _imap_ok (bool): IMAP protocol is connected and operational.
        _caldav_ok (bool): CalDAV protocol is connected and operational.
        _carddav_ok (bool): CardDAV protocol is connected and operational.
        _cycle_count (int): Monitor cycles since last successful :meth:`connect`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict | None = None
        self._imap_handler = ImapHandler()
        self._caldav_handler = CaldavHandler()
        self._carddav_handler = CarddavHandler()
        self._imap_ok: bool = False
        self._caldav_ok: bool = False
        self._carddav_ok: bool = False
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_id(self) -> str:
        """Return the capability identifier.

        Returns:
            str: Always ``"mail"``.
        """
        return "mail"

    def get_manifest(self) -> dict:
        """Return the parsed ``manifest.yaml`` contents (cached after first load).

        Returns:
            dict: Manifest dict including ``id``, ``name``, ``version``,
            ``entry_class``.
        """
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

    # ------------------------------------------------------------------
    # Lifecycle — configure
    # ------------------------------------------------------------------

    def configure(self, credentials: dict) -> None:
        """Validate credentials, probe each protocol, and persist active ones.

        Args:
            credentials: Must contain ``email`` (str) and ``password`` (str).

        Raises:
            ValueError: If email or password is missing, the provider is not
                supported, or every protocol probe fails.
        """
        email = (credentials.get("email") or credentials.get("username") or "").strip()
        password = (credentials.get("password") or "").strip()
        if not email:
            raise ValueError("[mail] configure(): 'email' is required.")
        if not password:
            raise ValueError("[mail] configure(): 'password' is required.")

        provider = discover_provider(email)
        if provider is None:
            raise ValueError(
                f"[mail] Unsupported provider for '{email}'. "
                "Supported: Google, Apple, Yahoo, Outlook."
            )

        # Persist base credentials
        self.store_credential(_K_EMAIL, email)
        self.store_credential(_K_PASSWORD, password)
        self.store_credential(_K_PROVIDER, provider.name)

        # Persist IMAP/SMTP connection settings
        if provider.imap:
            self.store_credential(_K_IMAP_HOST, provider.imap.host)
            self.store_credential(_K_IMAP_PORT, str(provider.imap.port))
            self.store_credential(_K_IMAP_TLS, "1" if provider.imap.tls else "0")
        if provider.smtp:
            self.store_credential(_K_SMTP_HOST, provider.smtp.host)
            self.store_credential(_K_SMTP_PORT, str(provider.smtp.port))
            self.store_credential(_K_SMTP_TLS, "1" if provider.smtp.tls else "0")

        # Probe each protocol independently
        active_protocols: list[str] = []

        # --- IMAP ---
        if provider.imap:
            try:
                client = self._imap_handler.open_client(
                    host=provider.imap.host,
                    port=provider.imap.port,
                    tls=provider.imap.tls,
                    email=email,
                    password=password,
                )
                if client is not None:
                    try:
                        client.logout()
                    except Exception:
                        pass
                    active_protocols.append("imap")
                    logger.info("[mail] IMAP probe: OK")
                else:
                    logger.warning("[mail] IMAP probe: failed (open_client returned None)")
            except Exception as exc:
                logger.warning("[mail] IMAP probe: failed — %s", exc)

        # --- CalDAV ---
        if provider.caldav_url:
            caldav_url = provider.caldav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = self._caldav_handler.open_client(
                    url=caldav_url, username=email, password=password
                )
                if client is not None:
                    active_protocols.append("caldav")
                    logger.info("[mail] CalDAV probe: OK")
                else:
                    logger.warning("[mail] CalDAV probe: failed (open_client returned None)")
            except Exception as exc:
                logger.warning("[mail] CalDAV probe: failed — %s", exc)

        # --- CardDAV ---
        if provider.carddav_url:
            carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = self._carddav_handler.open_client(
                    url=carddav_url, username=email, password=password
                )
                if client is not None:
                    active_protocols.append("carddav")
                    logger.info("[mail] CardDAV probe: OK")
                else:
                    logger.warning("[mail] CardDAV probe: failed (open_client returned None)")
            except Exception as exc:
                logger.warning("[mail] CardDAV probe: failed — %s", exc)

        if not active_protocols:
            self.delete_credentials()
            raise ValueError(
                f"[mail] All protocol probes failed for '{email}'. "
                "Check credentials and ensure app passwords are enabled."
            )

        self.store_credential(_K_PROTOCOLS, json.dumps(active_protocols))
        logger.info("[mail] configure() — active protocols: %s", active_protocols)

        # Final round-trip test via connect()
        if not self.connect():
            self.delete_credentials()
            raise ValueError(
                f"[mail] Post-configure connect() failed for '{email}'."
            )

    # ------------------------------------------------------------------
    # Lifecycle — connect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish connections for all stored active protocols.

        Loads credentials, probes only the protocols listed in
        ``mail:protocols``, and sets per-protocol flags.

        Returns:
            bool: ``True`` if at least one protocol connected successfully.
        """
        email = self.load_credential(_K_EMAIL)
        password = self.load_credential(_K_PASSWORD)
        protocols_raw = self.load_credential(_K_PROTOCOLS)

        if not email or not password:
            logger.warning("[mail] connect(): credentials missing.")
            return False

        try:
            protocols: list[str] = json.loads(protocols_raw) if protocols_raw else []
        except (json.JSONDecodeError, TypeError):
            protocols = []

        if not protocols:
            logger.warning("[mail] connect(): no active protocols stored.")
            return False

        provider = discover_provider(email)
        if provider is None:
            logger.error("[mail] connect(): unsupported provider for '%s'.", email)
            return False

        self._imap_ok = False
        self._caldav_ok = False
        self._carddav_ok = False

        # --- IMAP ---
        if "imap" in protocols and provider.imap:
            try:
                client = self._imap_handler.open_client(
                    host=provider.imap.host,
                    port=provider.imap.port,
                    tls=provider.imap.tls,
                    email=email,
                    password=password,
                )
                if client is not None:
                    try:
                        client.logout()
                    except Exception:
                        pass
                    self._imap_ok = True
                    logger.info("[mail] IMAP connected.")
            except Exception as exc:
                logger.warning("[mail] IMAP connect failed: %s", exc)

        # --- CalDAV ---
        if "caldav" in protocols and provider.caldav_url:
            caldav_url = provider.caldav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = self._caldav_handler.open_client(
                    url=caldav_url, username=email, password=password
                )
                if client is not None:
                    self._caldav_ok = True
                    logger.info("[mail] CalDAV connected.")
            except Exception as exc:
                logger.warning("[mail] CalDAV connect failed: %s", exc)

        # --- CardDAV ---
        if "carddav" in protocols and provider.carddav_url:
            carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = self._carddav_handler.open_client(
                    url=carddav_url, username=email, password=password
                )
                if client is not None:
                    self._carddav_ok = True
                    logger.info("[mail] CardDAV connected.")
            except Exception as exc:
                logger.warning("[mail] CardDAV connect failed: %s", exc)

        any_ok = self._imap_ok or self._caldav_ok or self._carddav_ok
        self._connected = any_ok

        return any_ok

    # ------------------------------------------------------------------
    # Lifecycle — disconnect
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Tear down all connections, cancel pending scheduled items, and delete credentials."""
        self._imap_ok = False
        self._caldav_ok = False
        self._carddav_ok = False
        self._connected = False

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    "UPDATE scheduled_items SET status='cancelled' "
                    "WHERE source='mail' AND status='pending'"
                )
                conn.commit()
            logger.info("[mail] Scheduled items cancelled.")
        except Exception as exc:
            logger.warning("[mail] disconnect cleanup: %s", exc)

        self.delete_credentials()
        logger.info("[mail] Disconnected and credentials removed.")

    # ------------------------------------------------------------------
    # Cognitive pipeline — ingest
    # ------------------------------------------------------------------

    def ingest(self) -> list:
        """Fetch raw items from all connected protocols.

        Returns items from IMAP (header dicts), CalDAV (event dicts), and
        CardDAV (contact dicts) combined into a single list.  Returns ``[]``
        when not connected.

        Returns:
            list[dict]: Combined raw items from all active protocols.
        """
        if not self.is_connected():
            return []

        items: list[dict] = []
        email = self.load_credential(_K_EMAIL)
        password = self.load_credential(_K_PASSWORD)
        provider = discover_provider(email or "")
        if not provider:
            return []

        if self._imap_ok and provider.imap:
            try:
                watermark_raw = self.load_credential(_K_WATERMARK)
                watermark = int(watermark_raw) if watermark_raw else None
                client = self._imap_handler.open_client(
                    host=provider.imap.host,
                    port=provider.imap.port,
                    tls=provider.imap.tls,
                    email=email,
                    password=password,
                )
                if client is not None:
                    try:
                        new_items, new_wm = self._imap_handler.ingest(client, watermark)
                        items.extend(new_items)
                        if new_wm and new_wm != watermark:
                            self.store_credential(_K_WATERMARK, str(new_wm))
                    finally:
                        try:
                            client.logout()
                        except Exception:
                            pass
            except Exception as exc:
                logger.error("[mail] ingest() IMAP: %s", exc)

        if self._caldav_ok and provider.caldav_url:
            try:
                caldav_url = provider.caldav_url.replace(_PLACEHOLDER_USERNAME, email)
                client = self._caldav_handler.open_client(
                    url=caldav_url, username=email, password=password
                )
                if client is not None:
                    events = self._caldav_handler.ingest(client)
                    items.extend(events)
            except Exception as exc:
                logger.error("[mail] ingest() CalDAV: %s", exc)

        if self._carddav_ok and provider.carddav_url:
            try:
                carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email)
                client = self._carddav_handler.open_client(
                    url=carddav_url, username=email, password=password
                )
                if client is not None:
                    contacts = self._carddav_handler.sync_contacts(client)
                    items.extend(contacts)
            except Exception as exc:
                logger.error("[mail] ingest() CardDAV: %s", exc)

        return items

    # ------------------------------------------------------------------
    # Cognitive pipeline — understand
    # ------------------------------------------------------------------

    def understand(self, items: list) -> list:
        """Classify and enrich raw items from :meth:`ingest`.

        Applies IMAP triage and sender indexing to email headers.
        CalDAV and CardDAV items are already processed inline by their handlers.

        Args:
            items: Raw items returned by :meth:`ingest`.

        Returns:
            list[dict]: Items with triage/enrichment applied where applicable.
        """
        if not items:
            return items
        imap_items = [it for it in items if "subject" in it and "uid" in it]
        if imap_items:
            self._imap_handler.understand(imap_items)
        return items

    # ------------------------------------------------------------------
    # Cognitive pipeline — monitor
    # ------------------------------------------------------------------

    def _do_monitor(self) -> None:
        """Run one monitor cycle with per-protocol cadence management.

        Cadence:
        - IMAP: every cycle (~5 min)
        - CalDAV: every 3rd cycle (~15 min)
        - CardDAV: every 12th cycle (~60 min)

        Auto-reconnects when not connected before running.
        """
        if not self.is_connected():
            self.connect()
        if not self.is_connected():
            return

        self._cycle_count += 1
        now = utc_now()

        email = self.load_credential(_K_EMAIL)
        password = self.load_credential(_K_PASSWORD)
        provider = discover_provider(email or "")
        if not provider:
            return

        # --- IMAP: every cycle ---
        if self._imap_ok and provider.imap:
            try:
                watermark_raw = self.load_credential(_K_WATERMARK)
                watermark = int(watermark_raw) if watermark_raw else None
                client = self._imap_handler.open_client(
                    host=provider.imap.host,
                    port=provider.imap.port,
                    tls=provider.imap.tls,
                    email=email,
                    password=password,
                )
                if client is not None:
                    try:
                        new_items, new_wm = self._imap_handler.ingest(client, watermark)
                        if new_items:
                            self._imap_handler.understand(new_items)
                        if new_wm and new_wm != watermark:
                            self.store_credential(_K_WATERMARK, str(new_wm))
                        self._imap_handler.inject_inbox_hint(client)
                    finally:
                        try:
                            client.logout()
                        except Exception:
                            pass
            except Exception as exc:
                logger.error("[mail] _do_monitor() IMAP: %s", exc)

        # --- CalDAV: every 3rd cycle ---
        if self._caldav_ok and self._cycle_count % 3 == 0 and provider.caldav_url:
            try:
                caldav_url = provider.caldav_url.replace(_PLACEHOLDER_USERNAME, email)
                client = self._caldav_handler.open_client(
                    url=caldav_url, username=email, password=password
                )
                if client is not None:
                    events = self._caldav_handler.ingest(client)
                    self._caldav_handler.upsert_events(events, now)
            except Exception as exc:
                logger.error("[mail] _do_monitor() CalDAV: %s", exc)

        # --- CardDAV: every 12th cycle ---
        if self._carddav_ok and self._cycle_count % 12 == 0 and provider.carddav_url:
            try:
                carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email)
                client = self._carddav_handler.open_client(
                    url=carddav_url, username=email, password=password
                )
                if client is not None:
                    self._carddav_handler.monitor(client)
            except Exception as exc:
                logger.error("[mail] _do_monitor() CardDAV: %s", exc)

    # ------------------------------------------------------------------
    # Cognitive pipeline — act
    # ------------------------------------------------------------------

    def act(self, action: str, params: dict) -> dict:
        """Dispatch a write action to the appropriate protocol handler.

        Args:
            action: Tool name, e.g. ``"send_email"``, ``"create_event"``.
            params: Action-specific parameters.

        Returns:
            dict: Result from the handler, or ``{"error": ...}`` if the action
            is unknown.
        """
        tool_map = {t["name"]: t["handler"] for t in self.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        return handler(topic="", params=params)

    # ------------------------------------------------------------------
    # Scheduler registration
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> list:
        """Return tool definitions conditioned on which protocols are connected.

        IMAP tools (when ``_imap_ok``):
            ``search_email``, ``read_email``, ``send_email``, ``manage_email``

        CalDAV tools (when ``_caldav_ok``):
            ``list_events``, ``get_event``, ``create_event``, ``update_event``,
            ``delete_event``, ``find_free_slots``, ``get_attendees``

        CardDAV tools (when ``_carddav_ok``):
            ``list_contacts``, ``get_contact``

        Returns:
            list[dict]: Tool definition dicts with ``name``, ``description``,
            ``parameters``, ``handler``, and ``timeout`` keys.
        """
        tools: list[dict] = []
        cap = self  # closure capture

        # ------------------------------------------------------------------
        # Connection helpers
        # ------------------------------------------------------------------

        def _open_imap_client():
            _email = cap.load_credential(_K_EMAIL)
            _password = cap.load_credential(_K_PASSWORD)
            _provider = discover_provider(_email or "")
            if not _provider or not _provider.imap:
                raise ValueError("IMAP not available for this provider.")
            return cap._imap_handler.open_client(
                host=_provider.imap.host,
                port=_provider.imap.port,
                tls=_provider.imap.tls,
                email=_email,
                password=_password,
            )

        def _open_caldav_client():
            _email = cap.load_credential(_K_EMAIL)
            _password = cap.load_credential(_K_PASSWORD)
            _provider = discover_provider(_email or "")
            if not _provider or not _provider.caldav_url:
                raise ValueError("CalDAV not available for this provider.")
            _url = _provider.caldav_url.replace(_PLACEHOLDER_USERNAME, _email)
            return cap._caldav_handler.open_client(
                url=_url, username=_email, password=_password
            )

        # ------------------------------------------------------------------
        # IMAP tools
        # ------------------------------------------------------------------

        if self._imap_ok:
            def _search_email(topic, params, config=None, telemetry=None):
                if not cap._imap_ok:
                    return {"error": _ERR_IMAP_NOT_CONNECTED}
                try:
                    client = _open_imap_client()
                    if client is None:
                        return {"error": "Failed to open IMAP connection."}
                    try:
                        return cap._imap_handler.search(client, params)
                    finally:
                        try:
                            client.logout()
                        except Exception:
                            pass
                except Exception as exc:
                    return {"error": str(exc)}

            def _read_email(topic, params, config=None, telemetry=None):
                if not cap._imap_ok:
                    return {"error": _ERR_IMAP_NOT_CONNECTED}
                try:
                    client = _open_imap_client()
                    if client is None:
                        return {"error": "Failed to open IMAP connection."}
                    try:
                        return cap._imap_handler.read_email(client, params)
                    finally:
                        try:
                            client.logout()
                        except Exception:
                            pass
                except Exception as exc:
                    return {"error": str(exc)}

            def _send_email(topic, params, config=None, telemetry=None):
                if not cap._imap_ok:
                    return {"error": _ERR_IMAP_NOT_CONNECTED}
                try:
                    _email = cap.load_credential(_K_EMAIL)
                    _password = cap.load_credential(_K_PASSWORD)
                    _provider = discover_provider(_email or "")
                    if not _provider or not _provider.smtp:
                        return {"error": "SMTP provider not available."}
                    return cap._imap_handler.send_email(
                        smtp_host=_provider.smtp.host,
                        smtp_port=_provider.smtp.port,
                        smtp_tls=_provider.smtp.tls,
                        email=_email,
                        password=_password,
                        params=params,
                    )
                except Exception as exc:
                    return {"error": str(exc)}

            def _manage_email(topic, params, config=None, telemetry=None):
                if not cap._imap_ok:
                    return {"error": _ERR_IMAP_NOT_CONNECTED}
                try:
                    client = _open_imap_client()
                    if client is None:
                        return {"error": "Failed to open IMAP connection."}
                    try:
                        return cap._imap_handler.manage_email(client, params)
                    finally:
                        try:
                            client.logout()
                        except Exception:
                            pass
                except Exception as exc:
                    return {"error": str(exc)}

            tools += [
                {
                    "name": "search_email",
                    "description": (
                        "Search emails in INBOX by sender, subject, keyword, "
                        "date range, or triage category."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sender": {"type": "string", "description": "Filter by sender address"},
                            "subject": {"type": "string", "description": "Filter by subject keyword"},
                            "keyword": {"type": "string", "description": "Full-text keyword search"},
                            "date_from": {"type": "string", "description": "ISO date lower bound (YYYY-MM-DD)"},
                            "date_to": {"type": "string", "description": "ISO date upper bound (YYYY-MM-DD)"},
                            "triage": {
                                "type": "string",
                                "description": "Filter by triage category: actionable, informational, or noise",
                            },
                            "unanswered": {"type": "boolean", "description": "Only unanswered emails"},
                            "limit": {"type": "integer", "description": "Max results (default 20, max 100)"},
                        },
                    },
                    "handler": _search_email,
                    "timeout": 30,
                },
                {
                    "name": "read_email",
                    "description": "Fetch the full plain-text body of an email by IMAP UID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "integer", "description": "IMAP UID of the email"},
                        },
                        "required": ["uid"],
                    },
                    "handler": _read_email,
                    "timeout": 30,
                },
                {
                    "name": "send_email",
                    "description": "Send an email via SMTP.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Recipient email address"},
                            "subject": {"type": "string", "description": "Email subject line"},
                            "body": {"type": "string", "description": "Plain-text email body"},
                            "in_reply_to": {
                                "type": "string",
                                "description": "Message-ID for threading this as a reply",
                            },
                        },
                        "required": ["to", "subject", "body"],
                    },
                    "handler": _send_email,
                    "timeout": 30,
                },
                {
                    "name": "manage_email",
                    "description": "Manage an email: delete, mark as read, mark as important, or move to spam.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "integer", "description": "IMAP UID of the email"},
                            "operation": {
                                "type": "string",
                                "enum": ["delete", "mark_read", "mark_important", "move_to_spam"],
                                "description": "The management operation to perform.",
                            },
                        },
                        "required": ["uid", "operation"],
                    },
                    "handler": _manage_email,
                    "timeout": 15,
                },
            ]

        # ------------------------------------------------------------------
        # CalDAV tools
        # ------------------------------------------------------------------

        if self._caldav_ok:
            def _create_event(topic, params, config=None, telemetry=None):
                if not cap._caldav_ok:
                    return {"error": _ERR_CALDAV_NOT_CONNECTED}
                try:
                    client = _open_caldav_client()
                    if client is None:
                        return {"error": _ERR_CALDAV_OPEN_FAILED}
                    return cap._caldav_handler.create_event(client, params)
                except Exception as exc:
                    return {"error": str(exc)}

            def _update_event(topic, params, config=None, telemetry=None):
                if not cap._caldav_ok:
                    return {"error": _ERR_CALDAV_NOT_CONNECTED}
                try:
                    client = _open_caldav_client()
                    if client is None:
                        return {"error": _ERR_CALDAV_OPEN_FAILED}
                    return cap._caldav_handler.update_event(client, params)
                except Exception as exc:
                    return {"error": str(exc)}

            def _delete_event(topic, params, config=None, telemetry=None):
                if not cap._caldav_ok:
                    return {"error": _ERR_CALDAV_NOT_CONNECTED}
                try:
                    client = _open_caldav_client()
                    if client is None:
                        return {"error": _ERR_CALDAV_OPEN_FAILED}
                    return cap._caldav_handler.delete_event(client, params)
                except Exception as exc:
                    return {"error": str(exc)}

            def _find_free_slots(topic, params, config=None, telemetry=None):
                if not cap._caldav_ok:
                    return {"error": _ERR_CALDAV_NOT_CONNECTED}
                return cap._caldav_handler.find_free_slots(params)

            def _get_attendees(topic, params, config=None, telemetry=None):
                if not cap._caldav_ok:
                    return {"error": _ERR_CALDAV_NOT_CONNECTED}
                return cap._caldav_handler.get_attendees(params)

            tools += [
                {
                    "name": "create_event",
                    "description": "Create a new calendar event on the CalDAV server.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Event title"},
                            "dtstart": {"type": "string", "description": "Start (ISO 8601 UTC)"},
                            "dtend": {"type": "string", "description": "End (ISO 8601 UTC)"},
                            "location": {"type": "string", "description": "Optional location"},
                            "description": {"type": "string", "description": "Optional description"},
                            "calendar_name": {"type": "string", "description": "Target calendar name"},
                        },
                        "required": ["summary", "dtstart", "dtend"],
                    },
                    "handler": _create_event,
                    "timeout": 30,
                },
                {
                    "name": "update_event",
                    "description": "Update an existing calendar event by UID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "string", "description": _DESC_CALDAV_UID},
                            "summary": {"type": "string", "description": "New title"},
                            "dtstart": {"type": "string", "description": "New start (ISO 8601 UTC)"},
                            "dtend": {"type": "string", "description": "New end (ISO 8601 UTC)"},
                            "location": {"type": "string", "description": "New location"},
                            "description": {"type": "string", "description": "New description"},
                        },
                        "required": ["uid"],
                    },
                    "handler": _update_event,
                    "timeout": 30,
                },
                {
                    "name": "delete_event",
                    "description": "Delete a calendar event from the CalDAV server by UID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "string", "description": "CalDAV event UID to delete"},
                        },
                        "required": ["uid"],
                    },
                    "handler": _delete_event,
                    "timeout": 30,
                },
                {
                    "name": "find_free_slots",
                    "description": "Find free time slots within working hours.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date_from": {"type": "string", "description": "Window start (ISO 8601)"},
                            "date_to": {"type": "string", "description": "Window end (ISO 8601)"},
                            "min_duration_minutes": {
                                "type": "integer",
                                "description": "Minimum slot length in minutes (default 30)",
                            },
                            "working_hours_start": {
                                "type": "integer",
                                "description": "Working day start hour 0-23 (default 8)",
                            },
                            "working_hours_end": {
                                "type": "integer",
                                "description": "Working day end hour 0-23 (default 18)",
                            },
                        },
                    },
                    "handler": _find_free_slots,
                    "timeout": 15,
                },
                {
                    "name": "get_attendees",
                    "description": "Return resolved attendee details for a calendar event.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "string", "description": _DESC_CALDAV_UID},
                        },
                        "required": ["uid"],
                    },
                    "handler": _get_attendees,
                    "timeout": 15,
                },
            ]

        # ------------------------------------------------------------------
        # CardDAV tools
        # ------------------------------------------------------------------

        if self._carddav_ok:
            def _list_contacts(topic, params, config=None, telemetry=None):
                if not cap._carddav_ok:
                    return {"error": "Mail (CardDAV) not connected."}
                return cap._carddav_handler.list_contacts(params)

            def _get_contact(topic, params, config=None, telemetry=None):
                if not cap._carddav_ok:
                    return {"error": "Mail (CardDAV) not connected."}
                return cap._carddav_handler.get_contact(params)

            tools += [
                {
                    "name": "list_contacts",
                    "description": "List contacts from the knowledge store, optionally filtered by query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Free-text search term"},
                            "limit": {"type": "integer", "description": "Max results (default 20, max 50)"},
                        },
                    },
                    "handler": _list_contacts,
                    "timeout": 15,
                },
                {
                    "name": "get_contact",
                    "description": "Look up a single contact by email address or display name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identifier": {
                                "type": "string",
                                "description": "Email address or display name to look up",
                            },
                        },
                        "required": ["identifier"],
                    },
                    "handler": _get_contact,
                    "timeout": 15,
                },
            ]

        return tools

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_details(self) -> dict:
        """Return detailed health state including per-protocol flags.

        Returns:
            dict: ``{"connected": bool, "protocols": {"imap": bool, "caldav": bool, "carddav": bool}}``.
        """
        return {
            "connected": self.is_connected(),
            "protocols": {
                "imap": self._imap_ok,
                "caldav": self._caldav_ok,
                "carddav": self._carddav_ok,
            },
        }
