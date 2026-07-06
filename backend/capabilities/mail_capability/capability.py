

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import yaml

from capabilities.base import AbstractCapability
from capabilities.mail_capability.caldav_handler import CaldavHandler
from capabilities.mail_capability.carddav_handler import CarddavHandler
from capabilities.mail_capability.imap_handler import ImapHandler, SmtpCreds
from capabilities.mail_capability.providers import ServerSettings, UnifiedProvider, build_custom_provider, \
    discover_provider
from services.database import Database
from services.file_mapper_service import FileMapperService
from utils.data_utils import parse_json_column

if TYPE_CHECKING:
    from capabilities.mail_capability.caldav_handler import _CalDAVClient  # noqa: PLC2701
    from capabilities.mail_capability.imap_handler import _ImapClient  # noqa: PLC2701
    from capabilities.mail_capability.carddav_handler import _CaldavClientProto  # noqa: PLC2701

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("mail_capability", "manifest.yaml")

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

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict[str, object] | None = None
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
        return "mail"

    def get_manifest(self) -> dict[str, object]:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _resolve_provider(self, email: str) -> UnifiedProvider | None:
        provider = discover_provider(email)
        if provider is not None:
            return provider
        # Fall back to stored connection fields (custom provider)
        imap_host = self.load_credential(_K_IMAP_HOST)
        if not imap_host:
            return None
        smtp_host = self.load_credential(_K_SMTP_HOST) or None
        return build_custom_provider(
            imap_host=imap_host,
            imap_port=int(self.load_credential(_K_IMAP_PORT) or "993"),
            imap_tls=self.load_credential(_K_IMAP_TLS) != "0",
            smtp_host=smtp_host,
            smtp_port=int(self.load_credential(_K_SMTP_PORT) or "587"),
            smtp_tls=self.load_credential(_K_SMTP_TLS) == "1",
            caldav_url=self.load_credential(_K_CALDAV_URL) or None,
            carddav_url=self.load_credential(_K_CARDDAV_URL) or None,
        )

    # ------------------------------------------------------------------
    # Lifecycle — configure helpers
    # ------------------------------------------------------------------

    def _resolve_or_build_provider(self, email: str, credentials: dict[str, object]) -> UnifiedProvider:
        provider = discover_provider(email)
        if provider is not None:
            return provider

        imap_host = (cast(str, credentials.get("imap_host")) or "").strip()
        if not imap_host:
            raise ValueError(
                f"[mail] Unsupported provider for '{email}'. "
                "Supported: Google, Apple, Yahoo, Outlook. "
                "For custom servers, pass imap_host."
            )
        _imap_tls = str(credentials.get("imap_tls", "1")).lower()
        _smtp_host = (cast(str, credentials.get("smtp_host")) or "").strip() or None
        _smtp_tls = str(credentials.get("smtp_tls", "0")).lower()
        return build_custom_provider(
            imap_host=imap_host,
            imap_port=int(cast(int, credentials.get("imap_port", 993))),
            imap_tls=_imap_tls not in ("0", "false", "no"),
            smtp_host=_smtp_host,
            smtp_port=int(cast(int, credentials.get("smtp_port", 587))),
            smtp_tls=_smtp_tls not in ("0", "false", "no"),
            caldav_url=(cast(str, credentials.get("caldav_url")) or "").strip() or None,
            carddav_url=(cast(str, credentials.get("carddav_url")) or "").strip() or None,
        )

    def _persist_provider_credentials(self, email: str, password: str, provider: UnifiedProvider) -> None:
        self.store_credential(_K_EMAIL, email)
        self.store_credential(_K_PASSWORD, password)
        self.store_credential(_K_PROVIDER, provider.name)
        if provider.imap:
            self.store_credential(_K_IMAP_HOST, provider.imap.host)
            self.store_credential(_K_IMAP_PORT, str(provider.imap.port))
            self.store_credential(_K_IMAP_TLS, "1" if provider.imap.tls else "0")
        if provider.smtp:
            self.store_credential(_K_SMTP_HOST, provider.smtp.host)
            self.store_credential(_K_SMTP_PORT, str(provider.smtp.port))
            self.store_credential(_K_SMTP_TLS, "1" if provider.smtp.tls else "0")
        if provider.caldav_url:
            self.store_credential(
                _K_CALDAV_URL,
                provider.caldav_url.replace(_PLACEHOLDER_USERNAME, email),
            )
        if provider.carddav_url:
            self.store_credential(
                _K_CARDDAV_URL,
                provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email),
            )

    def _probe_protocols(self, email: str, password: str, provider: UnifiedProvider) -> list[str]:
        active: list[str] = []

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
                    active.append("imap")
                    logger.info("[mail] IMAP probe: OK")
                else:
                    logger.warning("[mail] IMAP probe: failed (open_client returned None)")
            except Exception as exc:
                logger.warning("[mail] IMAP probe: failed — %s", exc)

        if provider.caldav_url:
            caldav_url = provider.caldav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = cast("_ImapClient | None", self._caldav_handler.open_client(
                    url=caldav_url, username=email, password=password
                ))
                if client is not None:
                    active.append("caldav")
                    logger.info("[mail] CalDAV probe: OK")
                else:
                    logger.warning("[mail] CalDAV probe: failed (open_client returned None)")
            except Exception as exc:
                logger.warning("[mail] CalDAV probe: failed — %s", exc)

        if provider.carddav_url:
            carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = cast("_ImapClient | None", self._carddav_handler.open_client(
                    url=carddav_url, username=email, password=password
                ))
                if client is not None:
                    active.append("carddav")
                    logger.info("[mail] CardDAV probe: OK")
                else:
                    logger.warning("[mail] CardDAV probe: failed (open_client returned None)")
            except Exception as exc:
                logger.warning("[mail] CardDAV probe: failed — %s", exc)

        return active

    # ------------------------------------------------------------------
    # Lifecycle — configure
    # ------------------------------------------------------------------

    def configure(self, credentials: dict[str, object]) -> None:
        email = (cast(str, credentials.get("email") or credentials.get("username") or "")).strip()
        password = (cast(str, credentials.get("password")) or "").strip()
        if not email:
            raise ValueError("[mail] configure(): 'email' is required.")
        if not password:
            raise ValueError("[mail] configure(): 'password' is required.")

        provider = self._resolve_or_build_provider(email, credentials)
        self._persist_provider_credentials(email, password, provider)
        active_protocols = self._probe_protocols(email, password, provider)

        if not active_protocols:
            self.delete_credentials()
            raise ValueError(
                f"[mail] All protocol probes failed for '{email}'. "
                "Check credentials and ensure app passwords are enabled."
            )

        self.store_credential(_K_PROTOCOLS, json.dumps(active_protocols))
        logger.info("[mail] configure() — active protocols: %s", active_protocols)

        if not self.connect():
            self.delete_credentials()
            raise ValueError(
                f"[mail] Post-configure connect() failed for '{email}'."
            )

    # ------------------------------------------------------------------
    # Lifecycle — connect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        email = self.load_credential(_K_EMAIL)
        password = self.load_credential(_K_PASSWORD)
        protocols_raw = self.load_credential(_K_PROTOCOLS)

        if not email or not password:
            logger.warning("[mail] connect(): credentials missing.")
            return False

        protocols: list[str] = cast("list[str]", parse_json_column(protocols_raw, default=[]))

        if not protocols:
            logger.warning("[mail] connect(): no active protocols stored.")
            return False

        provider = self._resolve_provider(email)
        if provider is None:
            logger.error("[mail] connect(): no provider for '%s'.", email)
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
                client = cast("_ImapClient | None", self._caldav_handler.open_client(
                    url=caldav_url, username=email, password=password
                ))
                if client is not None:
                    self._caldav_ok = True
                    logger.info("[mail] CalDAV connected.")
            except Exception as exc:
                logger.warning("[mail] CalDAV connect failed: %s", exc)

        # --- CardDAV ---
        if "carddav" in protocols and provider.carddav_url:
            carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, email)
            try:
                client = cast("_ImapClient | None", self._carddav_handler.open_client(
                    url=carddav_url, username=email, password=password
                ))
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
        self._imap_ok = False
        self._caldav_ok = False
        self._carddav_ok = False
        self._connected = False
        self._cycle_count = 0

        try:
            with Database.transaction() as conn:
                conn.execute(
                    "UPDATE scheduled_items SET status='cancelled' "
                    "WHERE source='mail' AND status='pending'"
                )
            logger.info("[mail] Scheduled items cancelled.")
        except Exception as exc:
            logger.warning("[mail] disconnect cleanup: %s", exc)

        self.delete_credentials()
        logger.info("[mail] Disconnected and credentials removed.")

    # ------------------------------------------------------------------
    # Cognitive pipeline — ingest
    # ------------------------------------------------------------------

    def ingest(self) -> list[object]:
        if not self.is_connected():
            return []

        items: list[object] = []
        email = self.load_credential(_K_EMAIL)
        password = self.load_credential(_K_PASSWORD)
        provider = self._resolve_provider(email or "")
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
                    email=cast(str, email),
                    password=cast(str, password),
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
                caldav_url = provider.caldav_url.replace(_PLACEHOLDER_USERNAME, cast(str, email))
                client = cast("_ImapClient | None", self._caldav_handler.open_client(
                    url=caldav_url, username=cast(str, email), password=cast(str, password)
                ))
                if client is not None:
                    events = self._caldav_handler.ingest(cast("_CalDAVClient", client))
                    items.extend(events)
            except Exception as exc:
                logger.error("[mail] ingest() CalDAV: %s", exc)

        if self._carddav_ok and provider.carddav_url:
            try:
                carddav_url = provider.carddav_url.replace(_PLACEHOLDER_USERNAME, cast(str, email))
                client = cast("_ImapClient | None", self._carddav_handler.open_client(
                    url=carddav_url, username=cast(str, email), password=cast(str, password)
                ))
                if client is not None:
                    contacts = self._carddav_handler.sync_contacts(cast("_CaldavClientProto", client))
                    items.extend(contacts)
            except Exception as exc:
                logger.error("[mail] ingest() CardDAV: %s", exc)

        return items

    # ------------------------------------------------------------------
    # Cognitive pipeline — understand
    # ------------------------------------------------------------------

    def understand(self, items: list[object]) -> list[object]:
        if not items:
            return items
        imap_items = [cast("dict[str, object]", it) for it in items if "subject" in cast("dict[str, object]", it) and "uid" in cast("dict[str, object]", it)]
        if imap_items:
            self._imap_handler.understand(imap_items)
        return items

    # ------------------------------------------------------------------
    # Cognitive pipeline — monitor
    # ------------------------------------------------------------------

    def _monitor_imap(self, email: str, password: str, provider: UnifiedProvider) -> None:
        try:
            watermark_raw = self.load_credential(_K_WATERMARK)
            watermark = int(watermark_raw) if watermark_raw else None
            client = self._imap_handler.open_client(
                host=cast("ServerSettings", provider.imap).host,
                port=cast("ServerSettings", provider.imap).port,
                tls=cast("ServerSettings", provider.imap).tls,
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

    def _monitor_carddav(self, email: str, password: str, provider: UnifiedProvider) -> None:
        try:
            carddav_url = cast(str, provider.carddav_url).replace(_PLACEHOLDER_USERNAME, email)
            client = self._carddav_handler.open_client(
                url=carddav_url, username=email, password=password
            )
            if client is not None:
                self._carddav_handler.monitor(client)
        except Exception as exc:
            logger.error("[mail] _do_monitor() CardDAV: %s", exc)

    def _do_monitor(self) -> None:
        if not self.is_connected():
            self.connect()
        if not self.is_connected():
            return

        email = self.load_credential(_K_EMAIL)
        password = self.load_credential(_K_PASSWORD)
        provider = self._resolve_provider(email or "")
        if not provider:
            return

        if self._imap_ok and provider.imap:
            self._monitor_imap(cast(str, email), cast(str, password), provider)

        if self._carddav_ok and self._cycle_count % 12 == 0 and provider.carddav_url:
            self._monitor_carddav(cast(str, email), cast(str, password), provider)

        self._cycle_count += 1

    # ------------------------------------------------------------------
    # SMTP / guard helpers
    # ------------------------------------------------------------------

    def _load_smtp_creds(self) -> SmtpCreds:
        smtp_host = self.load_credential(_K_SMTP_HOST)
        if not smtp_host:
            raise ValueError("SMTP not configured for this provider.")
        return SmtpCreds(
            host=smtp_host,
            port=int(self.load_credential(_K_SMTP_PORT) or "587"),
            tls=self.load_credential(_K_SMTP_TLS) == "1",
            from_addr=cast(str, self.load_credential(_K_EMAIL)),
            password=cast(str, self.load_credential(_K_PASSWORD)),
        )

    # ------------------------------------------------------------------
    # Tools — connection helpers
    # ------------------------------------------------------------------

    def _open_imap_client(self) -> "_ImapClient | None":
        _email = self.load_credential(_K_EMAIL)
        _password = self.load_credential(_K_PASSWORD)
        _provider = self._resolve_provider(_email or "")
        if not _provider or not _provider.imap:
            raise ValueError("IMAP not available for this provider.")
        return self._imap_handler.open_client(
            host=_provider.imap.host,
            port=_provider.imap.port,
            tls=_provider.imap.tls,
            email=cast(str, _email),
            password=cast(str, _password),
        )

    def _open_caldav_client(self) -> "_CalDAVClient | None":
        _email = self.load_credential(_K_EMAIL)
        _password = self.load_credential(_K_PASSWORD)
        _provider = self._resolve_provider(_email or "")
        if not _provider or not _provider.caldav_url:
            raise ValueError("CalDAV not available for this provider.")
        _url = _provider.caldav_url.replace(_PLACEHOLDER_USERNAME, cast(str, _email))
        return cast("_CalDAVClient | None", self._caldav_handler.open_client(
            url=_url, username=cast(str, _email), password=cast(str, _password)
        ))

    # ------------------------------------------------------------------
    # Tools — IMAP handler methods
    # ------------------------------------------------------------------

    def _th_search_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        try:
            client = self._open_imap_client()
            if client is None:
                return {"error": "Failed to open IMAP connection."}
            try:
                return self._imap_handler.search(client, params)
            finally:
                try:
                    client.logout()
                except Exception:
                    pass
        except Exception as exc:
            return {"error": str(exc)}

    def _th_read_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        try:
            client = self._open_imap_client()
            if client is None:
                return {"error": "Failed to open IMAP connection."}
            try:
                return self._imap_handler.read_email(client, params)
            finally:
                try:
                    client.logout()
                except Exception:
                    pass
        except Exception as exc:
            return {"error": str(exc)}

    def _th_draft_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        try:
            _email = self.load_credential(_K_EMAIL)
            client = self._open_imap_client()
            if client is None:
                return {"error": "Failed to open IMAP connection."}
            try:
                return self._imap_handler.draft_email(
                    client, from_addr=cast(str, _email), params=params,
                )
            finally:
                try:
                    client.logout()
                except Exception:
                    pass
        except Exception as exc:
            return {"error": str(exc)}

    def _th_manage_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        try:
            client = self._open_imap_client()
            if client is None:
                return {"error": "Failed to open IMAP connection."}
            try:
                return self._imap_handler.manage_email(client, params)
            finally:
                try:
                    client.logout()
                except Exception:
                    pass
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tools — SMTP handler methods
    # ------------------------------------------------------------------

    def _th_send_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        try:
            creds = self._load_smtp_creds()
            return self._imap_handler.send_email(creds=creds, params=params)
        except Exception as exc:
            return {"error": str(exc)}

    def _th_reply_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        uid = params.get("uid")
        if not uid:
            return {"error": "uid is required for reply"}
        try:
            creds = self._load_smtp_creds()
            client = self._open_imap_client()
            if client is None:
                return {"error": "Failed to open IMAP connection."}
            try:
                original = self._imap_handler.read_email(client, {"uid": uid})
            finally:
                try:
                    client.logout()
                except Exception:
                    pass
            if "error" in original:
                return original
            result = self._imap_handler.send_email(
                creds=creds,
                params={
                    "to": original.get("from_addr", ""),
                    "subject": f"Re: {original.get('subject', '')}",
                    "body": params.get("body", ""),
                    "in_reply_to": original.get("message_id", ""),
                },
            )
            if result.get("success"):
                result["original"] = {
                    "from": original.get("from_addr", ""),
                    "subject": original.get("subject", ""),
                    "body": original.get("body", ""),
                    "date": original.get("date", ""),
                }
            return result
        except Exception as exc:
            return {"error": str(exc)}

    def _th_forward_email(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._imap_ok:
            return {"error": _ERR_IMAP_NOT_CONNECTED}
        uid = params.get("uid")
        if not uid:
            return {"error": "uid is required for forward"}
        to = (cast(str, params.get("to")) or "").strip()
        if not to:
            return {"error": "to is required for forward"}
        try:
            creds = self._load_smtp_creds()
            client = self._open_imap_client()
            if client is None:
                return {"error": "Failed to open IMAP connection."}
            try:
                original = self._imap_handler.read_email(client, {"uid": uid})
            finally:
                try:
                    client.logout()
                except Exception:
                    pass
            if "error" in original:
                return original
            from_name = original.get("from_name") or original.get("from_addr", "")
            forward_header = (
                f"---------- Forwarded message ----------\n"
                f"From: {from_name}\n"
                f"Date: {original.get('date', '')}\n"
                f"Subject: {original.get('subject', '')}\n\n"
            )
            body = forward_header + cast(str, original.get("body", ""))
            result = self._imap_handler.send_email(
                creds=creds,
                params={
                    "to": to,
                    "subject": f"Fwd: {original.get('subject', '')}",
                    "body": body,
                },
            )
            if result.get("success"):
                result["original"] = {
                    "from": original.get("from_addr", ""),
                    "subject": original.get("subject", ""),
                    "body": original.get("body", ""),
                    "date": original.get("date", ""),
                }
            return result
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tools — CalDAV handler methods
    # ------------------------------------------------------------------

    def _th_create_event(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._caldav_ok:
            return {"error": _ERR_CALDAV_NOT_CONNECTED}
        try:
            client = self._open_caldav_client()
            if client is None:
                return {"error": _ERR_CALDAV_OPEN_FAILED}
            return self._caldav_handler.create_event(client, params)
        except Exception as exc:
            return {"error": str(exc)}

    def _th_update_event(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._caldav_ok:
            return {"error": _ERR_CALDAV_NOT_CONNECTED}
        try:
            client = self._open_caldav_client()
            if client is None:
                return {"error": _ERR_CALDAV_OPEN_FAILED}
            return self._caldav_handler.update_event(client, params)
        except Exception as exc:
            return {"error": str(exc)}

    def _th_delete_event(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._caldav_ok:
            return {"error": _ERR_CALDAV_NOT_CONNECTED}
        try:
            client = self._open_caldav_client()
            if client is None:
                return {"error": _ERR_CALDAV_OPEN_FAILED}
            return self._caldav_handler.delete_event(client, params)
        except Exception as exc:
            return {"error": str(exc)}

    def _th_find_free_slots(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._caldav_ok:
            return {"error": _ERR_CALDAV_NOT_CONNECTED}
        return self._caldav_handler.find_free_slots(params)

    def _th_get_attendees(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._caldav_ok:
            return {"error": _ERR_CALDAV_NOT_CONNECTED}
        return self._caldav_handler.get_attendees(params)

    # ------------------------------------------------------------------
    # Tools — CardDAV handler methods
    # ------------------------------------------------------------------

    def _th_list_contacts(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._carddav_ok:
            return {"error": "Mail (CardDAV) not connected."}
        return self._carddav_handler.list_contacts(params)

    def _th_get_contact(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        if not self._carddav_ok:
            return {"error": "Mail (CardDAV) not connected."}
        return self._carddav_handler.get_contact(params)

    # ------------------------------------------------------------------
    # Tools — builder helpers
    # ------------------------------------------------------------------

    def _build_smtp_tools(self) -> list[dict[str, object]]:
        smtp_host = self.load_credential(_K_SMTP_HOST)
        if not smtp_host:
            return []
        return [
            {
                "name": "send_email",
                "description": "Send an email via SMTP.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email address"},
                        "subject": {"type": "string", "description": "Email subject line"},
                        "body": {"type": "string", "description": "Plain-text email body"},
                        "cc": {"type": "string", "description": "CC recipient email address"},
                        "in_reply_to": {
                            "type": "string",
                            "description": "Message-ID for threading this as a reply",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
                "handler": self._th_send_email,
                "timeout": 30,
            },
            {
                "name": "reply_email",
                "description": (
                    "Reply to an email by UID. Reads the original email, "
                    "then sends the reply. Returns the original email content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "integer", "description": "IMAP UID of the email to reply to"},
                        "body": {"type": "string", "description": "Plain-text reply body"},
                    },
                    "required": ["uid", "body"],
                },
                "handler": self._th_reply_email,
                "timeout": 30,
            },
            {
                "name": "forward_email",
                "description": (
                    "Forward an email by UID to a new recipient. Reads the "
                    "original email, then sends it. Returns the original content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "integer", "description": "IMAP UID of the email to forward"},
                        "to": {"type": "string", "description": "Recipient email address"},
                    },
                    "required": ["uid", "to"],
                },
                "handler": self._th_forward_email,
                "timeout": 30,
            },
        ]

    def _build_imap_tools(self) -> list[dict[str, object]]:
        return self._build_smtp_tools() + [
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
                "handler": self._th_search_email,
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
                "handler": self._th_read_email,
                "timeout": 30,
            },
            {
                "name": "draft_email",
                "description": "Create a draft email in the user's Drafts folder via IMAP.",
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
                "handler": self._th_draft_email,
                "timeout": 15,
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
                "handler": self._th_manage_email,
                "timeout": 15,
            },
        ]

    def _build_caldav_tools(self) -> list[dict[str, object]]:
        return [
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
                "handler": self._th_create_event,
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
                "handler": self._th_update_event,
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
                "handler": self._th_delete_event,
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
                "handler": self._th_find_free_slots,
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
                "handler": self._th_get_attendees,
                "timeout": 15,
            },
        ]

    def _build_carddav_tools(self) -> list[dict[str, object]]:
        return [
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
                "handler": self._th_list_contacts,
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
                "handler": self._th_get_contact,
                "timeout": 15,
            },
        ]

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = []

        if self._imap_ok:
            tools += self._build_imap_tools()

        if self._caldav_ok:
            tools += self._build_caldav_tools()

        if self._carddav_ok:
            tools += self._build_carddav_tools()

        return tools
