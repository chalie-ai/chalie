"""ImapCapability — IMAP email capability.

Connects to IMAP servers, fetches and normalizes email headers.
Uses provider_autodiscovery for settings.  imapclient imported lazily.
"""

from __future__ import annotations

import email as _email_mod
import email.policy
import email.utils
import logging
import pathlib
import re
from datetime import timedelta

import yaml

from capabilities.base import AbstractCapability
from capabilities.provider_autodiscovery import discover_email_settings
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

_K_EMAIL = "imap:email"
_K_PW = "imap:password"
_K_HOST = "imap:host"
_K_PORT = "imap:port"
_K_TLS = "imap:tls"
_K_WATERMARK = "imap:last_uid"
_K_SMTP_HOST = "smtp:host"
_K_SMTP_PORT = "smtp:port"
_K_SMTP_TLS = "smtp:tls"

_INITIAL_DAYS = 7

# ---------------------------------------------------------------------------
# Deterministic triage heuristics
# ---------------------------------------------------------------------------

_NOISE_ADDR_RE = re.compile(
    r"(noreply|no-reply|donotreply|notifications?|mailer-daemon|bounce|"
    r"marketing|newsletter|promo|updates?@)",
    re.IGNORECASE,
)

_ACTIONABLE_SUBJ_RE = re.compile(
    r"(deadline|confirm|approve|invoice|urgent|action.?required|please\s+re(ply|view|spond)|"
    r"payment|overdue|expir|reminder|required|asap|important|request|sign\b|"
    r"rsvp|accept|decline|awaiting)",
    re.IGNORECASE,
)


def classify_email(item: dict) -> str:
    """Classify an email header dict as noise, informational, or actionable."""
    if item.get("has_unsubscribe"):
        return "noise"
    addr = item.get("from_addr", "")
    if _NOISE_ADDR_RE.search(addr):
        return "noise"
    subj = item.get("subject", "")
    if item.get("in_reply_to"):
        return "actionable"
    if _ACTIONABLE_SUBJ_RE.search(subj):
        return "actionable"
    return "informational"


def _hdr(msg, name):
    """Return header *name* as a string, or empty string if absent."""
    v = msg.get(name)
    return str(v) if v else ""


def _safe_date(s):
    """Parse RFC 2822 date *s* into an ISO string, or empty string."""
    if not s:
        return ""
    try:
        return _email_mod.utils.parsedate_to_datetime(s).isoformat()
    except Exception:
        return ""


def parse_headers(uid: int, header_bytes: bytes) -> dict:
    """Parse raw email header bytes into a normalized dict."""
    msg = _email_mod.message_from_bytes(header_bytes, policy=_email_mod.policy.default)
    from_name, from_addr = _email_mod.utils.parseaddr(_hdr(msg, "From"))
    return {
        "uid": uid,
        "message_id": _hdr(msg, "Message-ID"),
        "in_reply_to": _hdr(msg, "In-Reply-To"),
        "subject": _hdr(msg, "Subject"),
        "from_name": from_name,
        "from_addr": from_addr,
        "to": _hdr(msg, "To"),
        "date": _safe_date(_hdr(msg, "Date")),
        "has_unsubscribe": "List-Unsubscribe" in msg,
    }


class ImapCapability(AbstractCapability):

    def __init__(self):
        super().__init__()
        self._mcache = None

    def get_id(self):
        return "imap"

    def get_manifest(self):
        if not self._mcache:
            with open(_MANIFEST_PATH) as f:
                self._mcache = yaml.safe_load(f)
        return self._mcache

    def configure(self, credentials):
        e, pw = credentials.get("email"), credentials.get("password")
        if not all([e, pw]):
            raise ValueError("[imap] email and password required")
        s = discover_email_settings(e)
        if not s:
            raise ValueError(f"[imap] Unsupported email provider for '{e}'")
        self.store_credential(_K_EMAIL, e)
        self.store_credential(_K_PW, pw)
        self.store_credential(_K_HOST, s.imap.host)
        self.store_credential(_K_PORT, str(s.imap.port))
        self.store_credential(_K_TLS, str(int(s.imap.tls)))
        self.store_credential(_K_SMTP_HOST, s.smtp.host)
        self.store_credential(_K_SMTP_PORT, str(s.smtp.port))
        self.store_credential(_K_SMTP_TLS, str(int(s.smtp.tls)))
        if not self.connect():
            self.delete_credentials()
            raise ValueError("[imap] Connection test failed")

    def connect(self):
        h = self.load_credential(_K_HOST)
        p = self.load_credential(_K_PORT)
        pw = self.load_credential(_K_PW)
        e = self.load_credential(_K_EMAIL)
        t = self.load_credential(_K_TLS)
        if not all([h, p, pw, e]):
            return False
        try:
            import imapclient
            c = imapclient.IMAPClient(h, port=int(p), ssl=t != "0", timeout=10)
            c.login(e, pw)
            c.logout()
            self._connected = True
            self._ensure_sync_registration()
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self):
        self._connected = False
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    "UPDATE scheduled_items SET status='cancelled' "
                    "WHERE source='imap' AND status='pending'"
                )
                conn.commit()
        except Exception as exc:
            logger.warning("[imap] disconnect cleanup: %s", exc)

    def _ensure_sync_registration(self):
        """Register IMAP sync handler + create recurring system scheduled_item."""
        try:
            from services.scheduler_service import register_system_handler
            register_system_handler('imap:sync', self.monitor)

            from services.database_service import get_shared_db_service
            import uuid
            db = get_shared_db_service()
            now = utc_now()

            with db.connection() as conn:
                cursor = conn.cursor()
                # Recurring sync item (every 15 minutes)
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid = ?",
                    ('system:imap:sync',),
                )
                if not cursor.fetchone():
                    cursor.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, recurrence, status,
                            topic, source, external_uid, hidden, created_at)
                         VALUES (?, 'system', 'IMAP email sync', ?, 'interval:15',
                                 'pending', 'imap:sync', 'imap', 'system:imap:sync',
                                 1, ?)""",
                        (uuid.uuid4().hex[:8], now.isoformat(), now.isoformat()),
                    )

                conn.commit()
            logger.info("[imap] Sync handler registered + scheduled_items ensured")
        except Exception as exc:
            logger.warning("[imap] _ensure_sync_registration: %s", exc)

    def _open_client(self):
        """Open and return an authenticated IMAPClient, or None."""
        h = self.load_credential(_K_HOST)
        p = self.load_credential(_K_PORT)
        pw = self.load_credential(_K_PW)
        e = self.load_credential(_K_EMAIL)
        t = self.load_credential(_K_TLS)
        if not all([h, p, pw, e]):
            return None
        try:
            import imapclient
            c = imapclient.IMAPClient(h, port=int(p), ssl=t != "0", timeout=30)
            c.login(e, pw)
            return c
        except Exception as exc:
            logger.error("[imap] _open_client: %s", exc)
            return None

    def ingest(self):
        """Fetch new email headers from INBOX and return normalized dicts.

        Uses a UID watermark for incremental fetching.  First run fetches
        the last 7 days; subsequent runs fetch only UIDs above the watermark.
        """
        client = self._open_client()
        if not client:
            return []
        try:
            client.select_folder("INBOX", readonly=True)
            wm = int(self.load_credential(_K_WATERMARK) or "0")
            since = (utc_now() - timedelta(days=_INITIAL_DAYS)).strftime("%d-%b-%Y")
            uids = client.search(["UID", f"{wm + 1}:*"] if wm else ["SINCE", since])
            if not uids:
                return []
            raw = client.fetch(uids, [b"RFC822.HEADER"])
            results = []
            max_uid = wm
            for u, d in raw.items():
                if u > wm:
                    results.append(parse_headers(u, d[b"RFC822.HEADER"]))
                    max_uid = max(max_uid, u)
            if max_uid > wm:
                self.store_credential(_K_WATERMARK, str(max_uid))
            return results
        except Exception as exc:
            logger.error("[imap] ingest: %s", exc)
            return []
        finally:
            try:
                client.logout()
            except Exception:
                pass

    def understand(self, items):
        """Classify headers and store as knowledge facts.

        Each email is tagged noise/informational/actionable via deterministic
        heuristics, then persisted via KnowledgeService so the LLM can reason
        about email context.
        """
        if not items:
            return items
        try:
            from services.database_service import get_shared_db_service
            from services.knowledge_service import KnowledgeService

            ks = KnowledgeService(get_shared_db_service())
            for item in items:
                item["triage"] = classify_email(item)
                item["is_thread"] = bool(item.get("in_reply_to"))
                self._store_email_fact(ks, item)
        except Exception as exc:
            logger.error("[imap] understand: %s", exc, exc_info=True)
        return items

    @staticmethod
    def _store_email_fact(ks, item):
        """Persist one email header as a knowledge fact + sender relationship."""
        key_id = item.get("message_id") or f"uid:{item['uid']}"
        sender = item.get("from_name") or item.get("from_addr", "unknown")
        ks.store(
            kind="fact", entity="email", key=f"msg:{key_id}",
            value=f"[{item['triage']}] {sender}: {item.get('subject', '')}",
            data=item, decay_class="fast", confidence=0.7, source="imap",
        )

    def monitor(self):
        """Fetch new headers and run triage + knowledge storage."""
        if not self.is_connected():
            self.connect()
        if not self.is_connected():
            return
        items = self.ingest()
        if items:
            self.understand(items)
            self._first_connect_greeting(items)
        self._inject_inbox_hint()

    # ------------------------------------------------------------------
    # First-connect greeting (one-time notification)
    # ------------------------------------------------------------------

    def _first_connect_greeting(self, items: list) -> None:
        """Create a one-time greeting notification after the first email ingest.

        Mirrors the CalDAV greeting pattern: checks for ``imap:greeting``
        in ``scheduled_items`` and inserts a notification with triage counts
        if it doesn't exist yet.
        """
        try:
            import uuid

            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            now = utc_now()
            greeting_uid = "imap:greeting"

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid = ?",
                    (greeting_uid,),
                )
                if cursor.fetchone():
                    return  # already greeted

                counts: dict[str, int] = {}
                for item in items:
                    cat = item.get("triage", "informational")
                    counts[cat] = counts.get(cat, 0) + 1

                total = len(items)
                parts = [f"Found {total} email{'s' if total != 1 else ''}"]
                if counts.get("actionable"):
                    parts.append(f"{counts['actionable']} actionable")
                if counts.get("informational"):
                    parts.append(f"{counts['informational']} informational")
                greeting_msg = f"Email connected! {', '.join(parts)}."

                cursor.execute(
                    """INSERT INTO scheduled_items
                       (id, item_type, message, due_at, status, topic,
                        source, external_uid, hidden, created_at)
                     VALUES (?, 'notification', ?, ?, 'pending', 'email',
                             'imap', ?, 0, ?)""",
                    (uuid.uuid4().hex[:8], greeting_msg, now.isoformat(),
                     greeting_uid, now.isoformat()),
                )
                conn.commit()
                logger.info("[imap] First-connect greeting: %s", greeting_msg)
        except Exception as exc:
            logger.error("[imap] _first_connect_greeting failed: %s", exc)

    # ------------------------------------------------------------------
    # World-state hint
    # ------------------------------------------------------------------

    def _inject_inbox_hint(self) -> None:
        """Push a compact inbox summary into WorldStateService.

        Mirrors CalDAV's ``_inject_schedule_hint`` pattern: query recent
        email facts from KnowledgeService, bucket by triage category,
        and write a one-line hint via ``notify_external_signal``.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.knowledge_service import KnowledgeService

            ks = KnowledgeService(get_shared_db_service())
            facts = ks.get_by_kind("fact", entity="email", limit=200)
            if not facts:
                return

            counts: dict[str, int] = {}
            top_actionable: str = ""
            for f in facts:
                d = f.get("data") or {}
                cat = d.get("triage", "informational")
                counts[cat] = counts.get(cat, 0) + 1
                if cat == "actionable" and not top_actionable:
                    top_actionable = d.get("from_name") or d.get("from_addr", "")

            actionable = counts.get("actionable", 0)
            informational = counts.get("informational", 0)
            if actionable == 0 and informational == 0:
                return

            parts = []
            if actionable:
                part = f"{actionable} actionable"
                if top_actionable:
                    part += f" (top: {top_actionable})"
                parts.append(part)
            if informational:
                parts.append(f"{informational} informational")
            noise = counts.get("noise", 0)
            if noise:
                parts.append(f"{noise} noise (filtered)")
            hint = f"Inbox: {', '.join(parts)}."

            from services.world_state_service import WorldStateService

            WorldStateService().notify_external_signal(
                signal_type="capability:imap",
                source="imap",
                content=hint,
                activation_energy=0.8 if actionable else 0.4,
            )
            logger.info("[imap] Injected inbox hint: %s", hint)
        except Exception as exc:
            logger.error("[imap] _inject_inbox_hint failed: %s", exc)

    # ------------------------------------------------------------------
    # SMTP send
    # ------------------------------------------------------------------

    def _open_smtp(self):
        """Open and return an authenticated SMTP connection, or None."""
        import smtplib

        host = self.load_credential(_K_SMTP_HOST)
        port = self.load_credential(_K_SMTP_PORT)
        email_addr = self.load_credential(_K_EMAIL)
        pw = self.load_credential(_K_PW)
        use_ssl = self.load_credential(_K_SMTP_TLS) == "1"
        if not all([host, port, email_addr, pw]):
            return None
        try:
            if use_ssl:
                conn = smtplib.SMTP_SSL(host, int(port), timeout=30)
            else:
                conn = smtplib.SMTP(host, int(port), timeout=30)
                conn.starttls()
            conn.login(email_addr, pw)
            return conn
        except Exception as exc:
            logger.error("[imap] _open_smtp: %s", exc)
            return None

    def _send_email(self, to: str, subject: str, body: str,
                    in_reply_to: str = "") -> dict:
        """Send an email via SMTP. Returns dict with success or error."""
        from email.mime.text import MIMEText

        sender = self.load_credential(_K_EMAIL)
        if not sender:
            return {"error": "No email configured"}

        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to

        conn = self._open_smtp()
        if not conn:
            return {"error": "Failed to connect to SMTP server"}
        try:
            conn.send_message(msg)
            return {"success": True, "to": to, "subject": subject}
        except Exception as exc:
            logger.error("[imap] _send_email: %s", exc)
            return {"error": str(exc)}
        finally:
            try:
                conn.quit()
            except Exception:
                pass

    def act(self, action, params):
        if action == "send_email":
            return self._send_email(
                to=params.get("to", ""),
                subject=params.get("subject", ""),
                body=params.get("body", ""),
                in_reply_to=params.get("in_reply_to", ""),
            )
        return {"error": f"'{action}' not implemented"}

    def get_tools(self):
        """Return IMAP tool definitions for dynamic registration."""
        capability = self

        def _search_execute(topic, params, config=None, telemetry=None):
            """Search ingested email headers by sender, subject, or triage."""
            try:
                from services.database_service import get_shared_db_service
                from services.knowledge_service import KnowledgeService

                ks = KnowledgeService(get_shared_db_service())
                facts = ks.get_by_kind("fact", entity="email", limit=200)

                sender = (params.get("sender") or "").lower()
                subject = (params.get("subject") or "").lower()
                triage = (params.get("triage") or "").lower()
                limit = int(params.get("limit", 20))

                results = []
                for fact in facts:
                    d = fact.get("data") or {}
                    if sender:
                        fn = (d.get("from_name") or "").lower()
                        fa = (d.get("from_addr") or "").lower()
                        if sender not in fn and sender not in fa:
                            continue
                    if subject and subject not in (d.get("subject") or "").lower():
                        continue
                    if triage and d.get("triage") != triage:
                        continue
                    results.append({
                        "uid": d.get("uid"),
                        "subject": d.get("subject", ""),
                        "from_name": d.get("from_name", ""),
                        "from_addr": d.get("from_addr", ""),
                        "date": d.get("date", ""),
                        "triage": d.get("triage", ""),
                        "is_thread": d.get("is_thread", False),
                    })
                    if len(results) >= limit:
                        break

                return {"emails": results, "count": len(results)}
            except Exception as exc:
                logger.error("[imap] imap_search_email failed: %s", exc)
                return {"error": str(exc)}

        def _send_execute(topic, params, config=None, telemetry=None):
            """Send or reply to an email via SMTP."""
            to = (params.get("to") or "").strip()
            subject = (params.get("subject") or "").strip()
            body = (params.get("body") or "").strip()
            if not to or not subject or not body:
                return {"error": "to, subject, and body are required"}
            return capability._send_email(
                to=to, subject=subject, body=body,
                in_reply_to=params.get("in_reply_to", ""),
            )

        return [
            {
                "name": "imap_send_email",
                "description": (
                    "Send or reply to an email. Provide recipient address, "
                    "subject, and plain-text body. For replies, include "
                    "in_reply_to with the original Message-ID to thread correctly."
                ),
                "parameters": {
                    "to": {
                        "type": "string",
                        "required": True,
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "required": True,
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "required": True,
                        "description": "Plain-text email body.",
                    },
                    "in_reply_to": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Message-ID of the email being replied to. "
                            "Sets In-Reply-To and References headers for threading."
                        ),
                    },
                },
                "returns": {
                    "success": {"type": "boolean", "description": "True if sent."},
                    "to": {"type": "string", "description": "Recipient address."},
                    "subject": {"type": "string", "description": "Subject sent."},
                    "error": {"type": "string", "description": "Error message if failed."},
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _send_execute,
            },
            {
                "name": "imap_search_email",
                "description": (
                    "Search ingested email by sender name/address, subject keyword, "
                    "or triage category (noise, informational, actionable). "
                    "Returns matching email headers from the knowledge store."
                ),
                "parameters": {
                    "sender": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Filter by sender name or email address (case-insensitive substring match)."
                        ),
                    },
                    "subject": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Filter by subject keyword (case-insensitive substring match)."
                        ),
                    },
                    "triage": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Filter by triage category: 'noise', 'informational', or 'actionable'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "required": False,
                        "description": "Maximum number of results to return (default 20).",
                    },
                },
                "returns": {
                    "emails": {
                        "type": "array",
                        "description": (
                            "List of email dicts with uid, subject, from_name, "
                            "from_addr, date, triage, is_thread."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of emails returned.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if the operation failed.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _search_execute,
            },
        ]
