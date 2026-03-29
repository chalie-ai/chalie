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
from datetime import datetime as _dt, timedelta

import numpy as np
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
_MAX_BODY_CHARS = 4000


def _imap_date(iso_str: str) -> str:
    """Convert an ISO date string (YYYY-MM-DD) to IMAP date format (DD-Mon-YYYY)."""
    try:
        return _dt.strptime(iso_str[:10], "%Y-%m-%d").strftime("%d-%b-%Y")
    except (ValueError, IndexError):
        return iso_str

# ---------------------------------------------------------------------------
# Embedding-based triage (replaces regex synonym maps — see feedback_no_synonym_maps)
# ---------------------------------------------------------------------------

_TRIAGE_ANCHORS = {
    "noise": (
        "automated notification, marketing newsletter, system alert, "
        "promotional offer, subscription digest, no-reply bulk email"
    ),
    "actionable": (
        "action required, please reply, deadline approaching, invoice "
        "payment due, approval needed, confirm attendance, follow-up request"
    ),
    "informational": (
        "general information, status update, FYI announcement, meeting "
        "notes shared, project update summary, weekly report"
    ),
}

_anchor_cache: dict | None = None


def _get_anchor_embeddings() -> dict:
    """Return cached anchor embeddings, computing on first call."""
    global _anchor_cache
    if _anchor_cache is not None:
        return _anchor_cache
    from services.embedding_service import get_embedding_service
    svc = get_embedding_service()
    _anchor_cache = {
        cat: svc.generate_embedding_np(text)
        for cat, text in _TRIAGE_ANCHORS.items()
    }
    return _anchor_cache


def _build_email_text(item: dict) -> str:
    """Build classification text from email signals."""
    parts = []
    subj = item.get("subject", "")
    if subj:
        parts.append(subj)
    sender = item.get("from_name") or item.get("from_addr", "")
    if sender:
        parts.append(f"from {sender}")
    return " ".join(parts) if parts else "email message"


def classify_email(item: dict) -> str:
    """Classify an email header dict as noise, informational, or actionable.

    Uses embedding similarity against canonical intent anchors.
    Only ``has_unsubscribe`` is a deterministic fast path; threaded replies
    (``in_reply_to``) go through the embedding classifier because many
    automated systems (newsletters, GitHub notifications) set In-Reply-To.
    """
    if item.get("has_unsubscribe"):
        return "noise"

    from services.embedding_service import get_embedding_service
    svc = get_embedding_service()
    email_emb = svc.generate_embedding_np(_build_email_text(item))
    anchors = _get_anchor_embeddings()

    best_cat = "informational"
    best_sim = -1.0
    for cat, anchor_emb in anchors.items():
        sim = float(np.dot(email_emb, anchor_emb))
        if sim > best_sim:
            best_sim = sim
            best_cat = cat
    return best_cat


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


def resolve_display_name(from_addr: str, from_name: str = "") -> str:
    """Resolve a sender to a human-readable display name via contact index.

    Returns the best available name: contact_resolver match > header name > email.
    """
    from capabilities.contact_resolver import resolve as _cr_resolve
    addr = (from_addr or "").strip()
    name = (from_name or "").strip()
    if addr:
        try:
            matches = _cr_resolve(addr, limit=1)
            if matches and matches[0].get("name"):
                return matches[0]["name"]
        except Exception as exc:
            logger.debug("[imap] resolve_display_name(%s): %s", addr, exc)
    return name or addr


def extract_body(raw_bytes: bytes, max_chars: int = _MAX_BODY_CHARS) -> str:
    """Extract plain-text body from raw RFC822 email bytes."""
    msg = _email_mod.message_from_bytes(raw_bytes, policy=_email_mod.policy.default)
    parts = list(msg.walk()) if msg.is_multipart() else [msg]
    by_type: dict[str, list[str]] = {"text/plain": [], "text/html": []}
    for p in parts:
        ct = p.get_content_type()
        if ct in by_type:
            c = p.get_content()
            if isinstance(c, str):
                by_type[ct].append(c)
    text = "\n".join(by_type["text/plain"]) or re.sub(
        r"<[^>]+>", "", "\n".join(by_type["text/html"])).strip()
    return (text[:max_chars] + "\n[truncated]") if len(text) > max_chars else text


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

    def ingest(self, client=None):
        """Fetch new email headers from INBOX and return normalized dicts.

        Uses a UID watermark for incremental fetching.  First run fetches
        the last 7 days; subsequent runs fetch only UIDs above the watermark.

        If *client* is provided (e.g. from a shared session), it is reused
        and **not** closed here.
        """
        _owns = client is None
        if _owns:
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
            if _owns:
                try:
                    client.logout()
                except Exception as exc:
                    logger.debug("[imap] logout after ingest: %s", exc)

    def understand(self, items):
        """Classify headers via deterministic triage heuristics.

        Tags each email as noise/informational/actionable. Also indexes
        sender identities for cross-capability contact resolution.
        """
        if not items:
            return items
        from capabilities.contact_resolver import index_person
        for item in items:
            item["triage"] = classify_email(item)
            item["is_thread"] = bool(item.get("in_reply_to"))
            if item.get("from_addr"):
                index_person(item["from_addr"], item.get("from_name"), source="imap")
        try:
            from capabilities.signal_bridge import emit_capability_signal
            for item in items:
                if item.get("triage") == "actionable":
                    sender = item.get("from_name") or item.get("from_addr", "")
                    subject = item.get("subject", "")
                    emit_capability_signal(
                        "imap", "new_knowledge",
                        f"Actionable email from {sender}: {subject}",
                        source="imap:triage",
                    )
        except Exception as exc:
            logger.debug("[imap] signal bridge emit failed: %s", exc)
        return items

    def _do_monitor(self):
        """Fetch new headers and run triage + knowledge storage.

        Opens a single IMAP connection and shares it across ingest() and
        _inject_inbox_hint() to avoid redundant TLS handshakes per cycle.
        """
        if not self.is_connected():
            self.connect()
        if not self.is_connected():
            return
        client = self._open_client()
        if not client:
            return
        try:
            items = self.ingest(client=client)
            if items:
                self.understand(items)
                self._first_connect_greeting(items)
            self._inject_inbox_hint(client=client)
        finally:
            try:
                client.logout()
            except Exception as exc:
                logger.debug("[imap] logout after monitor: %s", exc)

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

    def _inject_inbox_hint(self, client=None) -> None:
        """Push a compact inbox summary into WorldStateService.

        Queries the IMAP server directly for recent unseen emails,
        classifies them, and writes a one-line hint.

        If *client* is provided (e.g. from a shared session), it is reused
        and **not** closed here.
        """
        _owns = client is None
        if _owns:
            client = self._open_client()
        if not client:
            return
        try:
            client.select_folder("INBOX", readonly=True)
            since = (utc_now() - timedelta(days=3)).strftime("%d-%b-%Y")
            uids = client.search(["UNSEEN", "SINCE", since])
            if not uids:
                return

            raw = client.fetch(uids, [b"RFC822.HEADER"])
            counts: dict[str, int] = {}
            top_actionable: str = ""
            for u, d in raw.items():
                item = parse_headers(u, d[b"RFC822.HEADER"])
                cat = classify_email(item)
                counts[cat] = counts.get(cat, 0) + 1
                if cat == "actionable" and not top_actionable:
                    top_actionable = item.get("from_name") or item.get("from_addr", "")

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
        finally:
            if _owns:
                try:
                    client.logout()
                except Exception as exc:
                    logger.debug("[imap] logout after inbox_hint: %s", exc)

    # ------------------------------------------------------------------
    # Drafts folder discovery + IMAP APPEND
    # ------------------------------------------------------------------

    def _find_drafts_folder(self, client) -> str:
        """Locate the Drafts folder via RFC 6154 flag, fallback 'Drafts'."""
        try:
            for flags, _, name in client.list_folders():
                if b"\\Drafts" in flags:
                    return name
        except Exception:
            pass
        return "Drafts"

    def _save_draft(self, client, to: str, subject: str, body: str,
                    in_reply_to: str = "") -> dict:
        """Compose a MIME message and APPEND it to the Drafts folder."""
        from email.mime.text import MIMEText

        sender = self.load_credential(_K_EMAIL)
        if not sender:
            return {"error": "No email configured"}

        drafts = self._find_drafts_folder(client)
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to

        client.append(drafts, msg.as_bytes(),
                      flags=[b"\\Draft", b"\\Seen"])
        return {
            "success": True, "to": to, "subject": subject,
            "folder": drafts,
        }

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
            """Search emails directly on the IMAP server."""
            try:
                client = capability._open_client()
                if not client:
                    return {"error": "Cannot connect to email server"}
                try:
                    client.select_folder("INBOX", readonly=True)

                    criteria = []
                    sender = (params.get("sender") or "").strip()
                    subject = (params.get("subject") or "").strip()
                    keyword = (params.get("keyword") or "").strip()
                    date_from = (params.get("date_from") or "").strip()
                    date_to = (params.get("date_to") or "").strip()
                    if sender:
                        criteria.extend(["FROM", sender])
                    if subject:
                        criteria.extend(["SUBJECT", subject])
                    if keyword:
                        criteria.extend(["TEXT", keyword])
                    if date_from:
                        criteria.extend(["SINCE", _imap_date(date_from)])
                    if date_to:
                        criteria.extend(["BEFORE", _imap_date(date_to)])
                    if params.get("unanswered"):
                        criteria.append("UNANSWERED")
                    if not criteria:
                        since = (utc_now() - timedelta(days=7)).strftime("%d-%b-%Y")
                        criteria.extend(["SINCE", since])

                    uids = client.search(criteria)
                    limit = int(params.get("limit", 20))
                    uids = uids[-limit:] if len(uids) > limit else uids
                    if not uids:
                        return {"emails": [], "count": 0}

                    raw = client.fetch(uids, [b"RFC822.HEADER"])
                    triage_filter = (params.get("triage") or "").lower()
                    results = []
                    for u in sorted(raw.keys(), reverse=True):
                        item = parse_headers(u, raw[u][b"RFC822.HEADER"])
                        item["triage"] = classify_email(item)
                        item["is_thread"] = bool(item.get("in_reply_to"))
                        if triage_filter and item["triage"] != triage_filter:
                            continue
                        results.append({
                            "uid": item["uid"],
                            "subject": item.get("subject", ""),
                            "from_name": item.get("from_name", ""),
                            "from_addr": item.get("from_addr", ""),
                            "from_display": resolve_display_name(
                                item.get("from_addr", ""),
                                item.get("from_name", ""),
                            ),
                            "date": item.get("date", ""),
                            "triage": item["triage"],
                            "is_thread": item["is_thread"],
                        })
                    return {"emails": results, "count": len(results)}
                finally:
                    try:
                        client.logout()
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("[imap] imap_search_email failed: %s", exc)
                return {"error": str(exc)}

        def _read_execute(topic, params, config=None, telemetry=None):
            """Fetch the full plain-text body of an email by UID."""
            uid = params.get("uid")
            if not uid:
                return {"error": "uid is required"}
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                return {"error": "uid must be an integer"}
            client = capability._open_client()
            if not client:
                return {"error": "Cannot connect to email server"}
            try:
                client.select_folder("INBOX", readonly=True)
                raw = client.fetch([uid], [b"RFC822"])
                if uid not in raw:
                    return {"error": f"Email UID {uid} not found"}
                raw_bytes = raw[uid][b"RFC822"]
                headers = parse_headers(uid, raw_bytes)
                return {
                    "uid": uid,
                    "subject": headers.get("subject", ""),
                    "from_name": headers.get("from_name", ""),
                    "from_addr": headers.get("from_addr", ""),
                    "from_display": resolve_display_name(
                        headers.get("from_addr", ""),
                        headers.get("from_name", ""),
                    ),
                    "date": headers.get("date", ""),
                    "body": extract_body(raw_bytes),
                }
            except Exception as exc:
                logger.error("[imap] imap_read_email failed: %s", exc)
                return {"error": str(exc)}
            finally:
                try:
                    client.logout()
                except Exception:
                    pass

        def _send_execute(topic, params, config=None, telemetry=None):
            """Send or reply to an email via SMTP."""
            to = (params.get("to") or "").strip()
            subject = (params.get("subject") or "").strip()
            body = (params.get("body") or "").strip()
            if not to or not subject or not body:
                return {"error": "to, subject, and body are required"}
            # Auto-resolve name to email via ContactResolver
            resolved_from = None
            if "@" not in to:
                from capabilities.contact_resolver import resolve
                matches = resolve(to, limit=5)
                if len(matches) == 1:
                    resolved_from = to
                    to = matches[0]["email"]
                elif len(matches) > 1:
                    return {
                        "error": "ambiguous_recipient",
                        "message": f"Multiple contacts match '{to}'. Please clarify.",
                        "candidates": matches,
                    }
                else:
                    return {
                        "error": "unknown_recipient",
                        "message": f"No contacts found for '{to}'. Please provide an email address.",
                    }
            result = capability._send_email(
                to=to, subject=subject, body=body,
                in_reply_to=params.get("in_reply_to", ""),
            )
            if resolved_from:
                result["resolved_from"] = resolved_from
            return result

        def _mark_read_execute(topic, params, config=None, telemetry=None):
            """Mark one or more emails as read by setting the \\Seen flag."""
            uids = params.get("uids") or params.get("uid")
            if not uids:
                return {"error": "uid or uids is required"}
            if isinstance(uids, (int, str)):
                uids = [uids]
            try:
                uids = [int(u) for u in uids]
            except (TypeError, ValueError):
                return {"error": "uids must be integers"}
            client = capability._open_client()
            if not client:
                return {"error": "Cannot connect to email server"}
            try:
                client.select_folder("INBOX", readonly=False)
                client.add_flags(uids, [b"\\Seen"])
                return {"success": True, "uids": uids}
            except Exception as exc:
                logger.error("[imap] imap_mark_read failed: %s", exc)
                return {"error": str(exc)}
            finally:
                try:
                    client.logout()
                except Exception as exc:
                    logger.debug("[imap] logout after mark_read: %s", exc)

        def _draft_reply_execute(topic, params, config=None, telemetry=None):
            """Fetch original email, compose a reply draft, save to Drafts."""
            uid = params.get("uid")
            body = (params.get("body") or "").strip()
            if not uid:
                return {"error": "uid is required"}
            if not body:
                return {"error": "body is required"}
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                return {"error": "uid must be an integer"}
            client = capability._open_client()
            if not client:
                return {"error": "Cannot connect to email server"}
            try:
                client.select_folder("INBOX", readonly=True)
                raw = client.fetch([uid], [b"RFC822.HEADER"])
                if uid not in raw:
                    return {"error": f"Email UID {uid} not found"}
                headers = parse_headers(uid, raw[uid][b"RFC822.HEADER"])
                # Build reply subject
                orig_subj = headers.get("subject", "")
                reply_subj = orig_subj if orig_subj.lower().startswith("re:") \
                    else f"Re: {orig_subj}"
                # Reply-to address: use From of original
                reply_to = headers.get("from_addr", "")
                if not reply_to:
                    return {"error": "Cannot determine reply address"}
                message_id = headers.get("message_id", "")
                return capability._save_draft(
                    client, to=reply_to, subject=reply_subj,
                    body=body, in_reply_to=message_id,
                )
            except Exception as exc:
                logger.error("[imap] imap_draft_reply failed: %s", exc)
                return {"error": str(exc)}
            finally:
                try:
                    client.logout()
                except Exception:
                    pass

        def _snooze_execute(topic, params, config=None, telemetry=None):
            """Snooze an email — schedule a re-surface nudge at a given time."""
            uid = params.get("uid")
            snooze_until = (params.get("snooze_until") or "").strip()
            reason = (params.get("reason") or "").strip()
            if not uid or not snooze_until:
                return {"error": "uid and snooze_until are required"}
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                return {"error": "uid must be an integer"}

            from services.time_utils import parse_utc
            due = parse_utc(snooze_until)
            if due.year == 1:  # datetime.min sentinel = parse failure
                return {"error": "snooze_until must be a valid ISO 8601 datetime"}

            # Fetch email headers so the nudge message is informative
            client = capability._open_client()
            if not client:
                return {"error": "Cannot connect to email server"}
            try:
                client.select_folder("INBOX", readonly=True)
                raw = client.fetch([uid], [b"RFC822.HEADER"])
                if uid not in raw:
                    return {"error": f"Email UID {uid} not found"}
                headers = parse_headers(uid, raw[uid][b"RFC822.HEADER"])
            except Exception as exc:
                logger.error("[imap] imap_snooze fetch: %s", exc)
                return {"error": str(exc)}
            finally:
                try:
                    client.logout()
                except Exception as exc:
                    logger.debug("[imap] logout after snooze: %s", exc)

            subject = headers.get("subject", "")
            sender = headers.get("from_name") or headers.get("from_addr", "")

            # Build an LLM execution prompt for when the snooze fires
            prompt_parts = [
                f"Snoozed email from {sender}: \"{subject}\" (UID {uid}).",
                "Use imap_search_email with unanswered=true and "
                f"subject=\"{subject[:60]}\" to check if this email "
                "has been replied to.",
                "If still unanswered, remind the user about it.",
                "If already replied, confirm it's handled.",
            ]
            if reason:
                prompt_parts.append(f"User's note: {reason}")
            prompt_msg = " ".join(prompt_parts)

            # Create a scheduled_item that fires through the LLM pipeline
            import uuid
            from services.database_service import get_shared_db_service
            item_id = uuid.uuid4().hex[:8]
            now = utc_now()
            ext_uid = f"snooze:{uid}"

            db = get_shared_db_service()
            with db.connection() as conn:
                # Prevent duplicate snoozes for the same email
                existing = conn.execute(
                    "SELECT id FROM scheduled_items "
                    "WHERE external_uid = ? AND status = 'pending'",
                    (ext_uid,),
                ).fetchone()
                if existing:
                    return {
                        "error": "duplicate",
                        "message": f"Email UID {uid} already has a pending snooze.",
                    }
                conn.execute(
                    """INSERT INTO scheduled_items
                       (id, item_type, message, due_at, status, topic,
                        source, external_uid, hidden, created_at, is_prompt)
                     VALUES (?, 'prompt', ?, ?, 'pending', 'email',
                             'imap:snooze', ?, 0, ?, 1)""",
                    (item_id, prompt_msg, due.isoformat(),
                     ext_uid, now.isoformat()),
                )
                conn.commit()

            logger.info("[imap] Snoozed UID %d until %s (id=%s)",
                        uid, due.isoformat(), item_id)
            return {
                "success": True,
                "snooze_id": item_id,
                "email_uid": uid,
                "subject": subject,
                "sender": sender,
                "snooze_until": due.isoformat(),
            }

        return [
            {
                "name": "imap_mark_read",
                "description": (
                    "Mark one or more emails as read on the mail server by "
                    "setting the IMAP \\Seen flag. Accepts a single UID or a "
                    "list of UIDs. Use after reading or acting on an email so "
                    "it no longer appears as unread."
                ),
                "parameters": {
                    "uid": {
                        "type": "integer",
                        "required": False,
                        "description": "Single IMAP UID to mark as read.",
                    },
                    "uids": {
                        "type": "array",
                        "required": False,
                        "description": (
                            "List of IMAP UIDs to mark as read. "
                            "Provide uid or uids, not both."
                        ),
                    },
                },
                "returns": {
                    "success": {
                        "type": "boolean",
                        "description": "True if flags were set.",
                    },
                    "uids": {
                        "type": "array",
                        "description": "UIDs that were marked as read.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if failed.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _mark_read_execute,
            },
            {
                "name": "imap_send_email",
                "description": (
                    "Send or reply to an email. Provide recipient (email address "
                    "or contact name — names are auto-resolved via people index), "
                    "subject, and plain-text body. For replies, include "
                    "in_reply_to with the original Message-ID to thread correctly."
                ),
                "parameters": {
                    "to": {
                        "type": "string",
                        "required": True,
                        "description": (
                            "Recipient email address or contact name "
                            "(names auto-resolved via people index)."
                        ),
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
                "name": "imap_draft_reply",
                "description": (
                    "Draft a reply to an email and save it in the Drafts "
                    "folder. The user can review and send it from their "
                    "email client. Use when the user agrees to reply to an "
                    "email and you want to compose the draft for them."
                ),
                "parameters": {
                    "uid": {
                        "type": "integer",
                        "required": True,
                        "description": "IMAP UID of the email to reply to.",
                    },
                    "body": {
                        "type": "string",
                        "required": True,
                        "description": "Plain-text body of the reply draft.",
                    },
                },
                "returns": {
                    "success": {
                        "type": "boolean",
                        "description": "True if draft saved.",
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Reply subject line.",
                    },
                    "folder": {
                        "type": "string",
                        "description": "Drafts folder where saved.",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if failed.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _draft_reply_execute,
            },
            {
                "name": "imap_search_email",
                "description": (
                    "Search emails directly on the mail server by sender, subject, "
                    "keyword (full-text body search), date range, or triage category "
                    "(noise, informational, actionable). Returns live results from the inbox."
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
                    "keyword": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Full-text search across email headers and body "
                            "(server-side IMAP TEXT search)."
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Only emails on or after this date (ISO format YYYY-MM-DD)."
                        ),
                    },
                    "date_to": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Only emails before this date (ISO format YYYY-MM-DD)."
                        ),
                    },
                    "triage": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Filter by triage category: 'noise', 'informational', or 'actionable'."
                        ),
                    },
                    "unanswered": {
                        "type": "boolean",
                        "required": False,
                        "description": (
                            "If true, only return emails the user has not replied to "
                            "(IMAP \\Answered flag not set)."
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
            {
                "name": "imap_read_email",
                "description": (
                    "Read the full content of an email by its UID. "
                    "Returns subject, sender, date, and plain-text body. "
                    "Use imap_search_email first to find the UID."
                ),
                "parameters": {
                    "uid": {
                        "type": "integer",
                        "required": True,
                        "description": "The IMAP UID of the email to read.",
                    },
                },
                "returns": {
                    "uid": {"type": "integer", "description": "Email UID."},
                    "subject": {"type": "string", "description": "Subject line."},
                    "from_name": {"type": "string", "description": "Sender name."},
                    "from_addr": {"type": "string", "description": "Sender email."},
                    "date": {"type": "string", "description": "Send date (ISO)."},
                    "body": {"type": "string", "description": "Plain-text body."},
                    "error": {"type": "string", "description": "Error if failed."},
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _read_execute,
            },
            {
                "name": "imap_snooze",
                "description": (
                    "Snooze an email — schedule a reminder to resurface it at "
                    "a specified time. When the snooze fires, Chalie checks if "
                    "the email has been replied to and reminds the user if not. "
                    "Use when the user says 'remind me about this email later' "
                    "or 'follow up on this Friday'."
                ),
                "parameters": {
                    "uid": {
                        "type": "integer",
                        "required": True,
                        "description": "IMAP UID of the email to snooze.",
                    },
                    "snooze_until": {
                        "type": "string",
                        "required": True,
                        "description": (
                            "ISO 8601 datetime when the email should "
                            "resurface (e.g. '2026-04-01T09:00:00+00:00')."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "required": False,
                        "description": (
                            "Optional note about what to do with this email "
                            "(e.g. 'reply with Q1 numbers')."
                        ),
                    },
                },
                "returns": {
                    "success": {
                        "type": "boolean",
                        "description": "True if snoozed.",
                    },
                    "snooze_id": {
                        "type": "string",
                        "description": "Scheduled item ID.",
                    },
                    "email_uid": {
                        "type": "integer",
                        "description": "Email UID.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject.",
                    },
                    "sender": {
                        "type": "string",
                        "description": "Sender name or email.",
                    },
                    "snooze_until": {
                        "type": "string",
                        "description": "Snooze datetime (ISO).",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if failed.",
                    },
                },
                "constraints": {"timeout_seconds": 30},
                "handler": _snooze_execute,
            },
        ]
