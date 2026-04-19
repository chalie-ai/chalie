"""ImapHandler — protocol-specific IMAP/SMTP logic for MailCapability.

Plain class; no AbstractCapability.  Credentials are passed as parameters
so the parent MailCapability remains the sole credential owner.
"""

from __future__ import annotations

import email as _email_mod
import email.policy
import email.utils
import logging
import re
from datetime import timedelta

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_INITIAL_DAYS = 7
_MAX_BODY_CHARS = 4000


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _imap_date(iso_str: str) -> str:
    """ISO YYYY-MM-DD → IMAP DD-Mon-YYYY."""
    from datetime import datetime as _dt
    try:
        return _dt.strptime(iso_str[:10], "%Y-%m-%d").strftime("%d-%b-%Y")
    except (ValueError, IndexError):
        return iso_str


def _hdr(msg, name: str) -> str:
    v = msg.get(name)
    return str(v) if v else ""


def _safe_date(s: str) -> str:
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


def extract_body(raw_bytes: bytes, max_chars: int = _MAX_BODY_CHARS) -> str:
    """Extract plain-text body from raw RFC822 bytes; falls back to de-HTMLified text/html."""
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
        r"<[^>]+>", "", "\n".join(by_type["text/html"])
    ).strip()
    return (text[:max_chars] + "\n[truncated]") if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# ImapHandler
# ---------------------------------------------------------------------------

class ImapHandler:
    """Stateless IMAP/SMTP operations — credentials passed per-call."""

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def open_client(self, host: str, port: int, tls: bool,
                    email: str, password: str, timeout: int = 30):
        """Open and return an authenticated IMAPClient, or None on failure."""
        import imapclient
        try:
            c = imapclient.IMAPClient(host, port=port, ssl=tls, timeout=timeout)
            c.login(email, password)
            return c
        except Exception as exc:
            logger.error("[imap_handler] open_client: %s", exc)
            return None

    def open_smtp(self, host: str, port: int, tls: bool,
                  email: str, password: str, timeout: int = 30):
        """Open and return an authenticated SMTP connection, or None on failure."""
        import smtplib
        try:
            if tls:
                conn = smtplib.SMTP_SSL(host, port, timeout=timeout)
            else:
                conn = smtplib.SMTP(host, port, timeout=timeout)
                conn.starttls()
            conn.login(email, password)
            return conn
        except Exception as exc:
            logger.error("[imap_handler] open_smtp: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, client, watermark: int | None) -> tuple[list[dict], int | None]:
        """Fetch new email headers from INBOX since *watermark* UID.

        Returns (items, new_watermark).  Caller owns the client lifecycle.
        First run (watermark=None/0) fetches the last _INITIAL_DAYS days.
        """
        wm = watermark or 0
        try:
            client.select_folder("INBOX", readonly=True)
            since = (utc_now() - timedelta(days=_INITIAL_DAYS)).strftime("%d-%b-%Y")
            uids = client.search(["UID", f"{wm + 1}:*"] if wm else ["SINCE", since])
            if not uids:
                return [], wm or None
            raw = client.fetch(uids, [b"RFC822.HEADER"])
            results, max_uid = [], wm
            for u, d in raw.items():
                if u > wm:
                    results.append(parse_headers(u, d[b"RFC822.HEADER"]))
                    max_uid = max(max_uid, u)
            new_watermark = max_uid if max_uid > wm else (wm or None)
            return results, new_watermark
        except Exception as exc:
            logger.error("[imap_handler] ingest: %s", exc)
            return [], wm or None

    # ------------------------------------------------------------------
    # Understand
    # ------------------------------------------------------------------

    def understand(self, items: list[dict]) -> list[dict]:
        """Classify + index items: sets triage, is_thread; indexes senders; emits signals."""
        if not items:
            return items
        from capabilities.contact_resolver import index_person
        from capabilities.mail_capability.email_triage import classify_email
        for item in items:
            item["triage"] = classify_email(item)
            item["is_thread"] = bool(item.get("in_reply_to"))
            if item.get("from_addr"):
                index_person(item["from_addr"], item.get("from_name"), source="imap")
        return items

    # ------------------------------------------------------------------
    # WorldState inbox hint
    # ------------------------------------------------------------------

    def inject_inbox_hint(self, client, *, owns_client: bool = False) -> None:
        """Push a compact unseen-inbox summary into the world state singleton.

        If owns_client is True the client is closed in the finally block.
        """
        from capabilities.mail_capability.email_triage import classify_email
        try:
            client.select_folder("INBOX", readonly=True)
            since = (utc_now() - timedelta(days=3)).strftime("%d-%b-%Y")
            uids = client.search(["UNSEEN", "SINCE", since])
            if not uids:
                return
            raw = client.fetch(uids, [b"RFC822.HEADER"])
            counts: dict[str, int] = {}
            top_actionable = ""
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
            from services.world_state import world_state
            world_state.push_signal("inbox", hint, ttl=3600)
            logger.info("[imap_handler] inbox hint: %s", hint)
        except Exception as exc:
            logger.error("[imap_handler] inject_inbox_hint: %s", exc)
        finally:
            if owns_client:
                try:
                    client.logout()
                except Exception as exc:
                    logger.debug("[imap_handler] logout: %s", exc)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, client, params: dict) -> dict:
        """Execute IMAP search and return classified header results.

        Supported params: sender, subject, keyword, date_from, date_to,
        triage (post-filter), unanswered, limit.
        """
        from capabilities.mail_capability.email_triage import classify_email
        try:
            client.select_folder("INBOX", readonly=True)
            criteria: list = []
            sender  = (params.get("sender")    or "").strip()
            subject = (params.get("subject")   or "").strip()
            keyword = (params.get("keyword")   or "").strip()
            date_from = (params.get("date_from") or "").strip()
            date_to   = (params.get("date_to")   or "").strip()
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
                    "message_id": item.get("message_id", ""),
                    "subject": item.get("subject", ""),
                    "from_name": item.get("from_name", ""),
                    "from_addr": item.get("from_addr", ""),
                    "date": item.get("date", ""),
                    "triage": item["triage"],
                    "is_thread": item["is_thread"],
                })
            return {"emails": results, "count": len(results)}
        except Exception as exc:
            logger.error("[imap_handler] search: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Read full body
    # ------------------------------------------------------------------

    def read_email(self, client, params: dict) -> dict:
        """Fetch full plain-text body of an email by UID. Caller owns client."""
        uid = params.get("uid")
        if not uid:
            return {"error": "uid is required"}
        try:
            uid = int(uid)
        except (TypeError, ValueError):
            return {"error": "uid must be an integer"}
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
                "date": headers.get("date", ""),
                "body": extract_body(raw_bytes),
            }
        except Exception as exc:
            logger.error("[imap_handler] read_email: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send_email(self, *, smtp_host: str, smtp_port: int, smtp_tls: bool,
                   email: str, password: str, params: dict) -> dict:
        """Send an email via SMTP.

        Required params: to, subject, body.
        Optional params: in_reply_to (Message-ID for threading).
        """
        from email.mime.text import MIMEText
        to         = (params.get("to")         or "").strip()
        subject    = (params.get("subject")    or "").strip()
        body       = (params.get("body")       or "").strip()
        in_reply_to = (params.get("in_reply_to") or "").strip()
        if not to or not subject or not body:
            return {"error": "to, subject, and body are required"}
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = email
        msg["To"] = to
        msg["Subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to
        conn = self.open_smtp(host=smtp_host, port=smtp_port, tls=smtp_tls,
                               email=email, password=password)
        if not conn:
            return {"error": "Failed to connect to SMTP server"}
        try:
            conn.send_message(msg)
            return {"success": True, "to": to, "subject": subject}
        except Exception as exc:
            logger.error("[imap_handler] send_email: %s", exc)
            return {"error": str(exc)}
        finally:
            try:
                conn.quit()
            except Exception:
                pass
