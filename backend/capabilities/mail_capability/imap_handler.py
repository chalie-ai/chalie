"""Protocol-specific IMAP logic for MailCapability."""

from __future__ import annotations

import email as _email_mod
import email.message
import email.policy
import logging
import mimetypes
import os
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, cast

from services.filename_utils import safe_filename
from services.tmp_storage import new_tmp_path
from services.time_utils import utc_now

if TYPE_CHECKING:
    from typing import Protocol

    class _ImapClient(Protocol):
        def login(self, user: str, password: str) -> object: ...
        def select_folder(self, folder: str, readonly: bool = False) -> object: ...
        def search(self, criteria: list[object]) -> list[int]: ...
        def fetch(self, uids: list[int], data: list[bytes]) -> dict[int, dict[bytes, bytes]]: ...
        def append(self, folder: str, msg: bytes, flags: list[bytes] = ...) -> object: ...
        def delete_messages(self, uids: list[int]) -> object: ...
        def expunge(self, uids: list[int]) -> object: ...
        def set_flags(self, uids: list[int], flags: list[bytes]) -> object: ...
        def move(self, uids: list[int], folder: str) -> object: ...
        def copy(self, uids: list[int], folder: str) -> object: ...
        def list_folders(self) -> list[tuple[object, object, str]]: ...
        def logout(self) -> object: ...


@dataclass(frozen=True, slots=True)
class SmtpCreds:
    host: str
    port: int
    tls: bool
    from_addr: str
    password: str

logger = logging.getLogger(__name__)

_INITIAL_DAYS = 7
_SMTP_TIMEOUT = 15

_IMAP_DATE_FMT = "%d-%b-%Y"
# Hard cap on search results. Deliberately NOT a param: a model-controlled
# limit is unbounded mailbox dumping, so the cap is not part of the tool schema.
_SEARCH_LIMIT = 10

_IMAP_FETCH_HEADER = b"RFC822.HEADER"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_SPAM_NAMES = {"junk", "spam", "[gmail]/spam", "bulk mail", "spamverdacht"}
_DRAFTS_NAMES = {"drafts", "[gmail]/drafts", "draft", "entwürfe"}


def _find_folder(client: "_ImapClient", names: set[str]) -> str | None:
    folders = client.list_folders()
    for _flags, _delim, name in folders:
        if name.lower() in names:
            return name
    return None


def _find_spam_folder(client: "_ImapClient") -> str | None:
    return _find_folder(client, _SPAM_NAMES)


def _find_drafts_folder(client: "_ImapClient") -> str | None:
    return _find_folder(client, _DRAFTS_NAMES)


def _imap_date(iso_str: str) -> str:
    from datetime import datetime as _dt
    try:
        return _dt.strptime(iso_str[:10], "%Y-%m-%d").strftime(_IMAP_DATE_FMT)
    except (ValueError, IndexError):
        return iso_str


def _hdr(msg: _email_mod.message.Message, name: str) -> str:
    v = msg.get(name)
    return str(v) if v else ""


def _safe_date(s: str) -> str:
    if not s:
        return ""
    try:
        return _email_mod.utils.parsedate_to_datetime(s).isoformat()
    except Exception:
        return ""


def parse_headers(uid: int, header_bytes: bytes) -> dict[str, object]:
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


def _bodystructure_has_attachments(bs: object) -> bool:
    """Return True when an imapclient BODYSTRUCTURE contains an attachment
    content-disposition or a filename/name parameter.  Never raises."""
    try:
        def _walk(node: object) -> bool:
            if isinstance(node, bytes):
                if node.lower() in (b"attachment", b"filename", b"name"):
                    return True
            elif isinstance(node, str):
                if node.lower() in ("attachment", "filename", "name"):
                    return True
            elif isinstance(node, (tuple, list)):
                return any(_walk(child) for child in node)
            return False
        return _walk(bs)
    except Exception:
        return False


def _is_attachment(part: _email_mod.message.Message) -> bool:
    # mypy's email.Message stub lacks is_attachment; cast through Any to get
    # past the type checker while keeping the runtime behaviour unchanged.
    _p = cast("object", part)
    is_att = getattr(_p, "is_attachment", lambda: False)()
    has_filename = part.get_filename()
    maintype = part.get_content_maintype()
    return bool(is_att or (has_filename and maintype != "multipart"))


def extract_body(raw_bytes: bytes) -> str:
    msg = _email_mod.message_from_bytes(raw_bytes, policy=_email_mod.policy.default)
    parts = list(msg.walk()) if msg.is_multipart() else [msg]
    chunks: list[str] = []
    for p in parts:
        if _is_attachment(p):
            continue
        if p.get_content_type() == "text/plain":
            c = p.get_content()
            if isinstance(c, str):
                chunks.append(c)
    return "\n".join(chunks)


def _part_payload_bytes(p: _email_mod.message.Message) -> bytes:
    """Decoded payload of an attachment part.

    ``message/rfc822`` parts return ``None`` from ``get_payload(decode=True)``;
    there the inner message is serialized whole instead."""
    if p.get_content_maintype() == "message":
        inner = p.get_payload()
        if isinstance(inner, list) and inner:
            return cast("_email_mod.message.Message", inner[0]).as_bytes()
        return b""
    raw = p.get_payload(decode=True)
    return raw if isinstance(raw, bytes) else b""


def attachment_meta(raw_bytes: bytes) -> list[dict[str, object]]:
    """filename/size/mime of each attachment part — read-time discovery
    metadata, computed without writing anything to disk."""
    msg = _email_mod.message_from_bytes(raw_bytes, policy=_email_mod.policy.default)
    parts = list(msg.walk()) if msg.is_multipart() else [msg]
    meta: list[dict[str, object]] = []
    for n, p in enumerate(part for part in parts if _is_attachment(part)):
        meta.append({
            "filename": p.get_filename() or f"attachment_{n}",
            "size": len(_part_payload_bytes(p)),
            "mime": p.get_content_type(),
        })
    return meta


def extract_attachments(raw_bytes: bytes, dest_dir: str) -> list[dict[str, object]]:
    msg = _email_mod.message_from_bytes(raw_bytes, policy=_email_mod.policy.default)
    parts = list(msg.walk()) if msg.is_multipart() else [msg]
    existing: set[str] = set()
    if os.path.isdir(dest_dir):
        existing = set(os.listdir(dest_dir))
    results: list[dict[str, object]] = []
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for p in parts:
        if not _is_attachment(p):
            continue
        raw_name = p.get_filename() or ""
        name = safe_filename(raw_name)
        if not name:
            name = f"attachment_{n}"
            ext = mimetypes.guess_extension(p.get_content_type())
            if ext:
                name += ext
        # Handle collision within this dest_dir
        stem, ext = os.path.splitext(name)
        candidate = name
        counter = 2
        while candidate in existing:
            candidate = f"{stem}_{counter}{ext}"
            counter += 1
        existing.add(candidate)
        payload_bytes = _part_payload_bytes(p)
        path = os.path.join(dest_dir, candidate)
        with open(path, "wb") as fh:
            fh.write(payload_bytes)
        results.append({
            "path": path,
            "filename": candidate,
            "size": len(payload_bytes),
            "mime": p.get_content_type(),
        })
        n += 1
    return results


def build_outgoing_message(from_addr: str, params: dict[str, object]) -> _email_mod.message.EmailMessage:
    """Assemble an outgoing message from send-family params.

    Pure builder shared by send and draft: no network, no Message-ID —
    callers that need one (SMTP send) stamp it themselves.  Attachment
    paths are read here, so file errors surface before any connection."""
    msg = _email_mod.message.EmailMessage()
    msg.set_content((cast(str, params.get("body")) or "").strip())
    msg["From"] = from_addr
    msg["To"] = (cast(str, params.get("to")) or "").strip()
    msg["Subject"] = (cast(str, params.get("subject")) or "").strip()
    cc = (cast(str, params.get("cc")) or "").strip()
    if cc:
        msg["Cc"] = cc
    in_reply_to = (cast(str, params.get("in_reply_to")) or "").strip()
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    for path in cast("list[str]", params.get("attachments") or []):
        with open(path, "rb") as fh:
            data = fh.read()
        mime, _encoding = mimetypes.guess_type(path)
        maintype, subtype = (mime or "application/octet-stream").split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(path))
    return msg


# ---------------------------------------------------------------------------
# ImapHandler
# ---------------------------------------------------------------------------

class ImapHandler:
    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def open_client(self, host: str, port: int, tls: bool,
                    email: str, password: str, timeout: int = 30) -> "_ImapClient | None":
        import imapclient
        try:
            c = cast("_ImapClient", imapclient.IMAPClient(host, port=port, ssl=tls, timeout=timeout))
            c.login(email, password)
            return c
        except Exception as exc:
            logger.exception("[imap_handler] open_client: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, client: "_ImapClient", watermark: int | None) -> tuple[list[dict[str, object]], int | None]:
        wm = watermark or 0
        try:
            client.select_folder("INBOX", readonly=True)
            since = (utc_now() - timedelta(days=_INITIAL_DAYS)).strftime(_IMAP_DATE_FMT)
            uids = client.search(["UID", f"{wm + 1}:*"] if wm else ["SINCE", since])
            if not uids:
                return [], wm or None
            raw = client.fetch(uids, [_IMAP_FETCH_HEADER])
            results, max_uid = [], wm
            for u, d in raw.items():
                if u > wm:
                    results.append(parse_headers(u, d[_IMAP_FETCH_HEADER]))
                    max_uid = max(max_uid, u)
            new_watermark = max_uid if max_uid > wm else (wm or None)
            return results, new_watermark
        except Exception as exc:
            logger.exception("[imap_handler] ingest: %s", exc)
            return [], wm or None

    # ------------------------------------------------------------------
    # Understand
    # ------------------------------------------------------------------

    def understand(self, items: list[dict[str, object]]) -> None:
        if not items:
            return
        from capabilities.contact_resolver import index_person
        from capabilities.mail_capability.email_triage import classify_email
        for item in items:
            item["triage"] = classify_email(item)
            item["is_thread"] = bool(item.get("in_reply_to"))
            if item.get("from_addr"):
                index_person(cast(str, item["from_addr"]), cast("str | None", item.get("from_name")), source="imap")

    # ------------------------------------------------------------------
    # WorldState inbox hint
    # ------------------------------------------------------------------

    def search(self, client: "_ImapClient", params: dict[str, object]) -> dict[str, object]:
        from capabilities.mail_capability.email_triage import classify_email
        try:
            client.select_folder("INBOX", readonly=True)
            criteria: list[object] = []
            query: dict[str, object] = {}
            sender  = (cast(str, params.get("sender"))      or "").strip()
            subject = (cast(str, params.get("subject"))     or "").strip()
            keyword = (cast(str, params.get("keyword"))     or "").strip()
            date_from = (cast(str, params.get("date_from")) or "").strip()
            date_to   = (cast(str, params.get("date_to"))   or "").strip()
            if sender:
                criteria.extend(["FROM", sender])
                query["sender"] = sender
            if subject:
                criteria.extend(["SUBJECT", subject])
                query["subject"] = subject
            if keyword:
                criteria.extend(["TEXT", keyword])
                query["keyword"] = keyword
            if date_from:
                criteria.extend(["SINCE", _imap_date(date_from)])
                query["date_from"] = date_from
            if date_to:
                criteria.extend(["BEFORE", _imap_date(date_to)])
                query["date_to"] = date_to
            if params.get("unanswered"):
                criteria.append("UNANSWERED")
                query["unanswered"] = True
            triage_filter = (cast(str, params.get("triage")) or "").lower()
            if triage_filter:
                query["triage"] = triage_filter
            if not criteria:
                window_start = utc_now() - timedelta(days=7)
                criteria.extend(["SINCE", window_start.strftime(_IMAP_DATE_FMT)])
                query["date_from"] = window_start.strftime("%Y-%m-%d")
            uids = client.search(criteria)
            uids = uids[-_SEARCH_LIMIT:] if len(uids) > _SEARCH_LIMIT else uids
            if not uids:
                return {"emails": [], "count": 0, "query": query}
            raw = client.fetch(uids, [_IMAP_FETCH_HEADER, b"BODYSTRUCTURE"])
            results = []
            for u in sorted(raw.keys(), reverse=True):
                item = parse_headers(u, raw[u][_IMAP_FETCH_HEADER])
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
                    "has_attachments": _bodystructure_has_attachments(raw[u].get(b"BODYSTRUCTURE")),
                })
            return {"emails": results, "count": len(results), "query": query}
        except Exception as exc:
            logger.exception("[imap_handler] search: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Read full body
    # ------------------------------------------------------------------

    def read_email(self, client: "_ImapClient", params: dict[str, object]) -> dict[str, object]:
        uid = params.get("uid")
        if not uid:
            return {"error": "uid is required"}
        try:
            uid = int(cast(int, uid))
        except (TypeError, ValueError):
            return {"error": "uid must be an integer"}
        try:
            client.select_folder("INBOX", readonly=True)
            raw = client.fetch([uid], [b"RFC822"])
            if uid not in raw:
                return {"error": f"Email UID {uid} not found"}
            raw_bytes = raw[uid][b"RFC822"]
            headers = parse_headers(uid, raw_bytes)
            result: dict[str, object] = {
                "uid": uid,
                "message_id": headers.get("message_id", ""),
                "subject": headers.get("subject", ""),
                "from_name": headers.get("from_name", ""),
                "from_addr": headers.get("from_addr", ""),
                "date": headers.get("date", ""),
                "body": extract_body(raw_bytes),
            }
            attachments = attachment_meta(raw_bytes)
            if attachments:
                result["attachments"] = attachments
            return result
        except Exception as exc:
            logger.exception("[imap_handler] read_email: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Download attachments
    # ------------------------------------------------------------------

    def download_attachments(self, client: "_ImapClient", params: dict[str, object]) -> dict[str, object]:
        uid = params.get("uid")
        if not uid:
            return {"error": "uid is required"}
        try:
            uid = int(cast(int, uid))
        except (TypeError, ValueError):
            return {"error": "uid must be an integer"}
        try:
            client.select_folder("INBOX", readonly=True)
            raw = client.fetch([uid], [b"RFC822"])
            if uid not in raw:
                return {"error": f"Email UID {uid} not found"}
            raw_bytes = raw[uid][b"RFC822"]
            dir_name = new_tmp_path(f"email_{uid}_{uuid.uuid4().hex[:8]}")
            os.makedirs(dir_name, exist_ok=True)
            files = extract_attachments(raw_bytes, dir_name)
            return {"files": files, "uid": uid, "count": len(files)}
        except Exception as exc:
            logger.exception("[imap_handler] download_attachments: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def draft_email(self, client: "_ImapClient", *, from_addr: str, params: dict[str, object]) -> dict[str, object]:
        to = (cast(str, params.get("to")) or "").strip()
        subject = (cast(str, params.get("subject")) or "").strip()
        body = (cast(str, params.get("body")) or "").strip()
        if not to or not subject or not body:
            return {"error": "to, subject, and body are required"}
        # The draft action does not support attachments: strip the key so the
        # shared builder cannot attach files on this path.
        draft_params = {k: v for k, v in params.items() if k != "attachments"}
        msg = build_outgoing_message(from_addr, draft_params)
        drafts = _find_drafts_folder(client)
        if not drafts:
            return {"error": "No drafts folder found on this mail server"}
        try:
            client.append(drafts, msg.as_bytes(), flags=[b"\\Draft", b"\\Seen"])
            return {"success": True, "to": to, "subject": subject, "folder": drafts}
        except Exception as exc:
            logger.exception("[imap_handler] draft_email: %s", exc)
            return {"error": str(exc)}

    def send_email(self, *, creds: SmtpCreds, params: dict[str, object]) -> dict[str, object]:
        import smtplib

        to = (cast(str, params.get("to")) or "").strip()
        subject = (cast(str, params.get("subject")) or "").strip()
        body = (cast(str, params.get("body")) or "").strip()

        if not to or not subject or not body:
            return {"error": "to, subject, and body are required"}

        try:
            msg = build_outgoing_message(creds.from_addr, params)
        except OSError as exc:
            # File-read failures name the offending attachment path; SMTP
            # errors (also OSError subclasses) are handled separately below.
            logger.exception("[imap_handler] send_email: %s", exc)
            return {"error": f"attachment error: {exc}"}

        # Stamp our own Message-ID: the SMTP server mints one only AFTER send
        # when the client supplies none, so a self-stamped id is the only key
        # knowable at send time for the emails_sent ledger.
        stamp = _email_mod.utils.make_msgid()
        msg["Message-ID"] = stamp

        try:
            if creds.tls:
                with smtplib.SMTP_SSL(creds.host, creds.port, timeout=_SMTP_TIMEOUT) as server:
                    server.login(creds.from_addr, creds.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(creds.host, creds.port, timeout=_SMTP_TIMEOUT) as server:
                    server.ehlo()
                    try:
                        server.starttls()
                        server.ehlo()
                    except smtplib.SMTPNotSupportedError:
                        logger.warning("[imap_handler] STARTTLS not supported by %s:%s", creds.host, creds.port)
                    try:
                        server.login(creds.from_addr, creds.password)
                    except smtplib.SMTPNotSupportedError:
                        logger.warning("[imap_handler] AUTH not supported by %s:%s — sending unauthenticated", creds.host, creds.port)
                    server.send_message(msg)
            result: dict[str, object] = {"success": True, "to": to, "subject": subject, "message_id": stamp}
            attached = cast("list[str]", params.get("attachments") or [])
            if attached:
                result["attachments"] = [os.path.basename(p) for p in attached]
            return result
        except Exception as exc:
            logger.exception("[imap_handler] send_email: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Manage
    # ------------------------------------------------------------------

    def manage_email(self, client: "_ImapClient", params: dict[str, object]) -> dict[str, object]:
        uid = params.get("uid")
        if uid is None:
            return {"error": "uid is required"}
        try:
            uid = int(cast(int, uid))
        except (TypeError, ValueError):
            return {"error": "uid must be an integer"}
        operation = params.get("operation")
        _valid_ops = {"delete", "mark_read", "mark_important", "move_to_spam"}
        if operation not in _valid_ops:
            return {"error": f"operation must be one of: {', '.join(sorted(_valid_ops))}"}
        try:
            if operation == "delete":
                client.select_folder("INBOX")
                client.delete_messages([uid])
                client.expunge([uid])
                return {"success": True, "uid": uid, "operation": "delete"}
            elif operation == "mark_read":
                client.select_folder("INBOX")
                client.set_flags([uid], [b"\\Seen"])
                return {"success": True, "uid": uid, "operation": "mark_read"}
            elif operation == "mark_important":
                client.select_folder("INBOX")
                client.set_flags([uid], [b"\\Flagged"])
                return {"success": True, "uid": uid, "operation": "mark_important"}
            else:  # move_to_spam
                client.select_folder("INBOX")
                spam = _find_spam_folder(client)
                if not spam:
                    return {"error": "No spam/junk folder found on this mail server"}
                try:
                    client.move([uid], spam)
                except Exception:
                    client.copy([uid], spam)
                    client.delete_messages([uid])
                    client.expunge([uid])
                return {"success": True, "uid": uid, "operation": "move_to_spam"}
        except Exception as exc:
            logger.exception("[imap_handler] manage_email: %s", exc)
            return {"error": str(exc)}
