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

_INITIAL_DAYS = 7


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
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self):
        self._connected = False

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
        return items

    def monitor(self):
        pass

    def act(self, action, params):
        return {"error": f"'{action}' not implemented"}

    def get_tools(self):
        return []
