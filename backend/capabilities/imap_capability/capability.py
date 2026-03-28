"""ImapCapability — IMAP email capability skeleton.

Connects to IMAP servers and validates credentials.  Uses
provider_autodiscovery for settings.  imapclient imported lazily.
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from capabilities.base import AbstractCapability
from capabilities.provider_autodiscovery import discover_email_settings

logger = logging.getLogger(__name__)
_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"
_K_EMAIL = "imap:email"
_K_PW = "imap:password"
_K_HOST = "imap:host"
_K_PORT = "imap:port"
_K_TLS = "imap:tls"


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

    def ingest(self):
        return []

    def understand(self, items):
        return items

    def monitor(self):
        pass

    def act(self, action, params):
        return {"error": f"'{action}' not implemented"}

    def get_tools(self):
        return []
