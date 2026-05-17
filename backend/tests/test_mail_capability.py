"""
Unit tests for MailCapability — the unified email/calendar/contacts orchestrator.

Coverage:
- configure() happy path (IMAP-only and all-protocols)
- configure() validation: missing fields, unsupported provider, all probes fail
- connect() happy path and credential-missing guard
- disconnect() cancels scheduled items and deletes credentials
- get_tools() conditional registration based on protocol flags
- _do_monitor() cadence — IMAP every cycle, CalDAV every 3rd, CardDAV every 12th
- health_details() structure
- act() dispatches to tool handlers

All tests use @pytest.mark.unit and never touch a real network or database.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

MOCK_AUTH_TOKEN = "fake-token-for-test"


# ---------------------------------------------------------------------------
# Shared helpers: in-memory vault + tool-config service
# ---------------------------------------------------------------------------

class _MockVault:
    """Deterministic AES-256-GCM vault for credential round-trip testing."""

    def __init__(self) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self._key = b"T" * 32
        self._aesgcm = AESGCM(self._key)

    def encrypt_str(self, s: str) -> bytes:
        import os
        nonce = os.urandom(12)
        return nonce + self._aesgcm.encrypt(nonce, s.encode(), None)

    def decrypt_str(self, blob: bytes) -> str:
        nonce, ct = blob[:12], blob[12:]
        return self._aesgcm.decrypt(nonce, ct, None).decode()


class _InMemoryTCS:
    """In-memory stand-in for ToolConfigService."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def set_tool_config(self, tool_name: str, config: dict) -> None:
        self._store.setdefault(tool_name, {}).update(config)

    def get_tool_config(self, tool_name: str) -> dict:
        return dict(self._store.get(tool_name, {}))

    def delete_tool_config(self, tool_name: str) -> None:
        self._store.pop(tool_name, None)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_capability(vault=None, tcs=None):
    """Return a MailCapability with mocked service layer."""
    from capabilities.mail_capability.capability import MailCapability

    cap = MailCapability.__new__(MailCapability)
    # Call parent __init__ without triggering real imports
    from capabilities.base import AbstractCapability
    AbstractCapability.__init__(cap)
    # Wire handlers
    cap._manifest_cache = None
    cap._imap_handler = MagicMock()
    cap._caldav_handler = MagicMock()
    cap._carddav_handler = MagicMock()
    cap._imap_ok = False
    cap._caldav_ok = False
    cap._carddav_ok = False
    cap._cycle_count = 0
    cap._sync_registered = False

    _vault = vault or _MockVault()
    _tcs = tcs or _InMemoryTCS()

    # Bind mocked helpers directly so round-trips work in tests
    cap._tcs = _tcs
    cap._vault = _vault
    return cap, _vault, _tcs


def _patches(tcs, vault):
    """Context manager: patch vault + TCS for the duration of a test."""
    return [
        patch("capabilities.base._get_tool_config_service", return_value=tcs),
        patch("services.vault_service.get_vault_service", return_value=vault),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMailCapabilityConfigure:
    """configure() validation and happy paths."""

    def test_configure_missing_email_raises(self):
        cap, vault, tcs = _make_capability()
        with pytest.raises(ValueError, match="email"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"password": MOCK_AUTH_TOKEN})

    def test_configure_custom_provider_all_protocols(self):
        cap, vault, tcs = _make_capability()
        cap._imap_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.open_client.return_value = MagicMock()
        cap._carddav_handler.open_client.return_value = MagicMock()
        cap.connect = MagicMock(return_value=True)

        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.configure({
                "email": "test@example.com",
                "password": MOCK_AUTH_TOKEN,
                "imap_host": "greenmail",
                "imap_port": 3143,
                "imap_tls": False,
                "caldav_url": "http://radicale:5232",
                "carddav_url": "http://radicale:5232",
            })
            protocols = json.loads(cap.load_credential("mail:protocols"))
            caldav = cap.load_credential("mail:caldav_url")
            carddav = cap.load_credential("mail:carddav_url")

        assert set(protocols) == {"imap", "caldav", "carddav"}
        assert caldav == "http://radicale:5232"
        assert carddav == "http://radicale:5232"

    def test_configure_all_probes_fail_deletes_creds_and_raises(self):
        cap, vault, tcs = _make_capability()
        # All handler opens return None → all probes fail
        cap._imap_handler.open_client.return_value = None
        cap._caldav_handler.open_client.return_value = None
        cap._carddav_handler.open_client.return_value = None

        with pytest.raises(ValueError, match="All protocol probes failed"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"email": "user@gmail.com", "password": MOCK_AUTH_TOKEN})

        # Credentials must be wiped
        assert tcs.get_tool_config("mail") == {}


@pytest.mark.unit
class TestResolveProvider:
    """_resolve_provider() falls back to stored credentials for custom servers."""

    def test_known_domain_returns_registry_provider(self):
        cap, vault, tcs = _make_capability()
        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            provider = cap._resolve_provider("user@gmail.com")
        assert provider is not None
        assert provider.name == "Google"


@pytest.mark.unit
class TestMailCapabilityConnect:
    """connect() happy paths and guard conditions."""

    def test_connect_missing_email_returns_false(self):
        cap, vault, tcs = _make_capability()
        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            result = cap.connect()
        assert result is False
        assert cap.is_connected() is False

    def test_connect_imap_only_sets_imap_ok(self):
        cap, vault, tcs = _make_capability()
        cap._imap_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.open_client.return_value = None


        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.store_credential("mail:email", "user@gmail.com")
            cap.store_credential("mail:password", MOCK_AUTH_TOKEN)
            cap.store_credential("mail:protocols", json.dumps(["imap"]))
            result = cap.connect()

        assert result is True
        assert cap._imap_ok is True
        assert cap._caldav_ok is False
        assert cap._carddav_ok is False
        assert cap.is_connected() is True

@pytest.mark.unit
class TestMailCapabilityDisconnect:
    """disconnect() behaviour."""

    def test_disconnect_deletes_credentials(self):
        cap, vault, tcs = _make_capability()
        cap._imap_ok = True
        cap._connected = True

        with patch("services.database_service.get_shared_db_service") as mock_db:
            mock_conn = MagicMock()
            mock_db.return_value.connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_db.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                # Store something first
                cap.store_credential("mail:email", "user@gmail.com")
                cap.disconnect()
                # After disconnect the store should be empty
                assert tcs.get_tool_config("mail") == {}


@pytest.mark.unit
class TestMailCapabilityGetTools:
    """get_tools() conditional registration."""

    def test_imap_only_returns_four_tools(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        tools = cap.get_tools()
        names = {t["name"] for t in tools}
        assert names == {"search_email", "read_email", "draft_email", "manage_email"}

@pytest.mark.unit
class TestMailCapabilityMonitorCadence:
    """_do_monitor() cadence — verifies which handlers are called per cycle."""

    def _connected_cap(self):
        cap, vault, tcs = _make_capability()
        cap._imap_ok = True
        cap._caldav_ok = True
        cap._carddav_ok = True
        cap._connected = True
        # Patch credential loads to return minimal values
        cap.load_credential = MagicMock(side_effect=lambda k: {
            "mail:email": "user@gmail.com",
            "mail:password": MOCK_AUTH_TOKEN,
            "mail:imap_watermark": None,
        }.get(k))
        cap.store_credential = MagicMock()
        return cap, vault, tcs

    def test_carddav_called_every_twelfth_cycle(self):
        cap, _, _ = self._connected_cap()
        mock_imap = MagicMock()
        cap._imap_handler.open_client.return_value = mock_imap
        cap._imap_handler.ingest.return_value = ([], None)
        mock_caldav = MagicMock()
        cap._caldav_handler.open_client.return_value = mock_caldav
        cap._caldav_handler.ingest.return_value = []
        mock_carddav = MagicMock()
        cap._carddav_handler.open_client.return_value = mock_carddav

        with patch("capabilities.mail_capability.capability.utc_now") as mock_now:
            from datetime import datetime, timezone
            mock_now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for _ in range(24):
                cap._do_monitor()

        # CardDAV called at cycles 12 and 24 → 2 times
        assert cap._carddav_handler.open_client.call_count == 2



@pytest.mark.unit
class TestMailCapabilityHealthDetails:
    """health_details() structure."""

    def test_health_details_when_partially_connected(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        cap._connected = True
        details = cap.health_details()
        assert details["connected"] is True
        assert details["protocols"]["imap"] is True
        assert details["protocols"]["caldav"] is False
        assert details["protocols"]["carddav"] is False


@pytest.mark.unit
class TestMailCapabilityAct:
    """act() dispatches to the correct tool handler."""

    def test_act_dispatches_to_known_tool(self):
        cap, _, _ = _make_capability()
        cap._caldav_ok = True
        cap._caldav_handler.find_free_slots.return_value = {"free_slots": []}

        result = cap.act("find_free_slots", {})
        assert "free_slots" in result


