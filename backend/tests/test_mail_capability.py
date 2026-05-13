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

    def _patch_cred_helpers():
        with patch(
            "capabilities.base._get_tool_config_service", return_value=_tcs
        ), patch(
            "services.vault_service.get_vault_service", return_value=_vault
        ):
            pass

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
class TestMailCapabilityIdentity:
    """Identity methods."""

    def test_get_id(self):
        from capabilities.mail_capability.capability import MailCapability
        cap = MailCapability.__new__(MailCapability)
        from capabilities.base import AbstractCapability
        AbstractCapability.__init__(cap)
        cap._manifest_cache = None
        cap._imap_handler = MagicMock()
        cap._caldav_handler = MagicMock()
        cap._carddav_handler = MagicMock()
        cap._imap_ok = cap._caldav_ok = cap._carddav_ok = False
        cap._cycle_count = 0
        cap._sync_registered = False
        assert cap.get_id() == "mail"

    def test_get_manifest_returns_mail_id(self):
        from capabilities.mail_capability.capability import MailCapability
        cap = MailCapability.__new__(MailCapability)
        from capabilities.base import AbstractCapability
        AbstractCapability.__init__(cap)
        cap._manifest_cache = None
        cap._imap_handler = MagicMock()
        cap._caldav_handler = MagicMock()
        cap._carddav_handler = MagicMock()
        cap._imap_ok = cap._caldav_ok = cap._carddav_ok = False
        cap._cycle_count = 0
        cap._sync_registered = False
        manifest = cap.get_manifest()
        assert manifest["id"] == "mail"
        assert manifest["entry_class"] == "MailCapability"


@pytest.mark.unit
class TestMailCapabilityConfigure:
    """configure() validation and happy paths."""

    def test_configure_missing_email_raises(self):
        cap, vault, tcs = _make_capability()
        with pytest.raises(ValueError, match="email"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"password": "secret"})

    def test_configure_missing_password_raises(self):
        cap, vault, tcs = _make_capability()
        with pytest.raises(ValueError, match="password"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"email": "user@gmail.com"})

    def test_configure_unsupported_provider_raises(self):
        cap, vault, tcs = _make_capability()
        with pytest.raises(ValueError, match="Unsupported provider"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"email": "user@unknown-corp.io", "password": "x"})

    def test_configure_all_probes_fail_deletes_creds_and_raises(self):
        cap, vault, tcs = _make_capability()
        # All handler opens return None → all probes fail
        cap._imap_handler.open_client.return_value = None
        cap._caldav_handler.open_client.return_value = None
        cap._carddav_handler.open_client.return_value = None

        with pytest.raises(ValueError, match="All protocol probes failed"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"email": "user@gmail.com", "password": "app-pw"})

        # Credentials must be wiped
        assert tcs.get_tool_config("mail") == {}

    def test_configure_imap_only_success_stores_protocols(self):
        cap, vault, tcs = _make_capability()
        mock_imap_client = MagicMock()
        cap._imap_handler.open_client.return_value = mock_imap_client
        cap._caldav_handler.open_client.return_value = None
        cap._carddav_handler.open_client.return_value = None

        # Also mock connect() so the final connectivity test passes
        cap.connect = MagicMock(return_value=True)

        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.configure({"email": "user@outlook.com", "password": "app-pw"})

        # Outlook only has IMAP
        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            raw_protocols = cap.load_credential("mail:protocols")
        protocols = json.loads(raw_protocols)
        assert protocols == ["imap"]

    def test_configure_all_protocols_success(self):
        cap, vault, tcs = _make_capability()
        cap._imap_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.open_client.return_value = MagicMock()
        cap._carddav_handler.open_client.return_value = MagicMock()
        cap.connect = MagicMock(return_value=True)

        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.configure({"email": "user@gmail.com", "password": "app-pw"})
            raw = cap.load_credential("mail:protocols")

        protocols = json.loads(raw)
        assert set(protocols) == {"imap", "caldav", "carddav"}

    def test_configure_connect_failure_deletes_creds_and_raises(self):
        cap, vault, tcs = _make_capability()
        cap._imap_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.open_client.return_value = None
        cap._carddav_handler.open_client.return_value = None
        cap.connect = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Post-configure connect"):
            with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
                cap.configure({"email": "user@gmail.com", "password": "app-pw"})

        assert tcs.get_tool_config("mail") == {}


@pytest.mark.unit
class TestMailCapabilityConnect:
    """connect() happy paths and guard conditions."""

    def test_connect_missing_email_returns_false(self):
        cap, vault, tcs = _make_capability()
        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            result = cap.connect()
        assert result is False
        assert cap.is_connected() is False

    def test_connect_empty_protocols_returns_false(self):
        cap, vault, tcs = _make_capability()
        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.store_credential("mail:email", "user@gmail.com")
            cap.store_credential("mail:password", "pw")
            cap.store_credential("mail:protocols", json.dumps([]))
            result = cap.connect()
        assert result is False

    def test_connect_imap_only_sets_imap_ok(self):
        cap, vault, tcs = _make_capability()
        cap._imap_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.open_client.return_value = None


        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.store_credential("mail:email", "user@gmail.com")
            cap.store_credential("mail:password", "pw")
            cap.store_credential("mail:protocols", json.dumps(["imap"]))
            result = cap.connect()

        assert result is True
        assert cap._imap_ok is True
        assert cap._caldav_ok is False
        assert cap._carddav_ok is False
        assert cap.is_connected() is True

    def test_connect_all_protocols(self):
        cap, vault, tcs = _make_capability()
        cap._imap_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.open_client.return_value = MagicMock()
        cap._carddav_handler.open_client.return_value = MagicMock()


        with _patches(tcs, vault)[0], _patches(tcs, vault)[1]:
            cap.store_credential("mail:email", "user@gmail.com")
            cap.store_credential("mail:password", "pw")
            cap.store_credential("mail:protocols", json.dumps(["imap", "caldav", "carddav"]))
            result = cap.connect()

        assert result is True
        assert cap._imap_ok is True
        assert cap._caldav_ok is True
        assert cap._carddav_ok is True


@pytest.mark.unit
class TestMailCapabilityDisconnect:
    """disconnect() behaviour."""

    def test_disconnect_clears_all_flags(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        cap._caldav_ok = True
        cap._carddav_ok = True
        cap._connected = True

        with patch("capabilities.mail_capability.capability.MailCapability.delete_credentials"):
            with patch("services.database_service.get_shared_db_service") as mock_db:
                mock_conn = MagicMock()
                mock_db.return_value.connection.return_value.__enter__ = MagicMock(
                    return_value=mock_conn
                )
                mock_db.return_value.connection.return_value.__exit__ = MagicMock(
                    return_value=False
                )
                cap.disconnect()

        assert cap._imap_ok is False
        assert cap._caldav_ok is False
        assert cap._carddav_ok is False
        assert cap._connected is False

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

    def test_no_protocols_returns_empty(self):
        cap, _, _ = _make_capability()
        # All flags default to False
        tools = cap.get_tools()
        assert tools == []

    def test_imap_only_returns_four_tools(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        tools = cap.get_tools()
        names = {t["name"] for t in tools}
        assert names == {"search_email", "read_email", "send_email", "manage_email"}

    def test_caldav_only_returns_seven_tools(self):
        cap, _, _ = _make_capability()
        cap._caldav_ok = True
        tools = cap.get_tools()
        names = {t["name"] for t in tools}
        assert names == {
            "list_events", "get_event", "create_event",
            "update_event", "delete_event", "find_free_slots", "get_attendees",
        }

    def test_carddav_only_returns_two_tools(self):
        cap, _, _ = _make_capability()
        cap._carddav_ok = True
        tools = cap.get_tools()
        names = {t["name"] for t in tools}
        assert names == {"list_contacts", "get_contact"}

    def test_all_protocols_returns_thirteen_tools(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        cap._caldav_ok = True
        cap._carddav_ok = True
        tools = cap.get_tools()
        assert len(tools) == 13

    def test_all_tools_have_required_keys(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        cap._caldav_ok = True
        cap._carddav_ok = True
        for tool in cap.get_tools():
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "handler" in tool
            assert callable(tool["handler"])
            assert "timeout" in tool

    def test_imap_tool_returns_error_when_disconnected(self):
        cap, _, _ = _make_capability()
        cap._imap_ok = True
        tools = {t["name"]: t for t in cap.get_tools()}

        # Simulate mid-call disconnect
        cap._imap_ok = False
        result = tools["search_email"]["handler"]("", {})
        assert "error" in result

    def test_caldav_tool_returns_error_when_disconnected(self):
        cap, _, _ = _make_capability()
        cap._caldav_ok = True
        tools = {t["name"]: t for t in cap.get_tools()}

        cap._caldav_ok = False
        result = tools["list_events"]["handler"]("", {})
        assert "error" in result

    def test_carddav_tool_returns_error_when_disconnected(self):
        cap, _, _ = _make_capability()
        cap._carddav_ok = True
        tools = {t["name"]: t for t in cap.get_tools()}

        cap._carddav_ok = False
        result = tools["list_contacts"]["handler"]("", {})
        assert "error" in result


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
            "mail:password": "pw",
            "mail:imap_watermark": None,
        }.get(k))
        cap.store_credential = MagicMock()
        return cap, vault, tcs

    def test_imap_called_every_cycle(self):
        cap, _, _ = self._connected_cap()
        mock_client = MagicMock()
        cap._imap_handler.open_client.return_value = mock_client
        cap._imap_handler.ingest.return_value = ([], None)
        cap._caldav_handler.open_client.return_value = MagicMock()
        cap._caldav_handler.ingest.return_value = []
        cap._carddav_handler.open_client.return_value = MagicMock()

        with patch("capabilities.mail_capability.capability.utc_now") as mock_now:
            from datetime import datetime, timezone
            mock_now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for _ in range(3):
                cap._do_monitor()

        # IMAP should have been called 3 times (once per cycle)
        assert cap._imap_handler.open_client.call_count == 3

    def test_caldav_called_every_third_cycle(self):
        cap, _, _ = self._connected_cap()
        mock_imap = MagicMock()
        cap._imap_handler.open_client.return_value = mock_imap
        cap._imap_handler.ingest.return_value = ([], None)
        mock_caldav = MagicMock()
        cap._caldav_handler.open_client.return_value = mock_caldav
        cap._caldav_handler.ingest.return_value = []
        cap._carddav_handler.open_client.return_value = MagicMock()

        with patch("capabilities.mail_capability.capability.utc_now") as mock_now:
            from datetime import datetime, timezone
            mock_now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for _ in range(6):
                cap._do_monitor()

        # CalDAV called at cycles 3 and 6 → 2 times
        assert cap._caldav_handler.open_client.call_count == 2

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

    def test_monitor_reconnects_if_disconnected(self):
        cap, _, _ = self._connected_cap()
        cap._connected = False
        cap._imap_ok = False
        cap._caldav_ok = False
        cap._carddav_ok = False

        # connect() will do nothing (no creds) — just verify it's called
        cap.connect = MagicMock(return_value=False)
        cap._do_monitor()
        cap.connect.assert_called_once()


@pytest.mark.unit
class TestMailCapabilityHealthDetails:
    """health_details() structure."""

    def test_health_details_when_disconnected(self):
        cap, _, _ = _make_capability()
        details = cap.health_details()
        assert details == {
            "connected": False,
            "protocols": {"imap": False, "caldav": False, "carddav": False},
        }

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

    def test_act_unknown_action_returns_error(self):
        cap, _, _ = _make_capability()
        result = cap.act("launch_rocket", {})
        assert "error" in result

    def test_act_dispatches_to_known_tool(self):
        cap, _, _ = _make_capability()
        cap._caldav_ok = True
        cap._caldav_handler.list_events.return_value = {"events": [], "count": 0}

        result = cap.act("list_events", {"limit": 10})
        assert "events" in result


