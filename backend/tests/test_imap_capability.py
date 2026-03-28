"""Unit tests for ImapCapability — credential and connect skeleton."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_CREDS = {
    "imap:host": "imap.gmail.com", "imap:port": "993",
    "imap:password": "pw", "imap:email": "u@gmail.com", "imap:tls": "1",
}


def _make():
    from capabilities.imap_capability.capability import ImapCapability
    return ImapCapability()


@pytest.mark.unit
def test_identity():
    assert _make().get_id() == "imap"


@pytest.mark.unit
def test_configure_missing_fields():
    with pytest.raises(ValueError, match="email and password required"):
        _make().configure({"email": "u@gmail.com"})


@pytest.mark.unit
def test_configure_unsupported_domain():
    cap = _make()
    with patch.object(cap, "store_credential"):
        with pytest.raises(ValueError, match="Unsupported email provider"):
            cap.configure({"email": "u@unknown-xyz.com", "password": "pw"})


@pytest.mark.unit
def test_configure_known_provider():
    cap = _make()
    with patch.object(cap, "store_credential") as ms, \
         patch.object(cap, "connect", return_value=True):
        cap.configure({"email": "u@gmail.com", "password": "pw"})
    ms.assert_any_call("imap:host", "imap.gmail.com")


@pytest.mark.unit
def test_configure_connect_failure():
    cap = _make()
    with patch.object(cap, "store_credential"), \
         patch.object(cap, "connect", return_value=False), \
         patch.object(cap, "delete_credentials") as d:
        with pytest.raises(ValueError, match="Connection test failed"):
            cap.configure({"email": "u@gmail.com", "password": "bad"})
    d.assert_called_once()


@pytest.mark.unit
def test_connect_success():
    cap = _make()
    mc = MagicMock()
    with patch.object(cap, "load_credential", side_effect=_CREDS.get), \
         patch("imapclient.IMAPClient", return_value=mc):
        assert cap.connect() is True


@pytest.mark.unit
def test_connect_failure():
    cap = _make()
    with patch.object(cap, "load_credential", side_effect=_CREDS.get), \
         patch("imapclient.IMAPClient", side_effect=Exception("refused")):
        assert cap.connect() is False
