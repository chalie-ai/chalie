"""Unit tests for capabilities.provider_autodiscovery (shim over mail_capability.providers)."""

import pytest
from capabilities.provider_autodiscovery import (
    PROVIDERS, ServerSettings, discover_email_settings, list_supported_providers,
)


@pytest.mark.unit
@pytest.mark.parametrize("email,expected_name,expected_imap_host", [
    ("alice@gmail.com", "Google", "imap.gmail.com"),
    ("u@outlook.com", "Outlook", "outlook.office365.com"),
    ("u@hotmail.com", "Outlook", "outlook.office365.com"),
    ("u@icloud.com", "Apple", "imap.mail.me.com"),
    ("u@yahoo.com", "Yahoo", "imap.mail.yahoo.com"),
    # edge cases: case, whitespace, multiple @
    ("User@Gmail.COM", "Google", "imap.gmail.com"),
    ("weird@name@gmail.com", "Google", "imap.gmail.com"),
])
def test_known_provider(email, expected_name, expected_imap_host):
    s = discover_email_settings(email)
    assert s is not None
    assert s.provider_name == expected_name
    assert s.imap.host == expected_imap_host


@pytest.mark.unit
@pytest.mark.parametrize("email", [
    "", "not-an-email", "user@unknown-startup.xyz",
])
def test_returns_none_for_invalid_or_unknown(email):
    assert discover_email_settings(email) is None


@pytest.mark.unit
def test_full_settings_gmail():
    g = discover_email_settings("a@gmail.com")
    assert g.imap == ServerSettings("imap.gmail.com", 993, True)
    assert g.smtp == ServerSettings("smtp.gmail.com", 465, True)
    assert g.requires_app_password is True


@pytest.mark.unit
def test_list_supported_providers():
    names = list_supported_providers()
    assert names == sorted(names)
    assert len(names) == len({s.provider_name for s in PROVIDERS.values()})
    assert "Google" in names and "Outlook" in names and "Apple" in names


@pytest.mark.unit
def test_server_settings_frozen():
    s = ServerSettings("h", 993, True)
    with pytest.raises(AttributeError):
        s.host = "x"
