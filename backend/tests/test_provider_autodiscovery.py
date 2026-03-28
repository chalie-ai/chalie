"""Unit tests for capabilities.provider_autodiscovery."""

import pytest
from capabilities.provider_autodiscovery import (
    PROVIDERS, ServerSettings, discover_email_settings, list_supported_providers,
)


@pytest.mark.unit
@pytest.mark.parametrize("email,expected_name,expected_imap_host", [
    ("alice@gmail.com", "Gmail", "imap.gmail.com"),
    ("bob@googlemail.com", "Gmail", "imap.gmail.com"),
    ("u@outlook.com", "Outlook", "outlook.office365.com"),
    ("u@hotmail.com", "Outlook", "outlook.office365.com"),
    ("u@live.com", "Outlook", "outlook.office365.com"),
    ("u@icloud.com", "iCloud", "imap.mail.me.com"),
    ("u@me.com", "iCloud", "imap.mail.me.com"),
    ("u@mac.com", "iCloud", "imap.mail.me.com"),
    ("u@fastmail.com", "Fastmail", "imap.fastmail.com"),
    ("u@fastmail.fm", "Fastmail", "imap.fastmail.com"),
    ("u@yahoo.com", "Yahoo Mail", "imap.mail.yahoo.com"),
    ("u@yahoo.co.uk", "Yahoo Mail", "imap.mail.yahoo.com"),
    ("u@proton.me", "ProtonMail", "127.0.0.1"),
    ("u@protonmail.com", "ProtonMail", "127.0.0.1"),
    ("u@zoho.com", "Zoho Mail", "imap.zoho.com"),
    ("u@aol.com", "AOL Mail", "imap.aol.com"),
    ("u@gmx.com", "GMX Mail", "imap.gmx.com"),
    ("u@gmx.net", "GMX Mail", "imap.gmx.com"),
    # edge cases: case, whitespace, multiple @
    ("User@Gmail.COM", "Gmail", "imap.gmail.com"),
    ("user@ gmail.com ", "Gmail", "imap.gmail.com"),
    ("weird@name@gmail.com", "Gmail", "imap.gmail.com"),
])
def test_known_provider(email, expected_name, expected_imap_host):
    s = discover_email_settings(email)
    assert s is not None
    assert s.provider_name == expected_name
    assert s.imap.host == expected_imap_host


@pytest.mark.unit
@pytest.mark.parametrize("email", [
    "", "not-an-email", "@", "user@", "user@unknown-startup.xyz", "ceo@mycorp.io",
])
def test_returns_none_for_invalid_or_unknown(email):
    assert discover_email_settings(email) is None


@pytest.mark.unit
def test_full_settings_gmail_and_proton():
    g = discover_email_settings("a@gmail.com")
    assert g.imap == ServerSettings("imap.gmail.com", 993, True)
    assert g.smtp == ServerSettings("smtp.gmail.com", 465, True)
    assert g.requires_app_password is True
    p = discover_email_settings("a@proton.me")
    assert p.imap == ServerSettings("127.0.0.1", 1143, False)
    assert p.requires_app_password is False


@pytest.mark.unit
def test_list_supported_providers():
    names = list_supported_providers()
    assert names == sorted(names)
    assert len(names) == len({s.provider_name for s in PROVIDERS.values()})
    assert "Gmail" in names and "Outlook" in names and "iCloud" in names


@pytest.mark.unit
def test_server_settings_frozen():
    s = ServerSettings("h", 993, True)
    with pytest.raises(AttributeError):
        s.host = "x"
