from typing import cast

import pytest

from capabilities.mail_capability.providers import ServerSettings
from capabilities.provider_autodiscovery import (
    EmailProviderSettings,
    discover_email_settings, list_supported_providers,
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
def test_known_provider(email: str, expected_name: str, expected_imap_host: str) -> None:
    s = discover_email_settings(email)
    assert s is not None
    assert s.provider_name == expected_name
    assert s.imap.host == expected_imap_host


@pytest.mark.unit
@pytest.mark.parametrize("email", [
    "", "not-an-email", "user@unknown-startup.xyz",
])
def test_returns_none_for_invalid_or_unknown(email: str) -> None:
    assert discover_email_settings(email) is None


@pytest.mark.unit
def test_full_settings_gmail() -> None:
    g = discover_email_settings("a@gmail.com")
    assert cast(EmailProviderSettings, g).imap == ServerSettings("imap.gmail.com", 993, True)
    assert cast(EmailProviderSettings, g).smtp == ServerSettings("smtp.gmail.com", 465, True)
    assert cast(EmailProviderSettings, g).requires_app_password is True


@pytest.mark.unit
def test_list_supported_providers() -> None:
    names = list_supported_providers()
    assert "Google" in names and "Outlook" in names and "Apple" in names
