"""Unit tests for the unified mail provider registry."""

import pytest

from capabilities.mail_capability.providers import (
    ServerSettings,
    discover_provider,
    list_supported_providers,
)


@pytest.mark.unit
class TestDiscoverProvider:
    """discover_provider should resolve email domains to UnifiedProvider."""

    def test_gmail(self):
        p = discover_provider("user@gmail.com")
        assert p is not None
        assert p.name == "Google"
        assert p.imap == ServerSettings("imap.gmail.com", 993, True)
        assert p.smtp == ServerSettings("smtp.gmail.com", 465, True)
        assert p.caldav_url is not None
        assert p.carddav_url is not None
        assert p.requires_app_password is True

    def test_googlemail_alias(self):
        assert discover_provider("x@googlemail.com") is discover_provider("x@gmail.com")

    def test_apple_icloud(self):
        p = discover_provider("user@icloud.com")
        assert p is not None
        assert p.name == "Apple"
        assert p.caldav_url is not None
        assert p.carddav_url is not None

    def test_apple_me_alias(self):
        assert discover_provider("x@me.com") is discover_provider("x@icloud.com")

    def test_apple_mac_alias(self):
        assert discover_provider("x@mac.com") is discover_provider("x@icloud.com")

    def test_yahoo(self):
        p = discover_provider("user@yahoo.com")
        assert p is not None
        assert p.name == "Yahoo"
        assert p.imap is not None
        assert p.caldav_url is not None
        assert p.carddav_url is None  # Yahoo deprecated CardDAV

    def test_yahoo_couk_alias(self):
        assert discover_provider("x@yahoo.co.uk") is discover_provider("x@yahoo.com")

    def test_outlook(self):
        p = discover_provider("user@outlook.com")
        assert p is not None
        assert p.name == "Outlook"
        assert p.imap is not None
        assert p.caldav_url is None  # No CalDAV (Graph API only)
        assert p.carddav_url is None

    def test_hotmail_alias(self):
        assert discover_provider("x@hotmail.com") is discover_provider("x@outlook.com")

    def test_live_alias(self):
        assert discover_provider("x@live.com") is discover_provider("x@outlook.com")

    def test_unknown_domain_returns_none(self):
        assert discover_provider("user@randomdomain.org") is None

    def test_empty_string_returns_none(self):
        assert discover_provider("") is None

    def test_no_at_sign_returns_none(self):
        assert discover_provider("nope") is None

    def test_case_insensitive(self):
        assert discover_provider("User@Gmail.COM") is not None
        assert discover_provider("User@Gmail.COM").name == "Google"


@pytest.mark.unit
class TestListSupportedProviders:
    """list_supported_providers should return deduplicated, sorted names."""

    def test_returns_four_providers(self):
        names = list_supported_providers()
        assert len(names) == 4

    def test_sorted_alphabetically(self):
        names = list_supported_providers()
        assert names == sorted(names)

    def test_contains_all_providers(self):
        names = list_supported_providers()
        assert "Google" in names
        assert "Apple" in names
        assert "Yahoo" in names
        assert "Outlook" in names


@pytest.mark.unit
class TestUnifiedProviderProtocols:
    """Verify protocol availability per provider matches the plan."""

    def test_google_full_stack(self):
        p = discover_provider("u@gmail.com")
        assert p.imap is not None
        assert p.smtp is not None
        assert p.caldav_url is not None
        assert p.carddav_url is not None

    def test_apple_full_stack(self):
        p = discover_provider("u@icloud.com")
        assert p.imap is not None
        assert p.smtp is not None
        assert p.caldav_url is not None
        assert p.carddav_url is not None

    def test_yahoo_no_carddav(self):
        p = discover_provider("u@yahoo.com")
        assert p.imap is not None
        assert p.smtp is not None
        assert p.caldav_url is not None
        assert p.carddav_url is None

    def test_outlook_imap_only(self):
        p = discover_provider("u@outlook.com")
        assert p.imap is not None
        assert p.smtp is not None
        assert p.caldav_url is None
        assert p.carddav_url is None
