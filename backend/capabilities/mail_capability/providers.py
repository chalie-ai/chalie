"""
Unified provider registry for the mail capability.

Merges IMAP/SMTP, CalDAV, and CardDAV endpoint information into a single
lookup keyed by email domain.  Four providers are supported:

- **Google** (gmail.com, googlemail.com) — IMAP, CalDAV, CardDAV
- **Apple** (icloud.com, me.com, mac.com) — IMAP, CalDAV, CardDAV
- **Yahoo** (yahoo.com, yahoo.co.uk) — IMAP, CalDAV only
- **Outlook** (outlook.com, hotmail.com, live.com) — IMAP only

Usage::

    from capabilities.mail_capability.providers import discover_provider

    provider = discover_provider("user@gmail.com")
    if provider and provider.imap:
        # connect IMAP ...
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerSettings:
    """Connection settings for a single mail server (IMAP or SMTP)."""

    host: str
    port: int
    tls: bool


@dataclass(frozen=True, slots=True)
class UnifiedProvider:
    """All protocol endpoints for a single email provider."""

    name: str

    # IMAP / SMTP (None only if provider has no email, e.g. hypothetical
    # calendar-only providers — all four current providers have IMAP).
    imap: ServerSettings | None
    smtp: ServerSettings | None

    # CalDAV calendar — None for providers without calendar support.
    caldav_url: str | None
    caldav_principal_template: str | None

    # CardDAV contacts — None for providers without contacts support.
    carddav_url: str | None

    requires_app_password: bool

    # Human-readable setup instructions shown during onboarding.
    notes: str = ""


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

_GOOGLE = UnifiedProvider(
    name="Google",
    imap=ServerSettings("imap.gmail.com", 993, True),
    smtp=ServerSettings("smtp.gmail.com", 465, True),
    caldav_url="https://www.google.com/calendar/dav/{username}/user",
    caldav_principal_template="/calendar/dav/{username}/user",
    carddav_url="https://www.googleapis.com/carddav/v1/principals/{username}/lists/default/",
    requires_app_password=True,
    notes="Enable 2FA then create an App Password at myaccount.google.com/apppasswords",
)

_APPLE = UnifiedProvider(
    name="Apple",
    imap=ServerSettings("imap.mail.me.com", 993, True),
    smtp=ServerSettings("smtp.mail.me.com", 587, False),
    caldav_url="https://caldav.icloud.com/",
    caldav_principal_template="/{username}/principal/",
    carddav_url="https://contacts.icloud.com/",
    requires_app_password=True,
    notes="Generate an app-specific password at appleid.apple.com",
)

_YAHOO = UnifiedProvider(
    name="Yahoo",
    imap=ServerSettings("imap.mail.yahoo.com", 993, True),
    smtp=ServerSettings("smtp.mail.yahoo.com", 465, True),
    caldav_url="https://caldav.calendar.yahoo.com/",
    caldav_principal_template=None,
    carddav_url=None,  # Yahoo deprecated CardDAV
    requires_app_password=True,
    notes="Generate an App Password at login.yahoo.com > Account Security",
)

_OUTLOOK = UnifiedProvider(
    name="Outlook",
    imap=ServerSettings("outlook.office365.com", 993, True),
    smtp=ServerSettings("smtp.office365.com", 587, False),
    caldav_url=None,  # Microsoft requires Graph API + OAuth
    caldav_principal_template=None,
    carddav_url=None,  # No CardDAV support
    requires_app_password=True,
    notes="Enable 2FA then create an App Password in Microsoft account security",
)


# ---------------------------------------------------------------------------
# Domain → provider mapping
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, UnifiedProvider] = {
    "gmail.com": _GOOGLE,
    "googlemail.com": _GOOGLE,
    "icloud.com": _APPLE,
    "me.com": _APPLE,
    "mac.com": _APPLE,
    "yahoo.com": _YAHOO,
    "yahoo.co.uk": _YAHOO,
    "outlook.com": _OUTLOOK,
    "hotmail.com": _OUTLOOK,
    "live.com": _OUTLOOK,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_provider(email: str) -> UnifiedProvider | None:
    """Resolve all protocol endpoints from an email address.

    Returns ``None`` for unsupported domains.
    """
    if not email or "@" not in email:
        return None
    domain = email.rsplit("@", 1)[-1].strip().lower()
    if not domain:
        return None
    return PROVIDERS.get(domain)


def list_supported_providers() -> list[str]:
    """Return a sorted, deduplicated list of supported provider names."""
    return sorted({p.name for p in PROVIDERS.values()})


def build_custom_provider(
    *,
    imap_host: str | None = None,
    imap_port: int = 993,
    imap_tls: bool = True,
    smtp_host: str | None = None,
    smtp_port: int = 587,
    smtp_tls: bool = False,
    caldav_url: str | None = None,
    carddav_url: str | None = None,
) -> UnifiedProvider:
    """Build a provider from explicit connection fields.

    Used for custom or self-hosted mail servers (e.g. GreenMail + Radicale
    in test environments) that are not in the ``PROVIDERS`` registry.
    SMTP settings are optional; omit to disable outbound sending.
    """
    return UnifiedProvider(
        name="Custom",
        imap=ServerSettings(imap_host, imap_port, imap_tls) if imap_host else None,
        smtp=ServerSettings(smtp_host, smtp_port, smtp_tls) if smtp_host else None,
        caldav_url=caldav_url,
        caldav_principal_template=None,
        carddav_url=carddav_url,
        requires_app_password=False,
    )
