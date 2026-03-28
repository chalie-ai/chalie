"""
Shared email provider autodiscovery.

Given an email address, resolves IMAP/SMTP connection settings from a
hard-coded registry of well-known providers.
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
class EmailProviderSettings:
    """Resolved IMAP + SMTP settings for an email provider."""

    provider_name: str
    imap: ServerSettings
    smtp: ServerSettings
    requires_app_password: bool
    notes: str = ""


# -- Provider definitions (shared across domain aliases) --------------------

_GMAIL = EmailProviderSettings(
    "Gmail",
    ServerSettings("imap.gmail.com", 993, True),
    ServerSettings("smtp.gmail.com", 465, True),
    True, "Enable 2FA then create an App Password at myaccount.google.com/apppasswords",
)
_OUTLOOK = EmailProviderSettings(
    "Outlook",
    ServerSettings("outlook.office365.com", 993, True),
    ServerSettings("smtp.office365.com", 587, False),
    True, "Enable 2FA then create an App Password in Microsoft account security",
)
_ICLOUD = EmailProviderSettings(
    "iCloud",
    ServerSettings("imap.mail.me.com", 993, True),
    ServerSettings("smtp.mail.me.com", 587, False),
    True, "Generate an app-specific password at appleid.apple.com",
)
_FASTMAIL = EmailProviderSettings(
    "Fastmail",
    ServerSettings("imap.fastmail.com", 993, True),
    ServerSettings("smtp.fastmail.com", 465, True),
    True, "Create an App Password at Settings > Privacy & Security",
)
_YAHOO = EmailProviderSettings(
    "Yahoo Mail",
    ServerSettings("imap.mail.yahoo.com", 993, True),
    ServerSettings("smtp.mail.yahoo.com", 465, True),
    True, "Generate an App Password at login.yahoo.com > Account Security",
)
_PROTON = EmailProviderSettings(
    "ProtonMail",
    ServerSettings("127.0.0.1", 1143, False),
    ServerSettings("127.0.0.1", 1025, False),
    False, "Requires Proton Mail Bridge running locally",
)
_ZOHO = EmailProviderSettings(
    "Zoho Mail",
    ServerSettings("imap.zoho.com", 993, True),
    ServerSettings("smtp.zoho.com", 465, True),
    True,
)
_AOL = EmailProviderSettings(
    "AOL Mail",
    ServerSettings("imap.aol.com", 993, True),
    ServerSettings("smtp.aol.com", 465, True),
    True,
)
_GMX = EmailProviderSettings(
    "GMX Mail",
    ServerSettings("imap.gmx.com", 993, True),
    ServerSettings("mail.gmx.com", 587, False),
    False,
)

# -- Domain → settings mapping (aliases share the same object) -------------

PROVIDERS: dict[str, EmailProviderSettings] = {
    "gmail.com": _GMAIL, "googlemail.com": _GMAIL,
    "outlook.com": _OUTLOOK, "hotmail.com": _OUTLOOK, "live.com": _OUTLOOK,
    "icloud.com": _ICLOUD, "me.com": _ICLOUD, "mac.com": _ICLOUD,
    "fastmail.com": _FASTMAIL, "fastmail.fm": _FASTMAIL,
    "yahoo.com": _YAHOO, "yahoo.co.uk": _YAHOO,
    "proton.me": _PROTON, "protonmail.com": _PROTON,
    "zoho.com": _ZOHO,
    "aol.com": _AOL,
    "gmx.com": _GMX, "gmx.net": _GMX,
}


def discover_email_settings(email: str) -> EmailProviderSettings | None:
    """Resolve IMAP/SMTP settings from an email address.

    Returns ``None`` for unknown domains.
    """
    if not email or "@" not in email:
        return None
    domain = email.rsplit("@", 1)[-1].strip().lower()
    if not domain:
        return None
    return PROVIDERS.get(domain)


def list_supported_providers() -> list[str]:
    """Return a sorted, deduplicated list of supported provider names."""
    return sorted({s.provider_name for s in PROVIDERS.values()})
