"""
Shared email provider autodiscovery — backward-compatibility shim.

Delegates to :mod:`capabilities.mail_capability.providers` which is now
the single source of truth.  This module preserves the public API so
existing callers (tests, etc.) continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.email_utils import Email
from capabilities.mail_capability.providers import (
    PROVIDERS as _UNIFIED_PROVIDERS,
    ServerSettings,
)


@dataclass(frozen=True, slots=True)
class EmailProviderSettings:
    """Resolved IMAP + SMTP settings for an email provider."""

    provider_name: str
    imap: ServerSettings
    smtp: ServerSettings
    requires_app_password: bool
    notes: str = ""


# -- Build legacy PROVIDERS dict from unified registry -----------------------

PROVIDERS: dict[str, EmailProviderSettings] = {}
for _domain, _up in _UNIFIED_PROVIDERS.items():
    if _up.imap and _up.smtp:
        PROVIDERS[_domain] = EmailProviderSettings(
            provider_name=_up.name,
            imap=_up.imap,
            smtp=_up.smtp,
            requires_app_password=_up.requires_app_password,
            notes=_up.notes,
        )


def discover_email_settings(email: str) -> EmailProviderSettings | None:
    """Resolve IMAP/SMTP settings from an email address.

    Returns ``None`` for unknown domains.
    """
    if not email or "@" not in email:
        return None
    domain = Email.get_domain(email)
    if not domain:
        return None
    return PROVIDERS.get(domain)


def list_supported_providers() -> list[str]:
    """Return a sorted, deduplicated list of supported provider names."""
    return sorted({s.provider_name for s in PROVIDERS.values()})


# -- Email domain → CalDAV provider mapping ----------------------------------

_CALDAV_PROVIDERS: dict[str, str] = {}
for _domain, _up in _UNIFIED_PROVIDERS.items():
    if _up.caldav_url:
        _slug = _up.name.lower()
        _CALDAV_PROVIDERS[_domain] = _slug


def email_to_caldav_provider(email: str) -> str | None:
    """Map an email address to a CalDAV provider name.

    Returns ``None`` for domains without CalDAV support.
    """
    if not email or "@" not in email:
        return None
    domain = Email.get_domain(email)
    return _CALDAV_PROVIDERS.get(domain)
