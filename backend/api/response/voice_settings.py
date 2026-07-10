"""Response DTOs for the voice-settings endpoint group.

Voice settings is a singleton, not CRUD: ``VoiceSettingsResponse`` is the read
shape (stored enable flag plus live install status), while
``VoiceSettingsStateResponse`` is the post-toggle shape — the uniform superset
of both toggle branches (enable carries the install message; disable nulls
it).
"""

from __future__ import annotations

from .response import Response


class VoiceSettingsResponse(Response):
    """Read shape: stored enable flag plus live voice dependency status."""

    enabled: bool
    install_status: str
    error: str | None = None


class VoiceSettingsStateResponse(Response):
    """Post-toggle state; the uniform superset of the enable/disable branches."""

    enabled: bool
    status: str
    message: str | None = None
