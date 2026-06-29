"""DTOs for the voice-settings resource — the read/toggle HTTP contract.

Voice settings is not CRUD: a GET reads derived state (a stored boolean plus
live install status), a PUT toggles ``voice_enabled`` and triggers a background
install. ``VoiceSettingsState`` is the uniform superset of both toggle branches
(enable carries the install message; disable nulls it).
"""

from __future__ import annotations

from .base import DTO


class VoiceSettings(DTO):
    """Read shape: stored enable flag plus live voice dependency status."""

    enabled: bool
    install_status: str
    error: str | None = None


class VoiceSettingsUpdate(DTO):
    """Inbound body to toggle voice; ``enabled`` is required (422 when absent)."""

    enabled: bool


class VoiceSettingsState(DTO):
    """Post-toggle state; the uniform superset of the enable/disable branches."""

    enabled: bool
    status: str
    message: str | None = None