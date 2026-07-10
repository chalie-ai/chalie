"""Inbound DTO for the voice-settings endpoint group.

Voice settings is a singleton toggle, not CRUD: the only accepted body is the
enable flag itself. ``extra="forbid"`` (inherited from :class:`Request`)
rejects any other key outright.
"""

from __future__ import annotations

from .request import Request


class VoiceSettingsRequest(Request):
    """Inbound body to toggle voice on/off."""

    enabled: bool
