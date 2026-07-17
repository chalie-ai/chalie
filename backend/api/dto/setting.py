"""DTOs for ``GET/PUT /api/system/settings/<key>`` — opaque key/value settings.

The stored value is opaque (string, JSON, or null), so ``value`` stays a raw
``object`` passthrough. ``SettingWrite`` is the PUT body; an empty/falsey value
means delete the key (the handler decides), so it is NOT rejected by validation.
"""

from __future__ import annotations

from .base import DTO


class Setting(DTO):
    """One stored setting — key plus its opaque value."""

    key: str
    value: object


class SettingWrite(DTO):
    """Inbound body for a setting write — empty/falsey value means delete."""

    value: object = None