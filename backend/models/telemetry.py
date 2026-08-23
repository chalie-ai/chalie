"""Snapshot of the last client heartbeat, persisted as a JSON file by
``TelemetryService``. A plain transport-only value object — no Model
subclass, no SQL: it wraps the raw nested dict the frontend heartbeat
sends and projects its fields (timezone, locale, location, …) read-only
for the locale, world-state, and client-context consumers.
"""

from __future__ import annotations

from typing import cast


class Telemetry:
    """Read-only view over the raw nested heartbeat dict.

    Constructed with the snapshot as received (default: empty). Fields
    absent from the dict read as ``None``; nested groups read as
    ``dict[str, object] | None`` (a non-dict value degrades to ``None``,
    matching the old flattened-row read, where a non-dict group could
    never surface coordinates).
    """

    def __init__(self, ctx: dict[str, object] | None = None) -> None:
        self._ctx: dict[str, object] = dict(ctx) if ctx else {}

    # ── Read-only field projections ─────────────────────────────────────────

    @property
    def timezone(self) -> str | None:
        """IANA timezone name (e.g. ``Europe/Malta``), or None."""
        return self._str_field("timezone")

    @property
    def locale(self) -> str | None:
        """BCP-47 locale tag (e.g. ``en-MT``), or None."""
        return self._str_field("locale")

    @property
    def language(self) -> str | None:
        """Preferred language (e.g. ``en-GB``), or None."""
        return self._str_field("language")

    @property
    def currency(self) -> str | None:
        """ISO 4217 currency code (e.g. ``EUR``), or None."""
        return self._str_field("currency")

    @property
    def location_name(self) -> str | None:
        """Resolved locality label (e.g. ``Sliema, Malta``), or None."""
        return self._str_field("location_name")

    @property
    def location(self) -> dict[str, object] | None:
        """Raw GPS dict (``lat``/``lon``), or None when absent/non-dict."""
        return self._dict_field("location")

    @property
    def behavioral(self) -> dict[str, object] | None:
        """Behavioral signals group, or None when absent."""
        return self._dict_field("behavioral")

    @property
    def location_name_stale(self) -> bool:
        """True while a failed geocode retry is pending (default False)."""
        return bool(self._ctx.get("_location_name_stale", False))

    def as_dict(self) -> dict[str, object]:
        """The raw nested heartbeat dict this snapshot was built from."""
        return self._ctx

    # ── Typed field helpers ─────────────────────────────────────────────────

    def _str_field(self, key: str) -> str | None:
        value = self._ctx.get(key)
        return value if isinstance(value, str) else None

    def _dict_field(self, key: str) -> dict[str, object] | None:
        value = self._ctx.get(key)
        return cast("dict[str, object]", value) if isinstance(value, dict) else None
