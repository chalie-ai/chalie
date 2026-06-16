"""ClientContext — frozen snapshot of per-request client telemetry.

Captures the caller's location / locale / time at tool-dispatch time so
an Ability can localise its behaviour (weather lookups, date formatting,
currency conversion) without reaching into request-scoped globals itself.
Built from ``services.locale_service`` — the request-scoped contextvars
the client heartbeat populates. The tool dispatcher stores
``ClientContext.current().as_dict()`` onto ``ability.telemetry`` immediately
before ``run()``; abilities read the flat dict, so the value object is an
internal detail of how that dict is assembled.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientContext:
    """Immutable snapshot of the caller's location / locale / clock."""

    lat: float | None
    lon: float | None
    location_name: str
    city: str
    country: str
    time: str
    timezone: str
    locale: str
    language: str
    currency: str

    @classmethod
    def current(cls) -> "ClientContext | None":
        """Returns None when no context is stored yet (fresh boot, no
        heartbeat) or the locale services are unavailable."""
        try:
            from services.locale_service import (  # noqa: PLC0415
                format_date,
                get_currency,
                get_language,
                get_locale,
                get_location,
                get_timezone_name,
            )
            from services.time_utils import utc_now  # noqa: PLC0415

            location = get_location()
            loc_name = location.get("name") or ""
            city, country = "", ""
            if "," in loc_name:
                city, country = [p.strip() for p in loc_name.split(",", 1)]

            return cls(
                lat=location.get("lat"),
                lon=location.get("lon"),
                location_name=loc_name,
                city=city,
                country=country,
                time=format_date(utc_now(), "%H:%M:%S", for_ui=True) or "",
                timezone=get_timezone_name(),
                locale=get_locale(),
                language=get_language(),
                currency=get_currency(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ClientContext] snapshot failed: %s", exc)
            return None

    def as_dict(self) -> dict:
        """Flatten to the dict shape abilities read off ``self.telemetry``."""
        return asdict(self)
