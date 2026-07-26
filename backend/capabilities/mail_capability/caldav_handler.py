

from __future__ import annotations

import datetime as _dt_module
import logging
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, cast

from services.time_utils import utc_now, parse_utc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType
    from typing import Protocol

    class _ICalProp(Protocol):
        dt: object
        def to_ical(self) -> bytes: ...

    class _ICalComponent(Protocol):
        name: str
        def get(self, key: str) -> object: ...
        def pop(self, key: str, default: object = None) -> object: ...
        def add(self, name: str, value: object) -> None: ...

    class _ICalEvent(Protocol):
        icalendar_instance: object
        icalendar_component: object
        data: str | bytes
        def save(self) -> None: ...

    class _ICalObj(Protocol):
        """A constructed Calendar or Event object from the icalendar library."""
        def add(self, name: str, value: object) -> None: ...
        def add_component(self, component: "_ICalObj") -> None: ...
        def to_ical(self) -> bytes: ...
        def walk(self) -> "Iterable[_ICalComponent]": ...

    class _ICalCalendar(Protocol):
        def walk(self) -> "Iterable[_ICalComponent]": ...
        def add_component(self, component: object) -> None: ...
        def to_ical(self) -> bytes: ...

    class _ICalFactory(Protocol):
        """A Calendar or Event class from icalendar — callable as a constructor, with from_ical."""
        def __call__(self) -> "_ICalObj": ...
        def from_ical(self, ical_string: str) -> "_ICalObj": ...

    class _IVRecur(Protocol):
        """The icalendar ``vRecur`` value type — parses an RFC-5545 RRULE string."""
        def from_ical(self, ical_string: str) -> object: ...

    class _ICalLib(Protocol):
        """Structural type for the icalendar module."""
        Calendar: "_ICalFactory"
        Event: "_ICalFactory"
        vRecur: "_IVRecur"

    class _ICalEventMutable(Protocol):
        icalendar_instance: object
        icalendar_component: object
        data: str | bytes
        def save(self) -> None: ...
        def delete(self) -> None: ...

    class _CalendarObj(Protocol):
        name: str
        def date_search(self, start: object, end: object, expand: bool) -> "Iterable[object]": ...
        def search(self, uid: str) -> "list[_ICalEventMutable]": ...
        def save_event(self, ical_str: str) -> None: ...

    class _Principal(Protocol):
        def calendars(self) -> list["_CalendarObj"]: ...

    class _CalDAVClient(Protocol):
        def principal(self) -> "_Principal": ...

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional library imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    import caldav as _caldav_lib
    _CALDAV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _caldav_lib = cast("ModuleType", None)
    _CALDAV_AVAILABLE = False

try:
    import icalendar as _icalendar_lib
    _ICALENDAR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _icalendar_lib = cast("ModuleType", None)
    _ICALENDAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 10
_DEFAULT_PAST_DAYS = 30
_DEFAULT_FUTURE_DAYS = 30

# list_events window/limit defaults when the caller omits them.
_LIST_DEFAULT_FUTURE_DAYS = 7
_LIST_DEFAULT_LIMIT = 50
_LIST_MAX_LIMIT = 200

# Title→uid resolution scans a live window (no persisted mirror to query) so it
# reaches both recently-past and comfortably-future events by their summary.
_TITLE_MATCH_PAST_DAYS = 30
_TITLE_MATCH_FUTURE_DAYS = 365

_ERR_CALDAV_NOT_INSTALLED = "'caldav' package is not installed."
_ERR_ICAL_NOT_INSTALLED = "'icalendar' package is not installed."
# find_free_slots / get_attendees computed over the removed scheduled_items
# mirror and have no live-CalDAV replacement in scope — they return this honest,
# deliberate error rather than a stale free/busy result.
_ERR_OP_UNSUPPORTED = "This calendar operation is not currently supported."

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_date_utc(d: object) -> _dt_module.datetime:
    if isinstance(d, _dt_module.datetime):
        if d.tzinfo is not None:
            return d.astimezone(_dt_module.timezone.utc)
        return parse_utc(d)
    if isinstance(d, _dt_module.date):
        return parse_utc(_dt_module.datetime(d.year, d.month, d.day, 0, 0, 0))
    return parse_utc(cast("_dt_module.datetime | str", d))


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------


class CaldavHandler:

    # ------------------------------------------------------------------
    # Server connection
    # ------------------------------------------------------------------

    def open_client(self, url: str, username: str, password: str) -> "_CalDAVClient":
        if not _CALDAV_AVAILABLE:
            raise RuntimeError(_ERR_CALDAV_NOT_INSTALLED)
        return cast("_CalDAVClient", cast("Callable[..., object]", _caldav_lib.DAVClient)(
            url=url, username=username, password=password, timeout=_CONNECT_TIMEOUT
        ))

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        client: "_CalDAVClient",
        past_days: int = _DEFAULT_PAST_DAYS,
        future_days: int = _DEFAULT_FUTURE_DAYS,
    ) -> list[dict[str, object]]:
        if not _CALDAV_AVAILABLE:
            logger.error("[caldav_handler] ingest: 'caldav' package unavailable.")
            return []
        if not _ICALENDAR_AVAILABLE:
            logger.error("[caldav_handler] ingest: 'icalendar' package unavailable.")
            return []

        now = utc_now()
        range_start = now - timedelta(days=past_days)
        range_end = now + timedelta(days=future_days)

        try:
            principal = client.principal()
            calendars = principal.calendars()
        except Exception as exc:
            logger.exception("[caldav_handler] ingest: failed to list calendars: %s", exc)
            return []

        all_events: list[dict[str, object]] = []
        for calendar in calendars:
            cal_name = "Unknown"
            try:
                cal_name = getattr(calendar, "name", None) or "Unknown"
                for raw_event in calendar.date_search(
                    start=range_start, end=range_end, expand=True
                ):
                    try:
                        all_events.extend(self.parse_event(raw_event, cal_name))
                    except Exception as exc:
                        logger.warning(
                            "[caldav_handler] parse failed in '%s': %s", cal_name, exc
                        )
            except Exception as exc:
                logger.exception(
                    "[caldav_handler] fetch failed for '%s': %s", cal_name, exc
                )

        try:
            from capabilities.contact_resolver import index_person
            for event in all_events:
                for attendee in cast("list[str]", event.get("attendees", [])):
                    index_person(attendee, source="caldav")
        except Exception as exc:
            logger.debug("[caldav_handler] contact indexing skipped: %s", exc)

        logger.info("[caldav_handler] ingest complete — %d events", len(all_events))
        return all_events

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------

    def parse_event(self, raw_event: object, calendar_name: str) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []

        try:
            ical_instance = cast("_ICalEvent", raw_event).icalendar_instance
        except AttributeError:
            try:
                component = cast("_ICalEvent", raw_event).icalendar_component
                ical_instance = cast("_ICalLib", _icalendar_lib).Calendar()
                ical_instance.add_component(cast("_ICalObj", component))
            except Exception as exc:
                logger.warning("[caldav_handler] ical instance unavailable: %s", exc)
                return results

        for component in cast("_ICalObj", ical_instance).walk():
            if component.name != "VEVENT":
                continue
            dtstart_prop = component.get("DTSTART")
            if dtstart_prop is None:
                continue

            dt_raw = cast("_ICalProp", dtstart_prop).dt
            all_day: bool = (
                isinstance(dt_raw, _dt_module.date)
                and not isinstance(dt_raw, _dt_module.datetime)
            )
            dtstart = _make_date_utc(dt_raw)

            dtend_prop = component.get("DTEND")
            if dtend_prop is not None:
                dtend = _make_date_utc(cast("_ICalProp", dtend_prop).dt)
            else:
                dur_prop = component.get("DURATION")
                if dur_prop is not None:
                    try:
                        dtend = dtstart + cast(_dt_module.timedelta, cast("_ICalProp", dur_prop).dt)
                    except Exception:
                        dtend = dtstart
                else:
                    dtend = dtstart

            uid_prop = component.get("UID")
            uid: str = str(uid_prop) if uid_prop is not None else ""

            summary_prop = component.get("SUMMARY")
            summary: str = str(summary_prop) if summary_prop is not None else "No title"

            location_prop = component.get("LOCATION")
            location: str | None = str(location_prop) if location_prop is not None else None

            attendees_raw = component.get("ATTENDEE")
            if attendees_raw is None:
                attendees: list[str] = []
            elif isinstance(attendees_raw, list):
                attendees = [str(a).removeprefix("mailto:") for a in attendees_raw]
            else:
                attendees = [str(attendees_raw).removeprefix("mailto:")]

            rrule_prop = component.get("RRULE")
            recurrence: str | None = None
            if rrule_prop is not None:
                try:
                    recurrence = cast("_ICalProp", rrule_prop).to_ical().decode("utf-8")
                except Exception:
                    pass

            results.append({
                "uid": uid,
                "summary": summary,
                "dtstart": dtstart,
                "dtend": dtend,
                "location": location,
                "attendees": attendees,
                "recurrence": recurrence,
                "all_day": all_day,
                "calendar_name": calendar_name,
            })

        return results

    # ------------------------------------------------------------------
    # Tool handlers — server mutations (client passed in)
    # ------------------------------------------------------------------

    def create_event(self, client: "_CalDAVClient", params: dict[str, object]) -> dict[str, object]:
        if not _CALDAV_AVAILABLE:
            return {"error": _ERR_CALDAV_NOT_INSTALLED}
        if not _ICALENDAR_AVAILABLE:
            return {"error": _ERR_ICAL_NOT_INSTALLED}

        summary = (cast(str, params.get("summary")) or "").strip()
        dtstart_raw = (cast(str, params.get("dtstart")) or "").strip()
        dtend_raw = (cast(str, params.get("dtend")) or "").strip()

        if not summary:
            return {"error": "Parameter 'summary' is required."}
        if not dtstart_raw:
            return {"error": "Parameter 'dtstart' is required (ISO 8601 UTC)."}
        if not dtend_raw:
            return {"error": "Parameter 'dtend' is required (ISO 8601 UTC)."}

        try:
            dtstart = parse_utc(dtstart_raw)
            dtend = parse_utc(dtend_raw)
            location = cast(str, params.get("location")) or ""
            description = cast(str, params.get("description")) or ""
            cal_pref = cast(str, params.get("calendar_name")) or ""
            recurrence = (cast(str, params.get("recurrence")) or "").strip()

            principal = client.principal()
            calendars = principal.calendars()
            if not calendars:
                return {"error": "No calendars found on the CalDAV server."}

            target_cal = next(
                (c for c in calendars if getattr(c, "name", "") == cal_pref),
                calendars[0],
            )

            event_uid = str(uuid.uuid4())
            ical = cast("_ICalLib", _icalendar_lib).Calendar()
            ical.add("prodid", "-//CaldavHandler//EN")
            ical.add("version", "2.0")

            vevent = cast("_ICalLib", _icalendar_lib).Event()
            vevent.add("uid", event_uid)
            vevent.add("summary", summary)
            vevent.add("dtstart", dtstart)
            vevent.add("dtend", dtend)
            vevent.add("dtstamp", utc_now())
            if location:
                vevent.add("location", location)
            if description:
                vevent.add("description", description)
            if recurrence:
                try:
                    vevent.add(
                        "rrule",
                        cast("_ICalLib", _icalendar_lib).vRecur.from_ical(recurrence),
                    )
                except Exception as exc:
                    return {"error": f"Invalid recurrence rule {recurrence!r}: {exc}"}

            ical.add_component(vevent)
            target_cal.save_event(ical.to_ical().decode("utf-8"))

            cal_label = getattr(target_cal, "name", None) or "Unknown"
            logger.info(
                "[caldav_handler] Created event uid=%s summary=%r calendar=%r",
                event_uid, summary, cal_label,
            )
            return {
                "uid": event_uid,
                "summary": summary,
                "dtstart": dtstart.isoformat(),
                "dtend": dtend.isoformat(),
                "recurrence": recurrence or None,
                "calendar_name": cal_label,
            }
        except Exception as exc:
            logger.exception("[caldav_handler] create_event failed: %s", exc)
            return {"error": str(exc)}

    @staticmethod
    def _find_event_by_uid(client: "_CalDAVClient", uid: str) -> "_ICalEventMutable | None":
        principal = client.principal()
        for calendar in principal.calendars():
            try:
                results = calendar.search(uid=uid)
                if results:
                    return results[0]
            except Exception:
                continue
        return None

    @staticmethod
    def _parse_ical_from_event(found_event: "_ICalEventMutable") -> "_ICalObj":
        try:
            ical_data = (
                found_event.data if isinstance(found_event.data, str)
                else found_event.data.decode("utf-8")
            )
            return cast("_ICalLib", _icalendar_lib).Calendar.from_ical(ical_data)
        except AttributeError:
            return cast("_ICalObj", found_event.icalendar_instance)

    @staticmethod
    def _apply_event_updates(ical: "_ICalObj", params: dict[str, object]) -> None:
        for component in ical.walk():
            if component.name != "VEVENT":
                continue
            if "summary" in params and params["summary"] is not None:
                component.pop("SUMMARY", None)
                component.add("summary", str(params["summary"]))
            if "dtstart" in params and params["dtstart"] is not None:
                component.pop("DTSTART", None)
                component.add("dtstart", parse_utc(cast("str", params["dtstart"])))
            if "dtend" in params and params["dtend"] is not None:
                component.pop("DTEND", None)
                component.add("dtend", parse_utc(cast("str", params["dtend"])))
            if "location" in params:
                component.pop("LOCATION", None)
                if params["location"]:
                    component.add("location", str(params["location"]))
            if "description" in params:
                component.pop("DESCRIPTION", None)
                if params["description"]:
                    component.add("description", str(params["description"]))
            if "recurrence" in params:
                component.pop("RRULE", None)
                if params["recurrence"]:
                    component.add(
                        "rrule",
                        cast("_ICalLib", _icalendar_lib).vRecur.from_ical(
                            str(params["recurrence"])
                        ),
                    )
            break

    def update_event(self, client: "_CalDAVClient", params: dict[str, object]) -> dict[str, object]:
        if not _CALDAV_AVAILABLE:
            return {"error": _ERR_CALDAV_NOT_INSTALLED}
        if not _ICALENDAR_AVAILABLE:
            return {"error": _ERR_ICAL_NOT_INSTALLED}

        uid, err = self._resolve_write_uid(client, params)
        if err is not None:
            return err

        try:
            found_event = self._find_event_by_uid(client, uid)
            if found_event is None:
                return {"error": f"Event not found (UID: {uid})"}

            ical = self._parse_ical_from_event(found_event)
            self._apply_event_updates(ical, params)

            found_event.data = ical.to_ical().decode("utf-8")
            found_event.save()
            logger.info("[caldav_handler] Updated event uid=%s", uid)
            return {"uid": uid, "updated": True}
        except Exception as exc:
            logger.exception("[caldav_handler] update_event failed: %s", exc)
            return {"error": str(exc)}

    def delete_event(self, client: "_CalDAVClient", params: dict[str, object]) -> dict[str, object]:
        if not _CALDAV_AVAILABLE:
            return {"error": _ERR_CALDAV_NOT_INSTALLED}

        uid, err = self._resolve_write_uid(client, params)
        if err is not None:
            return err

        try:
            principal = client.principal()
            for calendar in principal.calendars():
                try:
                    results = calendar.search(uid=uid)
                    if results:
                        results[0].delete()
                        logger.info("[caldav_handler] Deleted event uid=%s", uid)
                        return {"uid": uid, "deleted": True}
                except Exception:
                    continue
            return {"error": f"Event not found (UID: {uid})"}
        except Exception as exc:
            logger.exception("[caldav_handler] delete_event failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tool handlers — live reads (query the CalDAV server directly)
    # ------------------------------------------------------------------

    def list_events(self, client: "_CalDAVClient", params: dict[str, object]) -> dict[str, object]:
        """List events in a window (default: now .. +7d), soonest first, capped by
        ``limit`` (default 50, max 200). Datetimes are serialized to ISO strings."""
        if not _CALDAV_AVAILABLE:
            return {"error": _ERR_CALDAV_NOT_INSTALLED}
        if not _ICALENDAR_AVAILABLE:
            return {"error": _ERR_ICAL_NOT_INSTALLED}
        try:
            start, end = self._resolve_window(params)
            events = self._collect_events(client, start, end)
            events.sort(key=lambda e: cast(_dt_module.datetime, e["dtstart"]))
            events = events[: self._resolve_limit(params)]
            return {
                "events": [self._serialize_event(e) for e in events],
                "count": len(events),
                "window": {"start": start.isoformat(), "end": end.isoformat()},
            }
        except Exception as exc:
            logger.exception("[caldav_handler] list_events failed: %s", exc)
            return {"error": str(exc)}

    def get_event(self, client: "_CalDAVClient", params: dict[str, object]) -> dict[str, object]:
        """Fetch one event by ``uid``, or by ``title`` (unique live substring match)."""
        if not _CALDAV_AVAILABLE:
            return {"error": _ERR_CALDAV_NOT_INSTALLED}
        if not _ICALENDAR_AVAILABLE:
            return {"error": _ERR_ICAL_NOT_INSTALLED}

        uid = (cast(str, params.get("uid")) or "").strip()
        title = (cast(str, params.get("title")) or "").strip()
        if not uid and not title:
            return {"error": "Parameter 'uid' or 'title' is required."}
        try:
            if uid:
                event = self._find_parsed_by_uid(client, uid)
                if event is None:
                    return {"error": f"Event not found (UID: {uid})"}
                return {"event": self._serialize_event(event)}

            matches = self._events_matching_title(client, title)
            if not matches:
                return {"error": f"No event found matching title {title!r}."}
            if len(matches) > 1:
                return {"error": self._ambiguous_title_error(title, matches)}
            return {"event": self._serialize_event(matches[0])}
        except Exception as exc:
            logger.exception("[caldav_handler] get_event failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_window(
        params: dict[str, object],
    ) -> tuple[_dt_module.datetime, _dt_module.datetime]:
        now = utc_now()
        raw_from = (cast(str, params.get("date_from")) or "").strip()
        raw_to = (cast(str, params.get("date_to")) or "").strip()
        start = parse_utc(raw_from) if raw_from else now
        if raw_to:
            end = parse_utc(raw_to)
            # A date-only upper bound should cover the WHOLE day, else events later
            # that day are missed. CalendarAbility resolves bare dates (e.g. '2026-07-19')
            # to LOCAL midnight in the user's timezone and converts to UTC — so for a
            # +02:00 user '2026-07-19' becomes 2026-07-18T22:00:00Z, which is NOT
            # midnight UTC but IS midnight locally. Detect the date-only signal by
            # checking both UTC midnight AND local-midnight (the latter being the
            # canonical "bare date" signal). An explicit local time (e.g. 08:30) must
            # stay as-is.
            _extended = False
            if end.hour == 0 and end.minute == 0 and end.second == 0:
                _extended = True
            else:
                # Try the local-timezone check; fall back silently on any failure so
                # locale lookups can never break window resolution.
                try:
                    from zoneinfo import ZoneInfo  # stdlib

                    from services.locale_service import get_timezone_name

                    tz = ZoneInfo(get_timezone_name())
                    local_end = end.astimezone(tz)
                    if (
                        local_end.hour == 0
                        and local_end.minute == 0
                        and local_end.second == 0
                    ):
                        _extended = True
                except Exception:
                    pass
            if _extended:
                end = end + timedelta(days=1)
        elif raw_from:
            end = start + timedelta(days=_LIST_DEFAULT_FUTURE_DAYS)
        else:
            end = now + timedelta(days=_LIST_DEFAULT_FUTURE_DAYS)
        # A lone past upper bound (start defaults to now) would invert the window and
        # be handed straight to date_search — swap so the query is always valid.
        if start > end:
            start, end = end, start
        return start, end

    @staticmethod
    def _resolve_limit(params: dict[str, object]) -> int:
        raw = params.get("limit")
        try:
            limit = int(cast("int | str", raw))
        except (TypeError, ValueError):
            limit = _LIST_DEFAULT_LIMIT
        return max(1, min(limit, _LIST_MAX_LIMIT))

    def _collect_events(
        self, client: "_CalDAVClient", start: _dt_module.datetime, end: _dt_module.datetime
    ) -> list[dict[str, object]]:
        principal = client.principal()
        out: list[dict[str, object]] = []
        for calendar in principal.calendars():
            cal_name = getattr(calendar, "name", None) or "Unknown"
            try:
                for raw_event in calendar.date_search(start=start, end=end, expand=True):
                    try:
                        out.extend(self.parse_event(raw_event, cal_name))
                    except Exception as exc:
                        logger.warning(
                            "[caldav_handler] parse failed in '%s': %s", cal_name, exc
                        )
            except Exception as exc:
                logger.exception("[caldav_handler] fetch failed for '%s': %s", cal_name, exc)
        return out

    def _find_parsed_by_uid(
        self, client: "_CalDAVClient", uid: str
    ) -> dict[str, object] | None:
        principal = client.principal()
        for calendar in principal.calendars():
            cal_name = getattr(calendar, "name", None) or "Unknown"
            try:
                results = calendar.search(uid=uid)
            except Exception:
                continue
            if results:
                parsed = self.parse_event(results[0], cal_name)
                if parsed:
                    return parsed[0]
        return None

    def _events_matching_title(
        self, client: "_CalDAVClient", title: str
    ) -> list[dict[str, object]]:
        """Client-side substring match over a live window — there is no persisted
        mirror to query. Recurrence occurrences sharing a uid collapse to a single
        representative (the soonest upcoming occurrence)."""
        now = utc_now()
        start = now - timedelta(days=_TITLE_MATCH_PAST_DAYS)
        end = now + timedelta(days=_TITLE_MATCH_FUTURE_DAYS)
        needle = title.lower()
        matched = [
            e
            for e in self._collect_events(client, start, end)
            if needle in str(e.get("summary", "")).lower()
        ]
        return self._representative_per_uid(matched, now)

    def _resolve_write_uid(
        self, client: "_CalDAVClient", params: dict[str, object]
    ) -> tuple[str, dict[str, object] | None]:
        """Return ``(uid, error)``. An explicit ``uid`` passes straight through;
        otherwise resolve a UNIQUE ``title`` substring match over the live window.
        Absent target or an ambiguous match returns an error dict (with
        candidates) and an empty uid."""
        uid = (cast(str, params.get("uid")) or "").strip()
        if uid:
            return uid, None
        title = (cast(str, params.get("title")) or "").strip()
        if not title:
            return "", {"error": "Parameter 'uid' or 'title' is required."}
        matches = self._events_matching_title(client, title)
        if not matches:
            return "", {"error": f"No event found matching title {title!r}."}
        if len(matches) > 1:
            return "", {"error": self._ambiguous_title_error(title, matches)}
        return str(matches[0].get("uid", "")), None

    @staticmethod
    def _representative_per_uid(
        events: list[dict[str, object]], now: _dt_module.datetime
    ) -> list[dict[str, object]]:
        """Collapse recurrence occurrences (one uid, many expanded VEVENTs) to ONE
        representative each — the soonest upcoming occurrence, or the most recent
        past one when the whole series is behind us — so a title match surfaces the
        occurrence a user means, not a stale month-old expansion. Uid-less events
        pass through untouched."""
        groups: dict[str, list[dict[str, object]]] = {}
        order: list[str] = []
        out: list[dict[str, object]] = []
        for event in events:
            uid = str(event.get("uid", ""))
            if not uid:
                out.append(event)
                continue
            if uid not in groups:
                groups[uid] = []
                order.append(uid)
            groups[uid].append(event)

        def _dtstart(event: dict[str, object]) -> _dt_module.datetime:
            return cast(_dt_module.datetime, event["dtstart"])

        for uid in order:
            occurrences = groups[uid]
            upcoming = [e for e in occurrences if _dtstart(e) >= now]
            best = min(upcoming, key=_dtstart) if upcoming else max(occurrences, key=_dtstart)
            out.append(best)
        return out

    @staticmethod
    def _ambiguous_title_error(title: str, matches: list[dict[str, object]]) -> str:
        """One self-contained error string naming each candidate's uid — the model
        disambiguates from this alone (``ToolResult.err`` carries only a message, no
        structured candidate list)."""
        listed = "; ".join(
            f"{str(e.get('summary', 'No title'))!r} (uid {e.get('uid', '')})"
            for e in matches[:10]
        )
        return (
            f"{len(matches)} events match title {title!r}; pass a uid to "
            f"disambiguate: {listed}."
        )

    @staticmethod
    def _serialize_event(event: dict[str, object]) -> dict[str, object]:
        out = dict(event)
        for key in ("dtstart", "dtend"):
            value = out.get(key)
            if isinstance(value, _dt_module.datetime):
                out[key] = value.isoformat()
        return out

    # ------------------------------------------------------------------
    # Tool handlers — free/busy + attendee resolution (out of scope)
    #
    # Both computed over the removed scheduled_items mirror and have no
    # live-CalDAV replacement in scope. They stay registered but return an
    # honest "unsupported" error rather than a stale result.
    # ------------------------------------------------------------------

    def find_free_slots(self, params: dict[str, object]) -> dict[str, object]:  # noqa: ARG002
        return {"error": _ERR_OP_UNSUPPORTED}

    def get_attendees(self, params: dict[str, object]) -> dict[str, object]:  # noqa: ARG002
        return {"error": _ERR_OP_UNSUPPORTED}
