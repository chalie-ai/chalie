"""
CaldavHandler — protocol-specific CalDAV logic for the mail capability.

Plain class (no AbstractCapability subclass). Credentials are passed per-call
for server-facing methods; DB-only methods query ``scheduled_items`` directly.
Signal emission uses ``capability_id='mail'`` throughout.
"""

from __future__ import annotations

import datetime as _dt_module
import json as _json
import logging
import uuid
from datetime import timedelta
from itertools import combinations

from services.time_utils import utc_now, parse_utc, get_user_tz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional library imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    import caldav as _caldav_lib  # type: ignore
    _CALDAV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _caldav_lib = None  # type: ignore
    _CALDAV_AVAILABLE = False

try:
    import icalendar as _icalendar_lib  # type: ignore
    _ICALENDAR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _icalendar_lib = None  # type: ignore
    _ICALENDAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 10
_BACK_TO_BACK_GAP = timedelta(minutes=5)
_DEFAULT_PAST_DAYS = 30
_DEFAULT_FUTURE_DAYS = 30

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_date_utc(d: object) -> _dt_module.datetime:
    """Convert date or datetime to UTC-aware datetime. date-only → midnight UTC."""
    if isinstance(d, _dt_module.datetime):
        if d.tzinfo is not None:
            return d.astimezone(_dt_module.timezone.utc)
        return parse_utc(d)
    if isinstance(d, _dt_module.date):
        return parse_utc(_dt_module.datetime(d.year, d.month, d.day, 0, 0, 0))
    return parse_utc(d)


def _events_overlap(a: dict, b: dict) -> bool:
    return max(a["dtstart"], b["dtstart"]) < min(a["dtend"], b["dtend"])


def _find_overlap_pairs(events: list, now: _dt_module.datetime) -> list:
    """Return ``(ev_a, ev_b, canon_key)`` tuples for upcoming overlapping events."""
    upcoming = [
        e for e in events
        if e.get("dtstart") and e.get("uid")
        and e["dtstart"] >= now
        and not e.get("all_day")
    ]
    return [
        (a, b, ":".join(sorted([a["uid"], b["uid"]])))
        for a, b in combinations(upcoming, 2)
        if a["uid"] != b["uid"] and _events_overlap(a, b)
    ]


def _find_back_to_back_pairs(events: list, now: _dt_module.datetime) -> list:
    """Return ``(ev_a, ev_b, gap_minutes, canon_key)`` for gaps < 5 minutes."""
    threshold = _BACK_TO_BACK_GAP.total_seconds() / 60
    upcoming = sorted(
        [e for e in events
         if e.get("dtstart") and e.get("dtend") and e.get("uid")
         and e["dtstart"] >= now and not e.get("all_day")],
        key=lambda e: e["dtstart"],
    )
    pairs = []
    for i in range(len(upcoming) - 1):
        a, b = upcoming[i], upcoming[i + 1]
        gap = (b["dtstart"] - a["dtend"]).total_seconds() / 60
        if a["uid"] != b["uid"] and 0 <= gap < threshold:
            pairs.append((a, b, round(gap), ":".join(sorted([a["uid"], b["uid"]]))))
    return pairs


def _get_user_tz():
    """Return user's ZoneInfo timezone or None.

    Delegates to the centralised ``get_user_tz()`` in time_utils.
    Returns None when the result is plain UTC (no user timezone detected).
    """
    tz = get_user_tz()
    if tz.key == "UTC":
        return None
    return tz


def _next_morning_8am() -> _dt_module.datetime:
    """Return the next 08:00 in the user's local timezone (UTC fallback)."""
    now = utc_now()
    tz = _get_user_tz()
    if tz:
        local_now = now.astimezone(tz)
        local_8am = local_now.replace(hour=8, minute=0, second=0, microsecond=0)
        if local_8am <= local_now:
            local_8am += timedelta(days=1)
        from datetime import timezone as _tz
        return local_8am.astimezone(_tz.utc)
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=8, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------


class CaldavHandler:
    """Protocol-specific CalDAV operations for the mail capability."""

    # ------------------------------------------------------------------
    # Server connection
    # ------------------------------------------------------------------

    def open_client(self, url: str, username: str, password: str):
        """Create and return an authenticated caldav.DAVClient."""
        if not _CALDAV_AVAILABLE:
            raise RuntimeError("'caldav' package is not installed.")
        return _caldav_lib.DAVClient(
            url=url, username=username, password=password, timeout=_CONNECT_TIMEOUT
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        client,
        past_days: int = _DEFAULT_PAST_DAYS,
        future_days: int = _DEFAULT_FUTURE_DAYS,
    ) -> list[dict]:
        """Fetch events from all calendars on *client* within the date range."""
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
            logger.error("[caldav_handler] ingest: failed to list calendars: %s", exc)
            return []

        all_events: list[dict] = []
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
                logger.error(
                    "[caldav_handler] fetch failed for '%s': %s", cal_name, exc
                )

        try:
            from capabilities.contact_resolver import index_person
            for event in all_events:
                for attendee in event.get("attendees", []):
                    index_person(attendee, source="caldav")
        except Exception as exc:
            logger.debug("[caldav_handler] contact indexing skipped: %s", exc)

        logger.info("[caldav_handler] ingest complete — %d events", len(all_events))
        return all_events

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------

    def parse_event(self, raw_event: object, calendar_name: str) -> list[dict]:
        """Parse a CalDAV resource into normalised event dicts (one per VEVENT)."""
        results: list[dict] = []

        try:
            ical_instance = raw_event.icalendar_instance
        except AttributeError:
            try:
                component = raw_event.icalendar_component
                ical_instance = _icalendar_lib.Calendar()
                ical_instance.add_component(component)
            except Exception as exc:
                logger.warning("[caldav_handler] ical instance unavailable: %s", exc)
                return results

        for component in ical_instance.walk():
            if component.name != "VEVENT":
                continue
            dtstart_prop = component.get("DTSTART")
            if dtstart_prop is None:
                continue

            dt_raw = dtstart_prop.dt
            all_day: bool = (
                isinstance(dt_raw, _dt_module.date)
                and not isinstance(dt_raw, _dt_module.datetime)
            )
            dtstart = _make_date_utc(dt_raw)

            dtend_prop = component.get("DTEND")
            if dtend_prop is not None:
                dtend = _make_date_utc(dtend_prop.dt)
            else:
                dur_prop = component.get("DURATION")
                if dur_prop is not None:
                    try:
                        dtend = dtstart + dur_prop.dt
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
                attendees: list = []
            elif isinstance(attendees_raw, list):
                attendees = [str(a).removeprefix("mailto:") for a in attendees_raw]
            else:
                attendees = [str(attendees_raw).removeprefix("mailto:")]

            rrule_prop = component.get("RRULE")
            recurrence: str | None = None
            if rrule_prop is not None:
                try:
                    recurrence = rrule_prop.to_ical().decode("utf-8")
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
    # Upsert (mark-sweep delta sync)
    # ------------------------------------------------------------------

    def upsert_events(self, events: list[dict], now: _dt_module.datetime) -> None:
        """Mark-sweep delta-sync events into scheduled_items (source='mail').

        Creates derivative items: 15-min alerts, conflict/b2b notifications,
        daily digest prompt, and a one-time greeting.
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                c = conn.cursor()

                # Mark existing mail events for stale check
                c.execute(
                    "UPDATE scheduled_items SET status='stale_check' "
                    "WHERE source='mail' AND item_type='event' AND status='pending'"
                )

                # Upsert each event
                for ev in events:
                    uid = ev.get("uid")
                    if not uid:
                        continue
                    external_uid = f"caldav:{uid}"
                    summary = ev.get("summary", "Event")
                    cal_name = ev.get("calendar_name", "")
                    location = ev.get("location") or ""
                    dtstart = ev.get("dtstart")
                    dtend = ev.get("dtend")

                    parts = [summary]
                    if cal_name:
                        parts.append(f"[{cal_name}]")
                    if location:
                        parts.append(f"@ {location}")
                    message = " ".join(parts)

                    metadata = _json.dumps({
                        "uid": uid,
                        "dtstart": dtstart.isoformat() if dtstart else None,
                        "dtend": dtend.isoformat() if dtend else None,
                        "location": location,
                        "attendees": ev.get("attendees", []),
                        "recurrence": ev.get("recurrence"),
                        "all_day": ev.get("all_day", False),
                        "calendar_name": cal_name,
                    })
                    due_at = dtstart.isoformat() if dtstart else now.isoformat()

                    c.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, status, channel,
                            source, external_uid, metadata, hidden, created_at)
                         VALUES (?, 'event', ?, ?, 'pending', 'calendar',
                                 'mail', ?, ?, 1, ?)
                         ON CONFLICT(external_uid) DO UPDATE SET
                            message=excluded.message,
                            due_at=excluded.due_at,
                            metadata=excluded.metadata,
                            hidden=1,
                            status='pending'""",
                        (uuid.uuid4().hex[:8], message, due_at,
                         external_uid, metadata, now.isoformat()),
                    )

                # Cancel stale events not seen in this sync
                c.execute(
                    "UPDATE scheduled_items SET status='cancelled' "
                    "WHERE source='mail' AND item_type='event' AND status='stale_check'"
                )

                # 15-min alerts for upcoming non-all-day events within 24h
                upcoming_cutoff = now + timedelta(hours=24)
                for ev in events:
                    dtstart = ev.get("dtstart")
                    uid = ev.get("uid")
                    if (not dtstart or not uid or dtstart < now
                            or dtstart > upcoming_cutoff or ev.get("all_day")):
                        continue
                    alert_msg = f"In 15 min: {ev.get('summary', 'Event')}"
                    if ev.get("location"):
                        alert_msg += f" @ {ev['location']}"
                    c.execute(
                        """INSERT OR IGNORE INTO scheduled_items
                           (id, item_type, message, due_at, status, channel,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'mail', ?, 1, ?)""",
                        (uuid.uuid4().hex[:8], alert_msg,
                         (dtstart - timedelta(minutes=15)).isoformat(),
                         f"caldav:{uid}:alert", now.isoformat()),
                    )

                # Conflict detection
                for ev_a, ev_b, canon_key in _find_overlap_pairs(events, now):
                    conflict_msg = (
                        f"Schedule conflict: \"{ev_a.get('summary', 'Event')}\" and "
                        f"\"{ev_b.get('summary', 'Event')}\" overlap"
                    )
                    c.execute(
                        """INSERT OR IGNORE INTO scheduled_items
                           (id, item_type, message, due_at, status, channel,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'mail', ?, 1, ?)""",
                        (uuid.uuid4().hex[:8], conflict_msg, now.isoformat(),
                         f"caldav:conflict:{canon_key}", now.isoformat()),
                    )
                # Back-to-back warnings (< 5 min gap)
                for ev_a, ev_b, gap_min, canon_key in _find_back_to_back_pairs(events, now):
                    b2b_msg = (
                        f"Tight transition ({gap_min}min gap): "
                        f"\"{ev_a.get('summary', 'Event')}\" \u2192 "
                        f"\"{ev_b.get('summary', 'Event')}\""
                    )
                    c.execute(
                        """INSERT OR IGNORE INTO scheduled_items
                           (id, item_type, message, due_at, status, channel,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'mail', ?, 1, ?)""",
                        (uuid.uuid4().hex[:8], b2b_msg,
                         ev_a.get("dtend", now).isoformat(),
                         f"caldav:b2b:{canon_key}", now.isoformat()),
                    )

                # Daily digest (one-time insert, recurring)
                if not c.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid='caldav:daily-digest'"
                ).fetchone():
                    c.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, recurrence, status, channel,
                            source, external_uid, hidden, created_at, is_prompt)
                         VALUES (?, 'prompt',
                                 'Summarize today''s calendar: highlight key meetings, conflicts, and free blocks. Keep it brief — 3-4 sentences.',
                                 ?, 'daily', 'pending', 'calendar',
                                 'mail', 'caldav:daily-digest', 1, ?, 1)""",
                        (uuid.uuid4().hex[:8], _next_morning_8am().isoformat(), now.isoformat()),
                    )

                # First-connect greeting (one-time)
                if not c.execute(
                    "SELECT id FROM scheduled_items WHERE external_uid='caldav:greeting'"
                ).fetchone():
                    n = len(events)
                    greeting_msg = (
                        f"Calendar connected! Found {n} event{'s' if n != 1 else ''} "
                        f"across your calendars."
                    )
                    c.execute(
                        """INSERT INTO scheduled_items
                           (id, item_type, message, due_at, status, channel,
                            source, external_uid, hidden, created_at)
                         VALUES (?, 'notification', ?, ?, 'pending', 'calendar',
                                 'mail', 'caldav:greeting', 1, ?)""",
                        (uuid.uuid4().hex[:8], greeting_msg,
                         now.isoformat(), now.isoformat()),
                    )

                conn.commit()
                logger.info(
                    "[caldav_handler] upserted %d events + derivative items", len(events)
                )
        except Exception as exc:
            logger.error("[caldav_handler] upsert_events failed: %s", exc)

    # ------------------------------------------------------------------
    # Tool handlers — DB queries (no credentials needed)
    # ------------------------------------------------------------------

    def list_events(self, params: dict) -> dict:
        """Query scheduled_items for calendar events.

        params: date_from, date_to, calendar_name, limit (max 200, default 50).
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            date_from = params.get("date_from")
            date_to = params.get("date_to")
            calendar_name = params.get("calendar_name")
            limit = min(int(params.get("limit", 50)), 200)

            query = (
                "SELECT message, due_at, metadata FROM scheduled_items "
                "WHERE source='mail' AND item_type='event' AND status='pending'"
            )
            qp: list = []
            if date_from:
                query += " AND due_at >= ?"
                qp.append(date_from)
            if date_to:
                query += " AND due_at <= ?"
                qp.append(date_to)
            query += " ORDER BY due_at ASC LIMIT ?"
            qp.append(limit)

            with db.connection() as conn:
                rows = conn.execute(query, qp).fetchall()

            results = []
            for msg, due_at, meta_raw in rows:
                meta = _json.loads(meta_raw) if meta_raw else {}
                if calendar_name and meta.get("calendar_name", "").lower() != calendar_name.lower():
                    continue
                results.append({
                    "summary": meta.get("uid", msg),
                    "title": msg,
                    "dtstart": due_at,
                    "dtend": meta.get("dtend"),
                    "location": meta.get("location"),
                    "attendees": meta.get("attendees", []),
                    "all_day": meta.get("all_day", False),
                    "calendar_name": meta.get("calendar_name"),
                    "uid": meta.get("uid"),
                })
            return {"events": results, "count": len(results)}
        except Exception as exc:
            return {"error": f"Failed to list events: {exc}"}

    def get_event(self, params: dict) -> dict:
        """Fetch a single event by UID from scheduled_items."""
        uid = params.get("uid") or params.get("event_uid")
        if not uid:
            return {"error": "uid is required"}
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                row = conn.execute(
                    "SELECT message, due_at, metadata FROM scheduled_items "
                    "WHERE external_uid = ? AND item_type='event'",
                    (f"caldav:{uid}",),
                ).fetchone()

            if row:
                msg, due_at, meta_raw = row
                meta = _json.loads(meta_raw) if meta_raw else {}
                return {"event": {
                    "uid": uid,
                    "title": msg,
                    "dtstart": due_at,
                    "dtend": meta.get("dtend"),
                    "location": meta.get("location"),
                    "attendees": meta.get("attendees", []),
                    "all_day": meta.get("all_day", False),
                    "calendar_name": meta.get("calendar_name"),
                    "recurrence": meta.get("recurrence"),
                }}
            return {"error": f"Event '{uid}' not found"}
        except Exception as exc:
            return {"error": f"Failed to get event: {exc}"}

    # ------------------------------------------------------------------
    # Tool handlers — server mutations (client passed in)
    # ------------------------------------------------------------------

    def create_event(self, client, params: dict) -> dict:
        """Create a new VEVENT on the CalDAV server.

        Required params: summary, dtstart (ISO 8601 UTC), dtend (ISO 8601 UTC).
        Optional: location, description, calendar_name.
        """
        if not _CALDAV_AVAILABLE:
            return {"error": "'caldav' package is not installed."}
        if not _ICALENDAR_AVAILABLE:
            return {"error": "'icalendar' package is not installed."}

        summary = (params.get("summary") or "").strip()
        dtstart_raw = (params.get("dtstart") or "").strip()
        dtend_raw = (params.get("dtend") or "").strip()

        if not summary:
            return {"error": "Parameter 'summary' is required."}
        if not dtstart_raw:
            return {"error": "Parameter 'dtstart' is required (ISO 8601 UTC)."}
        if not dtend_raw:
            return {"error": "Parameter 'dtend' is required (ISO 8601 UTC)."}

        try:
            dtstart = parse_utc(dtstart_raw)
            dtend = parse_utc(dtend_raw)
            location = params.get("location") or ""
            description = params.get("description") or ""
            cal_pref = params.get("calendar_name") or ""

            principal = client.principal()
            calendars = principal.calendars()
            if not calendars:
                return {"error": "No calendars found on the CalDAV server."}

            target_cal = next(
                (c for c in calendars if getattr(c, "name", "") == cal_pref),
                calendars[0],
            )

            event_uid = str(uuid.uuid4())
            ical = _icalendar_lib.Calendar()
            ical.add("prodid", "-//CaldavHandler//EN")
            ical.add("version", "2.0")

            vevent = _icalendar_lib.Event()
            vevent.add("uid", event_uid)
            vevent.add("summary", summary)
            vevent.add("dtstart", dtstart)
            vevent.add("dtend", dtend)
            vevent.add("dtstamp", utc_now())
            if location:
                vevent.add("location", location)
            if description:
                vevent.add("description", description)

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
                "calendar_name": cal_label,
            }
        except Exception as exc:
            logger.error("[caldav_handler] create_event failed: %s", exc)
            return {"error": str(exc)}

    def update_event(self, client, params: dict) -> dict:
        """Update an existing CalDAV event by UID.

        Required params: uid.
        Optional: summary, dtstart, dtend, location, description.
        """
        if not _CALDAV_AVAILABLE:
            return {"error": "'caldav' package is not installed."}
        if not _ICALENDAR_AVAILABLE:
            return {"error": "'icalendar' package is not installed."}

        uid = (params.get("uid") or "").strip()
        if not uid:
            return {"error": "Parameter 'uid' is required."}

        try:
            principal = client.principal()
            found_event = None
            for calendar in principal.calendars():
                try:
                    results = calendar.search(uid=uid)
                    if results:
                        found_event = results[0]
                        break
                except Exception:
                    continue

            if found_event is None:
                return {"error": f"Event not found (UID: {uid})"}

            try:
                ical_data = (
                    found_event.data if isinstance(found_event.data, str)
                    else found_event.data.decode("utf-8")
                )
                ical = _icalendar_lib.Calendar.from_ical(ical_data)
            except AttributeError:
                ical = found_event.icalendar_instance

            for component in ical.walk():
                if component.name != "VEVENT":
                    continue
                if "summary" in params and params["summary"] is not None:
                    component.pop("SUMMARY", None)
                    component.add("summary", str(params["summary"]))
                if "dtstart" in params and params["dtstart"] is not None:
                    component.pop("DTSTART", None)
                    component.add("dtstart", parse_utc(params["dtstart"]))
                if "dtend" in params and params["dtend"] is not None:
                    component.pop("DTEND", None)
                    component.add("dtend", parse_utc(params["dtend"]))
                if "location" in params:
                    component.pop("LOCATION", None)
                    if params["location"]:
                        component.add("location", str(params["location"]))
                if "description" in params:
                    component.pop("DESCRIPTION", None)
                    if params["description"]:
                        component.add("description", str(params["description"]))
                break

            found_event.data = ical.to_ical().decode("utf-8")
            found_event.save()
            logger.info("[caldav_handler] Updated event uid=%s", uid)
            return {"uid": uid, "updated": True}
        except Exception as exc:
            logger.error("[caldav_handler] update_event failed: %s", exc)
            return {"error": str(exc)}

    def delete_event(self, client, params: dict) -> dict:
        """Delete a calendar event from the CalDAV server by UID."""
        if not _CALDAV_AVAILABLE:
            return {"error": "'caldav' package is not installed."}

        uid = (params.get("uid") or "").strip()
        if not uid:
            return {"error": "Parameter 'uid' is required."}

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
            logger.error("[caldav_handler] delete_event failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tool handlers — DB queries continued
    # ------------------------------------------------------------------

    def find_free_slots(self, params: dict) -> dict:
        """Find free time slots within working hours by querying scheduled_items.

        params: date_from, date_to, min_duration_minutes (30), working_hours_start (8),
                working_hours_end (18).
        """
        try:
            from datetime import timezone as _tz
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            date_from = params.get("date_from", utc_now().isoformat())
            date_to = params.get("date_to", (utc_now() + timedelta(days=7)).isoformat())
            min_minutes = int(params.get("min_duration_minutes", 30))
            wh_start = int(params.get("working_hours_start", 8))
            wh_end = int(params.get("working_hours_end", 18))

            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT due_at, metadata FROM scheduled_items "
                    "WHERE source='mail' AND item_type='event' AND status='pending' "
                    "AND due_at >= ? AND due_at <= ? ORDER BY due_at ASC",
                    (date_from, date_to),
                ).fetchall()

            busy = []
            for due_at_str, meta_raw in rows:
                meta = _json.loads(meta_raw) if meta_raw else {}
                start = parse_utc(due_at_str)
                end_str = meta.get("dtend")
                end = parse_utc(end_str) if end_str else start + timedelta(hours=1)
                if not meta.get("all_day", False):
                    busy.append((start, end))
            busy.sort(key=lambda x: x[0])

            tz = _get_user_tz() or _tz.utc
            window_start = parse_utc(date_from)
            window_end = parse_utc(date_to)
            work_windows: list[tuple] = []
            day = window_start.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
            last_day = window_end.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
            while day <= last_day:
                ws = max(day.replace(hour=wh_start).astimezone(_tz.utc), window_start)
                we = min(day.replace(hour=wh_end).astimezone(_tz.utc), window_end)
                if ws < we:
                    work_windows.append((ws, we))
                day += timedelta(days=1)

            slots = []
            for ww_start, ww_end in work_windows:
                cursor_time = ww_start
                for bstart, bend in busy:
                    if bend <= ww_start:
                        continue
                    if bstart >= ww_end:
                        break
                    clamped_start = max(bstart, ww_start)
                    if clamped_start > cursor_time:
                        gap = (clamped_start - cursor_time).total_seconds() / 60
                        if gap >= min_minutes:
                            slots.append({
                                "start": cursor_time.isoformat(),
                                "end": clamped_start.isoformat(),
                                "duration_minutes": int(gap),
                            })
                    cursor_time = max(cursor_time, min(bend, ww_end))
                if cursor_time < ww_end:
                    gap = (ww_end - cursor_time).total_seconds() / 60
                    if gap >= min_minutes:
                        slots.append({
                            "start": cursor_time.isoformat(),
                            "end": ww_end.isoformat(),
                            "duration_minutes": int(gap),
                        })

            return {"free_slots": slots, "count": len(slots)}
        except Exception as exc:
            return {"error": f"Failed to find free slots: {exc}"}

    def get_attendees(self, params: dict) -> dict:
        """Return resolved attendees for a calendar event by UID."""
        uid = params.get("uid") or params.get("event_uid")
        if not uid:
            return {"error": "uid is required"}
        try:
            from services.database_service import get_shared_db_service
            from capabilities.contact_resolver import resolve
            db = get_shared_db_service()

            with db.connection() as conn:
                row = conn.execute(
                    "SELECT message, metadata FROM scheduled_items "
                    "WHERE external_uid = ? AND item_type='event'",
                    (f"caldav:{uid}",),
                ).fetchone()

            if not row:
                return {"error": f"Event '{uid}' not found"}

            title, meta_raw = row
            meta = _json.loads(meta_raw) if meta_raw else {}
            resolved = []
            for email in meta.get("attendees", []):
                matches = resolve(email, limit=1)
                resolved.append({
                    "email": email,
                    "name": matches[0].get("name", "") if matches else "",
                })

            return {"event_title": title, "attendees": resolved, "count": len(resolved)}
        except Exception as exc:
            return {"error": f"Failed to get attendees: {exc}"}
