"""
PlaceAbility — Save, list, look up, or delete named places (home, work, gym, etc.).

Stores named locations in the data_graph with kind='place'. Each place record
carries the GPS coordinates at save time plus a user-supplied label. Duplicate
saves of the same name reinforce the existing row; conflicting coordinates trigger
a cosine-supersede replacement, matching the KIND_PLACE policy in data_graph_service.

Location data is read from the telemetry dict injected by act_dispatcher_service
at dispatch time (lat, lon, location_name keys). When GPS is not available, save
returns an informative error rather than storing a null record.
"""

import json
import logging

from abilities._base import Ability
from services.data_graph_service import KIND_PLACE, get_data_graph_service

logger = logging.getLogger(__name__)

LOG_PREFIX = "[PLACE ABILITY]"

_ACTION_SAVE = "save"
_ACTION_LIST = "list"
_ACTION_GET = "get"
_ACTION_DELETE = "delete"

_SOURCE_LABEL = "place_ability"

_ERR_NO_LOCATION = "No GPS location available. Please grant location permission in your browser."
_ERR_MISSING_NAME = "A place name is required for this action."
_ERR_NOT_FOUND = "No saved place found with that name."


class PlaceAbility(Ability):
    NAME = "place"
    SEARCH_TOOLTIP = "place and location lookup"
    POLICY_CATEGORY = "Places"
    POLICY_LABELS = {
        "delete": "Delete a saved place",
        "get": "Look up a saved place",
        "list": "List saved places",
        "save": "Save a named place",
    }
    SUMMARY = (
        "Save, list, or delete named places (home, work, gym, etc.). "
        "Use when the user wants to save their current location with a name, "
        "list their saved places, or delete a named place."
    )
    EXAMPLES = [
        "save this location as home",
        "remember this place as my office",
        "where is my home?",
        "list my saved places",
        "delete the gym location",
        "save my current location as work",
        "what places do I have saved?",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [_ACTION_SAVE, _ACTION_LIST, _ACTION_GET, _ACTION_DELETE],
                "description": (
                    "save — save the current location with a name. "
                    "list — list all saved places. "
                    "get — look up a specific saved place by name. "
                    "delete — remove a saved place by name."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "The label for the place (e.g. 'home', 'work', 'gym'). "
                    "Required for save, get, delete."
                ),
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 10

    def run(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "").lower()
        name = (params.get("name") or "").strip().lower()

        if action == _ACTION_SAVE:
            return self._handle_save(name, telemetry)
        if action == _ACTION_LIST:
            return self._handle_list()
        if action == _ACTION_GET:
            return self._handle_get(name)
        if action == _ACTION_DELETE:
            return self._handle_delete(name)

        return {"status": "error", "error": f"Unknown place action: {action}"}

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_save(self, name: str, telemetry: dict | None) -> dict:
        if not name:
            return {"status": "error", "error": _ERR_MISSING_NAME}

        lat, lon, location_name = _extract_location(telemetry)
        if lat is None or lon is None:
            return {"status": "error", "error": _ERR_NO_LOCATION}

        value = json.dumps({
            "lat": lat,
            "lon": lon,
            "name": location_name or name,
            "radius_m": _DEFAULT_RADIUS_M,
        })

        try:
            result = get_data_graph_service().store(
                kind=KIND_PLACE,
                key=name,
                value=value,
                source=_SOURCE_LABEL,
            )
        except Exception as exc:
            logger.exception("%s save failed for name=%s: %s", LOG_PREFIX, name, exc)
            return {"status": "error", "error": str(exc)}

        status = result.get("status") if result else "stored"
        logger.info("%s saved place name=%s lat=%s lon=%s status=%s", LOG_PREFIX, name, lat, lon, status)
        return {
            "status": "ok",
            "action": _ACTION_SAVE,
            "name": name,
            "lat": lat,
            "lon": lon,
            "location_name": location_name or name,
        }

    def _handle_list(self) -> dict:
        try:
            rows = get_data_graph_service().fetch(kinds=[KIND_PLACE])
        except Exception as exc:
            logger.exception("%s list failed: %s", LOG_PREFIX, exc)
            return {"status": "error", "error": str(exc)}

        places = [_row_to_place(r) for r in rows if r]
        return {
            "status": "ok",
            "action": _ACTION_LIST,
            "count": len(places),
            "places": places,
        }

    def _handle_get(self, name: str) -> dict:
        if not name:
            return {"status": "error", "error": _ERR_MISSING_NAME}

        try:
            rows = get_data_graph_service().fetch(kinds=[KIND_PLACE])
        except Exception as exc:
            logger.exception("%s get failed for name=%s: %s", LOG_PREFIX, name, exc)
            return {"status": "error", "error": str(exc)}

        matched = next((r for r in rows if r and r.get("key", "").lower() == name), None)
        if matched is None:
            return {"status": "error", "error": _ERR_NOT_FOUND, "name": name}

        return {
            "status": "ok",
            "action": _ACTION_GET,
            "place": _row_to_place(matched),
        }

    def _handle_delete(self, name: str) -> dict:
        if not name:
            return {"status": "error", "error": _ERR_MISSING_NAME}

        try:
            rows = get_data_graph_service().fetch(kinds=[KIND_PLACE])
        except Exception as exc:
            logger.exception("%s delete fetch failed for name=%s: %s", LOG_PREFIX, name, exc)
            return {"status": "error", "error": str(exc)}

        matched = next((r for r in rows if r and r.get("key", "").lower() == name), None)
        if matched is None:
            return {"status": "error", "error": _ERR_NOT_FOUND, "name": name}

        row_id = matched.get("id")
        if row_id is None:
            return {"status": "error", "error": "Place record has no id.", "name": name}

        try:
            deleted = get_data_graph_service().soft_delete_by_id(row_id)
        except Exception as exc:
            logger.exception("%s delete failed for name=%s id=%s: %s", LOG_PREFIX, name, row_id, exc)
            return {"status": "error", "error": str(exc)}

        if not deleted:
            return {"status": "error", "error": "Delete failed.", "name": name}

        logger.info("%s deleted place name=%s id=%s", LOG_PREFIX, name, row_id)
        return {"status": "ok", "action": _ACTION_DELETE, "name": name}


# ── Module-level helpers ──────────────────────────────────────────────────────

_DEFAULT_RADIUS_M = 200


def _extract_location(telemetry: dict | None) -> tuple[float | None, float | None, str | None]:
    """Pull lat, lon, location_name from telemetry; returns (None, None, None) when absent."""
    if not telemetry:
        return None, None, None
    lat = telemetry.get("lat")
    lon = telemetry.get("lon")
    location_name = telemetry.get("location_name") or None
    if lat is None or lon is None:
        return None, None, location_name
    try:
        return float(lat), float(lon), location_name
    except (TypeError, ValueError):
        return None, None, location_name


def _row_to_place(row: dict) -> dict:
    """Convert a data_graph row to a concise place dict for API responses."""
    key = row.get("key", "")
    raw_value = row.get("value") or "{}"
    try:
        payload = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return {
        "name": key,
        "lat": payload.get("lat"),
        "lon": payload.get("lon"),
        "location_name": payload.get("name", key),
        "radius_m": payload.get("radius_m", _DEFAULT_RADIUS_M),
        "saved_at": row.get("first_seen_at"),
    }
