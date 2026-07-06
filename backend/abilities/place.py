"""
PlaceAbility — Save, list, look up, or delete named places (home, work, gym, etc.).

Stores named locations in the data_graph with kind='place'. Each place record
carries the GPS coordinates at save time plus a user-supplied label. Duplicate
saves of the same name reinforce the existing row; a changed value (coordinates
or label) triggers an exact-key temporal-supersede replacement, not a cosine match.

Location data is read from the telemetry dict injected by act_dispatcher_service
at dispatch time (lat, lon, location_name keys). When GPS is not available, save
returns an informative error rather than storing a null record.

Every action returns a :class:`ToolResult`: ``ok`` with a structured body the
model can read (the saved record / the place list / the resolved place / the
deletion confirmation) or ``err`` with a stable kebab-case ``code``, a one-line
``hint``, and — for input errors — a ``valid`` ladder. The dispatcher renders the
wire envelope; this ability never formats one.
"""

import json
import logging
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult
from models.place import PlaceRow
from services.place_service import PlaceService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[PLACE ABILITY]"

_ACTION_SAVE = "save"
_ACTION_LIST = "list"
_ACTION_GET = "get"
_ACTION_DELETE = "delete"

_ACTIONS = (_ACTION_SAVE, _ACTION_LIST, _ACTION_GET, _ACTION_DELETE)

_SOURCE_LABEL = "place_ability"

_ERR_NO_LOCATION = "No GPS location available. Please grant location permission in your browser."
_ERR_NOT_FOUND = "No saved place found with that name."

_HINT_NO_LOCATION = "ask the user to grant location permission, then retry save."
_HINT_NOT_FOUND = "call place with action=list to see the names that exist."


class PlaceAbility(Ability):
    # Pre-gated by the dispatcher BEFORE run(): save/get/delete each require a
    # 'name'; list requires nothing. An unknown action → one unknown-action error
    # whose valid= names these keys; a known action missing 'name' → one
    # missing-params error. The ability's run() never sees a malformed call.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        _ACTION_SAVE: (Keys.name,),
        _ACTION_LIST: (),
        _ACTION_GET: (Keys.name,),
        _ACTION_DELETE: (Keys.name,),
    }

    def get_name(self) -> str:
        return "place"

    def get_summary(self) -> str:
        return (
            "Save, list, or delete named places (home, work, gym, etc.). "
            "Use when the user wants to save their current location with a name, "
            "list their saved places, or delete a named place."
        )

    def get_examples(self) -> list[str]:
        return [
            "save this location as home",
            "remember this place as my office",
            "where is my home?",
            "list my saved places",
            "delete the gym location",
            "save my current location as work",
            "what places do I have saved?",
        ]

    def get_search_tooltip(self) -> str:
        return "place and location lookup"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": [_ACTION_SAVE, _ACTION_LIST, _ACTION_GET, _ACTION_DELETE],
                "description": (
                    "save — save the current location with a name. "
                    "list — list all saved places. "
                    "get — look up a specific saved place by name. "
                    "delete — remove a saved place by name."
                ),
            },
            Keys.name: {
                "type": "string",
                "description": (
                    "The label for the place (e.g. 'home', 'work', 'gym'). "
                    "Required for save, get, delete."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        action = cast(str, params.get(Keys.action) or "").lower()
        name = cast(str, params.get(Keys.name) or "").strip().lower()

        if action == _ACTION_SAVE:
            return self._handle_save(name, self.telemetry)
        if action == _ACTION_LIST:
            return self._handle_list()
        if action == _ACTION_GET:
            return self._handle_get(name)
        if action == _ACTION_DELETE:
            return self._handle_delete(name)

        # ACTION_REQUIRED pre-gates unknown actions, so this is unreachable in
        # practice; kept as a self-correcting belt-and-braces error.
        return ToolResult.err(
            f"Unknown place action: {action}",
            code="unknown-action",
            hint="choose one of the valid actions below.",
            valid=_ACTIONS,
        )

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_save(self, name: str, telemetry: dict[str, object] | None) -> ToolResult:
        lat, lon, location_name = _extract_location(telemetry)
        if lat is None or lon is None:
            return ToolResult.err(_ERR_NO_LOCATION, code="no-location", hint=_HINT_NO_LOCATION)

        value = json.dumps({
            "lat": lat,
            "lon": lon,
            "name": location_name or name,
            "radius_m": _DEFAULT_RADIUS_M,
        })

        result = PlaceService().store(name, value, source=_SOURCE_LABEL)

        status = result.get("status", "stored")
        logger.info("%s saved place name=%s lat=%s lon=%s status=%s", LOG_PREFIX, name, lat, lon, status)
        return ToolResult.ok(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "location_name": location_name or name,
                "source": _SOURCE_LABEL,
                "status": status,
            },
            saved=name,
        )

    def _handle_list(self) -> ToolResult:
        rows = [r.to_dict() for r in PlaceRow.live().get()]
        places = [_row_to_place(r) for r in rows if r]
        return ToolResult.ok(places, count=len(places))

    def _handle_get(self, name: str) -> ToolResult:
        rows = [r.to_dict() for r in PlaceRow.live().get()]
        matched = next((r for r in rows if r and cast(str, r.get("key", "")).lower() == name), None)
        if matched is None:
            return ToolResult.err(_ERR_NOT_FOUND, code="not-found", hint=_HINT_NOT_FOUND, name=name)
        return ToolResult.ok(_row_to_place(matched))

    def _handle_delete(self, name: str) -> ToolResult:
        rows = [r.to_dict() for r in PlaceRow.live().get()]
        matched = next((r for r in rows if r and cast(str, r.get("key", "")).lower() == name), None)
        if matched is None:
            return ToolResult.err(_ERR_NOT_FOUND, code="not-found", hint=_HINT_NOT_FOUND, name=name)

        row_id = matched.get("id")
        if row_id is None:
            return ToolResult.err(
                "Saved place has no id and cannot be deleted.",
                code="delete-failed",
                name=name,
            )

        deleted = PlaceService().delete(cast(int, row_id))
        if not deleted:
            return ToolResult.err(
                f"Could not delete the place {name!r}.",
                code="delete-failed",
                hint="retry, or list the saved places to confirm it still exists.",
                name=name,
            )

        logger.info("%s deleted place name=%s id=%s", LOG_PREFIX, name, row_id)
        return ToolResult.ok({"name": name, "deleted": True}, deleted=name)


# ── Module-level helpers ──────────────────────────────────────────────────────

_DEFAULT_RADIUS_M = 200


def _extract_location(telemetry: dict[str, object] | None) -> tuple[float | None, float | None, str | None]:
    if not telemetry:
        return None, None, None
    lat = telemetry.get("lat")
    lon = telemetry.get("lon")
    location_name = cast(str | None, telemetry.get("location_name") or None)
    if lat is None or lon is None:
        return None, None, location_name
    try:
        return float(cast(float, lat)), float(cast(float, lon)), location_name
    except (TypeError, ValueError):
        return None, None, location_name


def _row_to_place(row: dict[str, object]) -> dict[str, object]:
    key = row.get("key", "")
    raw_value = cast(str, row.get("value") or "{}")
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
