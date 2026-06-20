"""HomeAbility — control smart home devices and automations via Home Assistant.

Delegates to ``HomeCapability``'s tool handlers through the shared
:class:`CapabilityAbility` base, which owns capability loading, the
not-connected / unknown-action / handler-unavailable errors, handler dispatch,
and result wrapping. This ability supplies its metadata, the flat-action wiring,
the ``ACTION_REQUIRED`` pre-gate map, and one safeguard the base cannot know
about: an entity guardrail.

**The entity guardrail is the point of this tool.** Home Assistant answers a
service call with HTTP 200 and the *list of states it changed*; for a
nonexistent ``entity_id`` that list is empty and the call changes nothing — so a
guessed-wrong entity used to report success while doing nothing (the REST
handler ignores the response body and returns ``{"status": "ok"}``
unconditionally). A weak local model that guesses ``light.ofice`` instead of
``light.office`` would be told the light turned off when it did not.

So before delegating an entity-bearing action (``get_state`` / ``control`` /
``subscribe_events``) to the base, :meth:`run` resolves the ``entity_id`` against
the live ``/api/states`` list (one read through the capability's own
``list_devices`` handler, so credentials stay encapsulated). An unknown entity
returns ``code=unknown-entity`` with the closest real entity ids as candidates —
the model self-corrects from one error instead of silently no-op'ing. The same
guard protects ``trigger_automation``'s ``automation_id`` (``code=unknown-
automation``). When the entity exists the call delegates to ``super().run()`` and
the base performs the dispatch.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``; the
dispatcher renders the wire envelope. This ability never formats an envelope and
never imports the skill-tag formatter, and there is no rich-media card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from typing import Protocol

    class _Capability(Protocol):
        def is_connected(self) -> bool: ...
        def get_tools(self) -> list[dict[str, object]]: ...

from abilities._capability import CapabilityAbility
from abilities._params import Keys
from abilities._result import ToolResult

# Actions whose ``entity_id`` must name a real entity. ``control`` is the silent-
# false-success case the guardrail kills; ``get_state`` would otherwise raise an
# opaque 404; ``subscribe_events`` would silently subscribe to nothing.
_ENTITY_ACTIONS = ("get_state", "control", "subscribe_events")

# Number of closest-match candidates surfaced in an unknown-entity / unknown-
# automation error so a weak model can re-issue with the right id.
_MAX_CANDIDATES = 5


class HomeAbility(CapabilityAbility):
    CAPABILITY_KEY: ClassVar[str] = "home"
    DEFAULT_ACTION: ClassVar[str] = "list_devices"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the Home Assistant integration in the Brain dashboard."
    )

    # The model-facing action maps 1:1 onto the capability's handler name. The
    # base uses this to look up the handler and to build the valid= ladder for
    # unknown-action errors, so the enum below is derived from it.
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {
        "list_devices": "list_devices",
        "get_state": "get_state",
        "control": "control",
        "list_automations": "list_automations",
        "trigger_automation": "trigger_automation",
        "subscribe_events": "subscribe_events",
    }

    # Pre-gated by the dispatcher BEFORE run() (and before the policy gate): a
    # known action missing a required param yields one code=missing-params error
    # naming all missing params. Read/list actions have no hard requirement.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "list_devices": (),
        "get_state": (Keys.entity_id,),
        "control": (Keys.entity_id, Keys.service),
        "list_automations": (),
        "trigger_automation": (Keys.automation_id,),
        "subscribe_events": (Keys.entity_id,),
    }

    def get_name(self) -> str:
        return "home"

    def get_summary(self) -> str:
        return (
            "Control smart home devices and automations via the connected "
            "Home Assistant instance. Available when the user asks about "
            "devices, lights, climate, sensors, or automations."
        )

    def get_examples(self) -> list[str]:
        return [
            "turn on the living room light",
            "what's the temperature in the bedroom",
            "is the front door locked",
            "turn off all lights downstairs",
            "what automations do I have",
            "trigger the good morning routine",
            "what devices are currently on",
            "set the thermostat to 21 degrees",
        ]

    def get_search_tooltip(self) -> str:
        return "smart home control"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": [
                    "list_devices",
                    "get_state",
                    "control",
                    "list_automations",
                    "trigger_automation",
                    "subscribe_events",
                ],
                "description": (
                    "list_devices -- list entities, optionally filtered by domain or area. "
                    "get_state -- get current state of one entity by entity_id. "
                    "control -- call a service on an entity (turn_on, turn_off, set_temperature, etc.). "
                    "list_automations -- list all automations with their enabled state. "
                    "trigger_automation -- manually trigger an automation by automation_id. "
                    "subscribe_events -- subscribe to real-time state changes for an entity."
                ),
            },
            Keys.entity_id: {
                "type": "string",
                "description": (
                    "HA entity ID, e.g. 'light.living_room'. Must name a real "
                    "entity — call list_devices first if unsure."
                ),
            },
            Keys.domain: {
                "type": "string",
                "description": "list_devices: filter by HA domain (light, switch, sensor, climate, lock, etc.).",
            },
            Keys.area: {
                "type": "string",
                "description": "list_devices: filter by area name.",
            },
            Keys.service: {
                "type": "string",
                "description": "control: service to call, e.g. 'turn_on', 'turn_off', 'toggle'.",
            },
            Keys.service_data: {
                "type": "object",
                "description": "control: extra service data (brightness, temperature, etc.).",
            },
            Keys.automation_id: {
                "type": "string",
                "description": (
                    "trigger_automation: automation entity_id (e.g. "
                    "'automation.good_morning'). Must name a real automation — call "
                    "list_automations first if unsure."
                ),
            },
        },
        "required": [Keys.action],
    }

    # ── Entry point — entity guardrail, then delegate to the base ──────────────

    def run(self, params: dict[str, object]) -> ToolResult:
        action = str(params.get(Keys.action, self.DEFAULT_ACTION)).lower()
        params[Keys.action] = action

        cap = self._connected_capability()
        if cap is not None:
            if action in _ENTITY_ACTIONS:
                guard = self._guard_entity(cap, action, str(params.get(Keys.entity_id, "")))
                if guard is not None:
                    return guard
            elif action == "trigger_automation":
                guard = self._guard_automation(
                    cap, action, str(params.get(Keys.automation_id, ""))
                )
                if guard is not None:
                    return guard

        # Connected + entity verified (or a non-addressed action, or not connected
        # — the base emits the canonical not-connected error in that case): the
        # base owns dispatch and result wrapping.
        return super().run(params)

    # ── Guardrails ─────────────────────────────────────────────────────────────

    def _guard_entity(self, cap: "_Capability", action: str, entity_id: str) -> "ToolResult | None":
        entities = self._live_entities(cap, prefix=None)
        if entity_id in entities:
            return None
        return ToolResult.err(
            f"No Home Assistant entity matches {entity_id!r}.",
            code="unknown-entity",
            hint=self._closest_hint(entity_id, entities, "list_devices"),
            valid=self._closest(entity_id, entities),
            action=action,
        )

    def _guard_automation(self, cap: "_Capability", action: str, automation_id: str) -> "ToolResult | None":
        automations = self._live_entities(cap, prefix="automation.")
        if automation_id in automations:
            return None
        return ToolResult.err(
            f"No Home Assistant automation matches {automation_id!r}.",
            code="unknown-automation",
            hint=self._closest_hint(automation_id, automations, "list_automations"),
            valid=self._closest(automation_id, automations),
            action=action,
        )

    # ── Live-state lookup + closest-match helpers ──────────────────────────────

    def _connected_capability(self) -> "_Capability | None":
        from capabilities import load_capabilities

        cap = load_capabilities().get(self.CAPABILITY_KEY)
        if cap is None or not cap.is_connected():
            return None
        return cast("_Capability", cap)

    @staticmethod
    def _live_entities(cap: "_Capability", *, prefix: str | None) -> list[str]:
        from services.innate_skills._capability import dispatch_capability_handler

        tool_map = {cast(str, t["name"]): t["handler"] for t in cap.get_tools()}
        handler = tool_map["list_devices"]
        result = dispatch_capability_handler(handler, {"limit": 1000}, None)
        devices = cast(list[dict[str, object]], result.get("devices", [])) if isinstance(result, dict) else []
        ids = [cast(str, d.get("entity_id", "")) for d in devices if d.get("entity_id")]
        if prefix is not None:
            ids = [eid for eid in ids if eid.startswith(prefix)]
        return ids

    @staticmethod
    def _closest(needle: str, candidates: list[str]) -> tuple[str, ...]:
        import difflib

        needle_l = needle.lower()
        substring = [c for c in candidates if needle_l in c.lower() or c.lower() in needle_l]
        rest = [c for c in candidates if c not in substring]
        rest_ranked = sorted(
            rest,
            key=lambda c: difflib.SequenceMatcher(None, needle_l, c.lower()).ratio(),
            reverse=True,
        )
        return tuple((substring + rest_ranked)[:_MAX_CANDIDATES])

    @classmethod
    def _closest_hint(cls, needle: str, candidates: list[str], list_action: str) -> str:
        closest = cls._closest(needle, candidates)
        if closest:
            return f"closest matches: {', '.join(closest)} — re-issue with an exact id."
        return f"call home with action={list_action} to see the real ids."
