"""
UserDeviceAbility — the user's device, network, battery and display
preferences on demand, read from the latest client heartbeat.

The interface posts a heartbeat every few minutes and the backend persists it
as the telemetry snapshot. None of that reaches the prompt on its own: the
model calls this tool when a reply depends on the user's hardware or
connection — screen size or input method, battery level, whether they are on
a slow or metered connection, dark mode, reduced motion. It takes no
parameters. The heartbeat's locale fields are carried by the system prompt
instead, and the raw coordinates never leave the backend.
"""

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from contracts.params.param_bag import ParamBag
from contracts.params.user_device_params_bag import UserDeviceParamsBag
from services.telemetry_service import TelemetryService


class UserDeviceAbility(Ability[UserDeviceParamsBag]):
    PARAMS: ClassVar[type[ParamBag] | None] = UserDeviceParamsBag
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("device", "battery", "network", "screen")
    NAME: ClassVar[str] = "user_device"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.INFORMATION

    #: Heartbeat groups the tool returns, in this order. Anything else in the
    #: snapshot (locale fields, coordinates, bookkeeping) stays out.
    _GROUPS: ClassVar[tuple[str, ...]] = ("device", "network", "battery", "preferences")

    _PARAMETERS: ClassVar[dict[str, object]] = {"type": "object", "properties": {}}

    def get_summary(self) -> str:
        return (
            "Returns info regarding user's device: hardware class, platform, "
            "screen, input method, network type, battery level and display "
            "preferences from the latest client heartbeat. Takes no parameters."
        )

    def get_examples(self) -> list[str]:
        return [
            "what device am I using right now?",
            "is my battery running low?",
            "am I on wifi or mobile data?",
            "will a wide table fit on my screen?",
            "am I in dark mode?",
            "is my laptop charging?",
        ]

    def get_search_tooltip(self) -> str:
        return "Returns info regarding user's device"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: UserDeviceParamsBag) -> ToolResult:
        snapshot = TelemetryService.read().as_dict()
        groups = {name: snapshot[name] for name in self._GROUPS if isinstance(snapshot.get(name), dict)}
        if not groups:
            return ToolResult.no_results(
                hint="No heartbeat from the user's interface yet; device info arrives once it is open.",
            )
        return ToolResult.ok(groups)
