"""SaveSynthesis — persist a versioned user-synthesis snapshot."""
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.save_synthesis_params_bag import SaveSynthesisParamsBag
from models.user_synthesis import UserSynthesisRow


class SaveSynthesis(Ability[SaveSynthesisParamsBag]):
    SYSTEM = True
    DISCOVERABLE: ClassVar[bool] = False  # synthesis-write tool; pinned on the user-synthesis config only
    NAME: ClassVar[str] = "save_synthesis"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": (Keys.summary,)
    }

    PARAMS: ClassVar[type[ParamBag] | None] = SaveSynthesisParamsBag

    def get_summary(self) -> str:
        return (
            "Persist the updated user synthesis. The text provided replaces "
            "the current synthesis as a new immutable version."
        )

    def get_examples(self) -> list[str]:
        return [
            "internal: persist the updated user synthesis",
            "internal: save a new version of the user profile",
            "internal: write the synthesized user summary",
            "internal: store the current user synthesis snapshot",
            "internal: commit the latest user profile summary",
            "internal: save the compiled user profile text",
        ]

    def get_search_tooltip(self) -> str:
        return "persist the user synthesis"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.summary: {
                "type": "string",
                "description": "The full replacement user-synthesis text, bullet-point form, max 200 words.",
            },
        },
        "required": [Keys.summary],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SaveSynthesisParamsBag) -> ToolResult:
        UserSynthesisRow.write(params.summary)
        row = UserSynthesisRow.latest()
        version = row.version if row is not None else 1
        return ToolResult.ok(f"Synthesis saved (version {version}).")
