"""Feature tests for the find_tools discoverable-roster relocation — the roster
lives in the tool DESCRIPTION, not the ``select`` property description.

Weak models reliably read a tool's top-level description but skip property
descriptions, so a failed ``query`` left them with no visible forward path to
the deterministic ``select`` fallback. Moving the roster into ``get_summary()``
makes ``select`` always reachable from a zero-hit query.

These run against the REAL production stack — real ``FindToolsAbility``, the
real ``AbilityRegistry`` discoverable roster, the real schema assembler
(``get_input_schema`` → ``_inject_framework_fields``), and the real dispatcher
formatter. No mocks. They assert the model-facing surface end-to-end:

  get_summary() → get_input_schema()["description"]   ← roster lives HERE now
  get_parameters() → input_schema.properties.select   ← roster MUST be gone

They FAIL on the pre-relocation code, where the roster was injected into
``properties.select.description`` and the tool description was a fixed
one-liner.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.find_tools import FindToolsAbility
from services.message_processor import MessageProcessor
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit

# A tool that is reliably DISCOVERABLE=True, hence always in the global roster.
_KNOWN_TOOL = "web_search"


def _stub_proc() -> MessageProcessor:
    proc = object.__new__(MessageProcessor)
    proc.config = make_stub_config()
    proc._active_tools = []
    return proc


class TestRosterLocation:

    def test_roster_is_in_tool_description_not_select_property(self) -> None:
        """The model-facing tool descriptor surfaces the discoverable roster in
        its top-level ``description``; the ``select`` property carries only its
        plain one-liner — no roster.

        Pre-relocation: description was the fixed "discover more tools" string
        and the roster lived in ``select.description`` → this FAILS on old code.
        """
        ability = FindToolsAbility()
        ability.mp = _stub_proc()

        schema = ability.get_input_schema()
        description = schema["description"]
        select_desc = schema["input_schema"]["properties"]["select"]["description"]

        assert "Selectable tools:" in description, (
            f"tool description must surface the roster. Got: {description!r}"
        )
        assert f"`{_KNOWN_TOOL}`" in description, (
            f"a real discoverable tool ({_KNOWN_TOOL}) must appear in the roster. "
            f"Got: {description!r}"
        )
        assert _KNOWN_TOOL not in select_desc, (
            "the roster must NOT remain in the select property description. "
            f"Got: {select_desc!r}"
        )
        assert "Available tools:" not in select_desc, (
            "the old select-description roster marker must be gone. "
            f"Got: {select_desc!r}"
        )

    def test_tool_advertised_in_description_is_actually_selectable(self) -> None:
        """Cross-step guard: every name the description advertises as selectable
        must really resolve through the production ``select`` path into
        ``active_tools``. A description that advertised an unselectable name
        would strand exactly the weak models this change is for.
        """
        ability = FindToolsAbility()
        proc = _stub_proc()
        ability.mp = proc

        assert f"`{_KNOWN_TOOL}`" in ability.get_input_schema()["description"]

        result = ToolDispatcher._render(
            "find_tools", ability.run({"select": [_KNOWN_TOOL]})
        )

        assert _KNOWN_TOOL in proc.active_tools, (
            f"the description advertises `{_KNOWN_TOOL}` as selectable, so select "
            f"must activate it. active_tools={proc.active_tools}"
        )
        assert "error=" not in result, (
            f"selecting an advertised tool must not error. result={result!r}"
        )
