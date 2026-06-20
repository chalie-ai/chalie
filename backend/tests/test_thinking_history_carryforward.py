"""Feature test — the high-deliberation ``thinking`` pre-pass MIRRORS THE PARENT
turn exactly: same user message, same tool surface; only the system prompt differs.

Bug history: the thinking pre-pass built its own bespoke user prompt (exploration
prefix + raw input) with NO conversation history, so token-per-request collapsed
(turn 1 ~15k → thinking ~2k → turn 2 ~16k) and the deliberation reasoned about the
latest message in a vacuum. The durable fix makes ThinkingConfig delegate the
user-message body to the PARENT's own rendered output and snapshot the parent's
live tool tier — so the thinking pass sends the identical request the parent is
about to send, the only delta being a lean deliberation system prompt.

This drives the REAL prod hot path the orchestrator fires at turn 0:
``ToolDispatcher(parent).dispatch("thinking", {})`` (message_processor._seed_turn_zero
line 418) → ``ThinkingAbility.run`` → ``ThinkingConfig.*`` → the real
``Providers.send`` request builder. Real SQLite (``db``), the real transcript
factory (``write_input_row``), the real ``thinking`` ability through the real
dispatcher/policy gate. The ONLY stand-in is the external LLM boundary
(``Providers._resolve``) — the single sanctioned seam, here a recording provider
that captures the EXACT request the thinking pass sends. Zero internal mocks.
"""

import sqlite3
from typing import TYPE_CHECKING, cast

import pytest
from unittest.mock import patch

from services.provider_api import ProviderApiRequest, ProviderApiResponse, ThinkingLevel

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


class _RecordingProvider:
    """Stand-in for the resolved LLM provider — the single sanctioned boundary.

    Captures every ``send(dto)`` request (system + messages + tools) and
    returns a one-shot ``NOTHING`` ProviderApiResponse (no tool calls) so the
    thinking ACT loop ends after a single send. ``estimate_request_tokens``
    returns 1 so the pre-flight over-cap check never triggers.

    Providers._resolve now returns a ProviderClient. Updated from
    send_messages/build_request_body interface to send(dto)/
    estimate_request_tokens(dto) interface.
    """

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self) -> None:
        self.sends: list[dict[str, object]] = []

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, dto: object) -> int:
        """Return 1 so the pre-flight over-cap check never triggers."""
        return 1

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        self.sends.append({
            "system": dto.system,
            "messages": dto.messages,
            "tools": dto.tools,
            "thinking_mode": dto.thinking_mode,
        })
        return ProviderApiResponse(text="NOTHING", model="recorder", tool_calls=None)


def _build_parent(raw_input: str) -> "MessageProcessor":
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.transcript_service import write_input_row

    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, raw_input, None)
    parent.config = UserConfig()
    parent.uid = write_input_row("user", "user", raw_input)
    # _setup seeds active_tools BEFORE _seed_turn_zero dispatches thinking; put the
    # parent in that same pre-dispatch state so the snapshot is faithful.
    parent.active_tools = list(parent.config.always_available or [])
    return parent


def test_thinking_prepass_mirrors_parent_user_message(db: sqlite3.Connection) -> None:
    """The thinking request body IS the parent's, verbatim — history included.

    Pre-fix: ThinkingConfig.get_user_prompt returned only an exploration prefix +
    raw input, so the captured request had no '## Previous Messages' and never
    mentioned BRATWURST. Post-fix it carries the parent's full rendered body."""
    from services.transcript_service import write_input_row
    from abilities._dispatcher import ToolDispatcher

    # Prior conversation on the real 'user' channel — the SAME factory production
    # uses to persist turns. The codeword is the cross-turn marker.
    write_input_row("user", "user", "Remember this: the codeword is BRATWURST.")
    write_input_row("user", "assistant", "Got it — the codeword is BRATWURST.")

    parent = _build_parent("What was the codeword again?")

    # The parent's exact rendered history block, captured at the pre-dispatch state
    # the thinking pass snapshots (no act-trail yet on either side).
    parent_prev = parent.get_previous_messages()
    assert "BRATWURST" in parent_prev, "fixture sanity: prior turns not persisted"

    recorder = _RecordingProvider()
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        # Exactly how _seed_turn_zero fires the high-deliberation pass.
        ToolDispatcher(parent).dispatch("thinking", {})

    assert recorder.sends, "thinking pre-pass never reached the provider boundary"
    thinking_req = recorder.sends[-1]
    content = cast(str, cast(list[dict[str, object]], thinking_req["messages"])[0]["content"])

    # Sanity: this really is the high-deliberation pass (config pins 'high').
    # thinking_mode is now a ThinkingLevel enum, not a plain string.
    assert thinking_req["thinking_mode"] == ThinkingLevel.HIGH

    # The mirror: the parent's exact Previous Messages block (verbatim) and input
    # line are present in the thinking request — it is the parent's body, not a stub.
    assert f"## Previous Messages\n{parent_prev}" in content, (
        "thinking pre-pass did not carry the parent's verbatim Previous Messages block"
    )
    assert "BRATWURST" in content, "thinking pre-pass dropped the prior-turn codeword"
    assert "user: What was the codeword again?" in content, (
        "thinking pre-pass dropped the parent's current input line"
    )


def test_thinking_system_prompt_is_deliberation_overlay_only(db: sqlite3.Connection) -> None:
    """The ONLY delta vs the parent is the system prompt: a lean deliberation
    overlay, NOT the parent's persona/unified system prompt."""
    from abilities._dispatcher import ToolDispatcher
    from abilities.thinking import _DELIBERATION_SYSTEM_PROMPT

    from services.processor_config import ProcessorConfig

    parent = _build_parent("Plan the trip.")
    parent_system = cast(ProcessorConfig, parent.config).get_system_prompt(parent)

    recorder = _RecordingProvider()
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        ToolDispatcher(parent).dispatch("thinking", {})

    thinking_system = recorder.sends[-1]["system"]
    assert thinking_system == _DELIBERATION_SYSTEM_PROMPT, (
        "thinking system prompt is not the deliberation overlay"
    )
    assert "DO NOT INVOKE TOOLS" in thinking_system
    # The deliberation overlay must NOT drag in the parent's full system prompt.
    assert thinking_system != parent_system


def test_thinking_prepass_mirrors_parent_tool_surface(db: sqlite3.Connection) -> None:
    """The thinking pass offers the SAME tool list the parent's turn does —
    snapshotted from the parent's live ``active_tools`` (parent-config-agnostic)."""
    from services.transcript_service import write_input_row
    from abilities._dispatcher import ToolDispatcher
    from abilities._registry import AbilityRegistry

    for i in range(6):
        write_input_row("user", "user", f"Earlier turn number {i} about project ATLAS.")
        write_input_row("user", "assistant", f"Acknowledged turn {i} on project ATLAS.")

    parent = _build_parent("Summarise where we are on ATLAS.")
    parent_tool_names = {t["name"] for t in AbilityRegistry.build_tools(parent)}
    assert parent_tool_names, "fixture sanity: parent has no tools to mirror"

    recorder = _RecordingProvider()
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        ToolDispatcher(parent).dispatch("thinking", {})

    thinking_tools = cast(list[dict[str, object]], recorder.sends[-1]["tools"] or [])
    thinking_tool_names = {t["name"] for t in thinking_tools}

    assert thinking_tool_names == parent_tool_names, (
        "thinking pass tool surface diverged from the parent's"
    )
    # And the conversation history rode along — not collapsed to a bare stub.
    content = cast(str, cast(list[dict[str, object]], recorder.sends[-1]["messages"])[0]["content"])
    assert "project ATLAS" in content
