"""Feature test: leaked markdown in the LLM's final response is rewritten to
Chalie's HTML subset on user-facing channels — background channel output
(discovery, encoders, compaction) is not mangled."""

import sqlite3
from typing import cast
from unittest.mock import patch

import pytest

from configs.channels import DiscoveryConfig, ScheduledConfig, UserConfig
from configs.enums.provider_type import ProviderType
from controllers.message_processor import MessageProcessor
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from services.processor_config import ProcessorConfig
from models.transcript import Transcript

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ``ProviderService.send`` (the new spine's provider chokepoint) builds its thin
# transport client via ``services.llm_clients.factory.build_client`` directly —
# it does not go through the old ``services.providers.Providers`` facade at all.
_BUILD_CLIENT = "services.provider_service.build_client"


class _RecordingProvider:
    """Returns one fixed response and no tool calls."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.sends: list[object] = []

    def get_context_limit(self) -> int:
        return 200000


    def send(self, dto: object) -> ProviderResponse:
        self.sends.append(dto)
        return ProviderResponse(text=self._response_text, model="recorder", tool_calls=None)


def _recent(channel: str, limit: int = 10) -> list[dict[str, object]]:
    """The channel's last ``limit`` transcript rows, oldest first — composed from
    the model's Query primitives (the same ``filter``/``order_by``/``get``/
    ``to_dict()`` shape ``Transcript.by_turn`` uses)."""
    rows = Transcript.filter("channel", channel).order_by("id DESC").limit(limit).get()
    rows.reverse()
    return [row.to_dict() for row in rows]


def _run_turn(config: ProcessorConfig, raw_input: str, response_text: str) -> str:
    """Drive the real production entry point — ``MessageProcessor.process`` —
    with the LLM boundary (``ProviderService``'s client factory) swapped for a
    fixed-response recorder, then join the turn's background drive thread via
    the real ``.result()`` for its final text."""
    recorder = _RecordingProvider(response_text)
    with patch(_BUILD_CLIENT, return_value=recorder):
        mp = MessageProcessor.process(config, raw_input)
        return mp.result()


def test_user_channel_markdown_is_converted_to_html(db: sqlite3.Connection) -> None:
    leaked = "**bold** then *italic* then _under_ then `code()`"
    expected = "<b>bold</b> then <i>italic</i> then <u>under</u> then <code>code()</code>"

    result = _run_turn(UserConfig(), "format this", leaked)

    # The chain's final response is the converted HTML.
    assert result == expected

    # The same converted HTML is what got persisted — no marker survives.
    history = _recent("user", limit=10)
    assistant = [r for r in history if r["role"] == "assistant"]
    assert assistant, "no assistant row persisted"
    content = assistant[-1]["content"]
    assert content == expected
    assert "**" not in content
    assert "`" not in content


def test_scheduled_channel_markdown_is_converted_to_html(db: sqlite3.Connection) -> None:
    """The bug this change fixes: a fired schedule got no response-format
    guidance in its system prompt, so nothing steered the model toward Chalie's
    HTML subset on a surface that renders it. This drives the real
    ``MessageProcessor.process`` entrypoint for ``ScheduledConfig`` end-to-end —
    not just the assembled prompt string — and asserts both halves of the fix:
    the real request this turn sends the provider carries the HTML contract,
    and the model's markdown response still lands as HTML on the ``schedule``
    channel's transcript, exactly as already proven for ``user`` above."""
    leaked = "**bold** then *italic* then _under_ then `code()`"
    expected = "<b>bold</b> then <i>italic</i> then <u>under</u> then <code>code()</code>"

    recorder = _RecordingProvider(leaked)
    with patch(_BUILD_CLIENT, return_value=recorder):
        mp = MessageProcessor.process(ScheduledConfig(), "write a status update")
        result = mp.result()

    # The real request this turn actually sent to the provider carries the HTML
    # contract — proof RENDERS_HTML reaches the schedule channel's real system
    # prompt on the live request-assembly path, not just a directly-built one.
    # A fresh schedule thread also fires a sibling ThreadGistConfig delegate call
    # (channel-labeling) through this same patched transport, so the schedule
    # turn's own request is picked out by its CHAT type — ThreadGistConfig sets
    # ``uses_delegate_provider``, so its send carries DELEGATE instead.
    sends = [cast("ProviderRequest", s) for s in recorder.sends]
    chat_sends = [s for s in sends if s.type == ProviderType.CHAT]
    assert chat_sends, f"no CHAT-type request was sent for the scheduled turn (saw types: {[s.type for s in sends]})"
    sent_system = chat_sends[0].system
    assert "## Response format" in sent_system
    assert "NEVER use markdown syntax" in sent_system

    # The chain's final response is the converted HTML.
    assert result == expected

    # The same converted HTML is what got persisted on the schedule channel.
    history = _recent("schedule", limit=10)
    assistant = [r for r in history if r["role"] == "assistant"]
    assert assistant, "no assistant row persisted on the schedule channel"
    content = assistant[-1]["content"]
    assert content == expected
    assert "**" not in content
    assert "`" not in content


def test_discovery_channel_markdown_is_left_verbatim(db: sqlite3.Connection) -> None:
    leaked = "**bold** and _under_ and `code`"

    result = _run_turn(DiscoveryConfig(), "run a research pass", leaked)

    # Gated off: returned text is the raw markdown, untouched.
    assert result == leaked

    history = _recent("discovery", limit=10)
    assistant = [r for r in history if r["role"] == "assistant"]
    assert assistant, "no assistant row persisted"
    content = assistant[-1]["content"]
    assert content == leaked
    assert "<b>" not in content
    assert "<u>" not in content
    assert "<code>" not in content


def test_inline_code_protects_emphasis_markers(db: sqlite3.Connection) -> None:
    """Markers INSIDE an inline code span survive verbatim — only the span itself
    becomes ``<code>``."""
    leaked = "run `a_b * c` not **real**"
    expected = "run <code>a_b * c</code> not <b>real</b>"

    result = _run_turn(UserConfig(), "show code", leaked)

    assert result == expected
    content = [r for r in _recent("user", limit=10) if r["role"] == "assistant"][-1]["content"]
    assert content == expected


def test_snake_case_identifiers_are_not_underlined(db: sqlite3.Connection) -> None:
    """``_`` between word chars (identifiers like ``snake_case``) must NOT be
    treated as an underline marker — the response is left verbatim."""
    leaked = "the field user_id maps to write_input_row"

    result = _run_turn(UserConfig(), "explain", leaked)

    assert result == leaked
    assert "<u>" not in result
    content = [r for r in _recent("user", limit=10) if r["role"] == "assistant"][-1]["content"]
    assert content == leaked
