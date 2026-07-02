"""Feature test: leaked markdown in the LLM's final response is rewritten to
Chalie's HTML subset on user-facing channels — background channel output (DMN,
encoders, compaction) is not mangled."""

import sqlite3
from unittest.mock import patch

import pytest

from configs.channels import UserConfig
from configs.channels.dmn import DmnConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.provider_api import ProviderApiResponse
from services.transcript_service import Transcript

pytestmark = pytest.mark.unit

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


class _RecordingProvider:
    """Returns one fixed response and no tool calls."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.sends: list[object] = []

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, dto: object) -> int:
        return 1

    def send(self, dto: object) -> ProviderApiResponse:
        self.sends.append(dto)
        return ProviderApiResponse(text=self._response_text, model="recorder", tool_calls=None)


def _run_turn(config: ProcessorConfig, raw_input: str, response_text: str) -> str:
    """Drive the real production entry point — MessageProcessor.process — with
    the LLM boundary (Providers._resolve) swapped for a fixed-response recorder."""
    recorder = _RecordingProvider(response_text)
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        return MessageProcessor.process(raw_input, config)


def test_user_channel_markdown_is_converted_to_html(db: sqlite3.Connection) -> None:
    leaked = "**bold** then *italic* then _under_ then `code()`"
    expected = "<b>bold</b> then <i>italic</i> then <u>under</u> then <code>code()</code>"

    result = _run_turn(UserConfig(), "format this", leaked)

    # The chain's final response is the converted HTML.
    assert result == expected

    # The same converted HTML is what got persisted — no marker survives.
    history = Transcript.get_recent("user", limit=10)
    assistant = [r for r in history if r["role"] == "assistant"]
    assert assistant, "no assistant row persisted"
    content = assistant[-1]["content"]
    assert content == expected
    assert "**" not in content
    assert "`" not in content


def test_dmn_channel_markdown_is_left_verbatim(db: sqlite3.Connection) -> None:
    leaked = "**bold** and _under_ and `code`"

    result = _run_turn(DmnConfig(), "reflect", leaked)

    # Gated off: returned text is the raw markdown, untouched.
    assert result == leaked

    history = Transcript.get_recent("dmn", limit=10)
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
    content = [r for r in Transcript.get_recent("user", limit=10) if r["role"] == "assistant"][-1]["content"]
    assert content == expected


def test_snake_case_identifiers_are_not_underlined(db: sqlite3.Connection) -> None:
    """``_`` between word chars (identifiers like ``snake_case``) must NOT be
    treated as an underline marker — the response is left verbatim."""
    leaked = "the field user_id maps to write_input_row"

    result = _run_turn(UserConfig(), "explain", leaked)

    assert result == leaked
    assert "<u>" not in result
    content = [r for r in Transcript.get_recent("user", limit=10) if r["role"] == "assistant"][-1]["content"]
    assert content == leaked
