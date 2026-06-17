"""Feature test: leaked markdown in the LLM's final response is rewritten to
Chalie's HTML subset on user-facing channels — background channel output (DMN,
encoders, compaction) is not mangled."""

import pytest
from unittest.mock import patch

from configs.channels import UserConfig
from configs.channels.dmn import DmnConfig
from services.message_processor import MessageProcessor
from services.provider_api import ProviderApiResponse
from services.transcript_service import get_recent, write_input_row

pytestmark = pytest.mark.unit

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


class _RecordingProvider:
    """Returns one fixed response and no tool calls."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.sends = []

    def get_context_limit(self):
        return 200000

    def estimate_request_tokens(self, dto):
        return 1

    def send(self, dto):
        self.sends.append(dto)
        return ProviderApiResponse(text=self._response_text, model="recorder", tool_calls=None)


def _build_mp(config, raw_input: str) -> MessageProcessor:
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, raw_input, {})
    mp.config = config
    mp.uid = write_input_row(config.channel, config.role, raw_input)
    mp.current_iteration = 0
    mp.thinking_level = "low"
    mp.thinking_override = None
    mp.active_tools = list(config.always_available or [])
    return mp


def test_user_channel_markdown_is_converted_to_html(db):
    leaked = "**bold** then *italic* then _under_ then `code()`"
    expected = "<b>bold</b> then <i>italic</i> then <u>under</u> then <code>code()</code>"

    recorder = _RecordingProvider(leaked)
    mp = _build_mp(UserConfig(), "format this")
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        result = mp._loop()
    mp._record(result)

    # The loop's final response is the converted HTML.
    assert result == expected

    # The same converted HTML is what got persisted — no marker survives.
    history = get_recent("user", limit=10)
    assistant = [r for r in history if r["role"] == "assistant"]
    assert assistant, "no assistant row persisted"
    content = assistant[-1]["content"]
    assert content == expected
    assert "**" not in content
    assert "`" not in content


def test_dmn_channel_markdown_is_left_verbatim(db):
    leaked = "**bold** and _under_ and `code`"

    recorder = _RecordingProvider(leaked)
    mp = _build_mp(DmnConfig(), "reflect")
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        result = mp._loop()
    mp._record(result)

    # Gated off: returned text is the raw markdown, untouched.
    assert result == leaked

    history = get_recent("dmn", limit=10)
    assistant = [r for r in history if r["role"] == "assistant"]
    assert assistant, "no assistant row persisted"
    content = assistant[-1]["content"]
    assert content == leaked
    assert "<b>" not in content
    assert "<u>" not in content
    assert "<code>" not in content


def test_inline_code_protects_emphasis_markers(db):
    """Markers INSIDE an inline code span survive verbatim — only the span itself
    becomes ``<code>``."""
    leaked = "run `a_b * c` not **real**"
    expected = "run <code>a_b * c</code> not <b>real</b>"

    recorder = _RecordingProvider(leaked)
    mp = _build_mp(UserConfig(), "show code")
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        result = mp._loop()
    mp._record(result)

    assert result == expected
    content = [r for r in get_recent("user", limit=10) if r["role"] == "assistant"][-1]["content"]
    assert content == expected


def test_snake_case_identifiers_are_not_underlined(db):
    """``_`` between word chars (identifiers like ``snake_case``) must NOT be
    treated as an underline marker — the response is left verbatim."""
    leaked = "the field user_id maps to write_input_row"

    recorder = _RecordingProvider(leaked)
    mp = _build_mp(UserConfig(), "explain")
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        result = mp._loop()
    mp._record(result)

    assert result == leaked
    assert "<u>" not in result
    content = [r for r in get_recent("user", limit=10) if r["role"] == "assistant"][-1]["content"]
    assert content == leaked
