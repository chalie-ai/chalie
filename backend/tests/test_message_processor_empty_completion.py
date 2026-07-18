"""Feature test: an LLM completion that is empty after reasoning-tag stripping
and carries no tool calls must be treated as "no response" — a failed provider
attempt that re-enters the existing retry budget — instead of being persisted
as an empty assistant reply.

``_strip_think_blocks`` strips unclosed ``<think>`` blocks to ``""``, and its
contract is: "Callers must treat an empty result as 'no response', never as an
empty answer." The MessageProcessor's ``_send_with_retry`` owns the retry
budget; before this test existed, an empty text with no tool calls would
settle the turn via ``_store(response.text)`` and ``_end(response.text)``,
leaving a blank assistant row in the transcript.

Drives the real production entry point (construct inertly, ``begin()``,
``result()`` — exactly what ``MessageProcessor.process()`` does) against the
real, fully-migrated SQLite database, with the real ``TurnExecutionService``
and ``TranscriptService`` doing every read/write. The only substitution is the
LLM network boundary: ``ProviderService`` builds its thin transport client via
``services.llm_clients.factory.build_client`` (mirrors the existing pattern in
``test_message_markdown_to_html.py``); the fake client's ``send()`` returns
the configured response without touching any internal flag directly.
"""

import sqlite3
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import _MAX_PROVIDER_ATTEMPTS, MessageProcessor
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_markdown_to_html.py for precedent).
_BUILD_CLIENT = "services.provider_service.build_client"


class _EmptyCompletionProvider:
    """Returns an empty completion (no text, no tool calls) on every send."""

    def __init__(self) -> None:
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        self.sends += 1
        return ProviderResponse(text="", model="empty-provider", tool_calls=None)


class _EmptyThenRealProvider:
    """Returns empty on first send, then real text on second."""

    def __init__(self, real_text: str) -> None:
        self._real_text = real_text
        self._attempts = 0

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        self._attempts += 1
        if self._attempts == 1:
            return ProviderResponse(text="", model="empty-provider", tool_calls=None)
        return ProviderResponse(text=self._real_text, model="real-provider", tool_calls=None)


class _EmptyWithToolCallsProvider:
    """Returns empty text but WITH tool calls — should NOT be treated as failure."""

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        return ProviderResponse(
            text="",
            model="tool-provider",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test"}}],
        )


def test_empty_completion_on_every_attempt_retries_and_crashes_turn(db: sqlite3.Connection) -> None:
    """When the provider returns empty text with no tool calls on every
    attempt, the turn must NOT persist an empty assistant row and must
    terminate as a crashed turn (ProviderRetriesExhaustedError is caught by
    _drive and stamps CRASHED), with _MAX_PROVIDER_ATTEMPTS send calls observed."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="write me a short essay")  # inert (I2)
    provider = _EmptyCompletionProvider()

    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()

    # The full resend budget was spent on the unusable completions
    assert provider.sends == _MAX_PROVIDER_ATTEMPTS

    # No assistant rows persisted — the empty completion was not stored
    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert assistant_rows == []

    # Turn execution ended in a crashed state (ProviderRetriesExhaustedError is caught by _drive)
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.ended_at is not None
    assert execution.state == TurnExecution.CRASHED


def test_empty_then_real_text_persists_real_text(db: sqlite3.Connection) -> None:
    """When the provider returns empty text on the first attempt and real
    text on the second, the turn must persist the real text exactly once and
    terminate successfully."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="write me a short essay")  # inert (I2)
    provider = _EmptyThenRealProvider("Hello there")

    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        result = mp.result()

    assert result == "Hello there"
    assert provider._attempts == 2  # exactly one retry recovered the turn
    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "Hello there"

    # Turn execution ended successfully
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.ended_at is not None
    assert execution.state == TurnExecution.COMPLETED


def test_empty_text_with_tool_calls_not_treated_as_failure(db: sqlite3.Connection) -> None:
    """When the provider returns empty text BUT with tool calls present,
    this must NOT be treated as a failure — the step proceeds to tool
    dispatch. This is a narrow unit test on _send_with_retry returning the
    response unchanged."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="test tool dispatch")  # inert (I2)

    # Directly test _send_with_retry: it should return the response unchanged
    # when tool_calls are present, even if text is empty
    from models.provider_request import ProviderRequest
    request = ProviderRequest(system="", messages=[], type=None)

    # Inject a fake provider service that returns empty text with tool calls
    original_provider_service = mp.provider_service
    fake_provider = _EmptyWithToolCallsProvider()
    mp.provider_service = fake_provider

    response = mp._send_with_retry(request)

    # Restore original provider service
    mp.provider_service = original_provider_service

    # The response should be returned as-is (not retried)
    assert response is not None
    assert response.text == ""
    assert response.tool_calls == [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
