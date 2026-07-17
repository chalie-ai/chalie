"""Feature test: a cancel that lands while the provider call is in flight must
discard the returned response, not persist it.

``_step()`` has no way to abort a provider call already in flight — every
client is one blocking, non-streaming request (``services/llm_clients/*``).
The only place a mid-flight cancel can be observed is the checkpoint right
after ``_send_with_retry`` returns and before the response is stored. Before
that checkpoint existed, the single-shot (no-tool-calls) branch called
``self._store(response.text)`` unconditionally, so a turn cancelled while the
model was still generating still persisted and rendered the full answer.

Drives the real production entry point (construct inertly, ``begin()``,
``result()`` — exactly what ``MessageProcessor.process()`` does) against the
real, fully-migrated SQLite database, with the real ``TurnExecutionService``
and ``TranscriptService`` doing every read/write. The only substitution is the
LLM network boundary: ``ProviderService`` builds its thin transport client via
``services.llm_clients.factory.build_client`` (mirrors the existing pattern in
``test_message_markdown_to_html.py``); the fake client's ``send()`` calls the
real ``TurnExecutionService.cancel()`` — the exact chokepoint both
``DELETE /api/thread/<turn_id>`` and a turn's own self-cancel ability route
through — before returning a normal, terminal (no tool calls) response. That
reproduces "the DELETE lands while the call is in flight" without touching any
internal flag directly.
"""

import sqlite3
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_markdown_to_html.py for precedent).
_BUILD_CLIENT = "services.provider_service.build_client"


class _RecordingProvider:
    """Returns one fixed, terminal response and no tool calls."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        return ProviderResponse(text=self._response_text, model="recorder", tool_calls=None)


class _CancellingProvider(_RecordingProvider):
    """Same as ``_RecordingProvider``, but simulates a cancel request landing
    while this "network call" is in flight: it flips ``cancel_requested`` via
    the real ``TurnExecutionService.cancel()`` chokepoint before returning the
    response — the model still finished generating a full answer, but the
    turn was asked to stop before that answer came back."""

    def __init__(self, mp: MessageProcessor, response_text: str) -> None:
        super().__init__(response_text)
        self._mp = mp

    def send(self, dto: object) -> ProviderResponse:
        cancelled = self._mp.turn_execution_service.cancel()
        assert cancelled is not None  # sanity: the real cancel write succeeded
        return super().send(dto)


def test_cancel_observed_after_provider_returns_discards_response_and_ends_cancelled(
    db: sqlite3.Connection,
) -> None:
    """A cancel that lands while the provider call is in flight, on a
    single-shot (no-tool-calls) turn, must discard the returned essay: zero
    assistant transcript rows for this turn, and the execution row ends
    CANCELLED rather than COMPLETED."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="write me a short essay")  # inert (I2)
    provider = _CancellingProvider(mp, "Here is the full essay the model produced.")

    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()

    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert assistant_rows == []  # the essay was discarded, never persisted

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.ended_at is not None
    assert execution.state == TurnExecution.CANCELLED
    assert bool(execution.cancel_requested) is True


def test_uncancelled_turn_persists_the_same_response(db: sqlite3.Connection) -> None:
    """Contrast case, same essay, same single-shot shape, no cancel: this
    proves the checkpoint — not some unrelated no-op — is what makes the
    difference above. Without a cancel request the response IS persisted and
    the execution row ends COMPLETED."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="write me a short essay")  # inert (I2)
    provider = _RecordingProvider("Here is the full essay the model produced.")

    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        result = mp.result()

    assert result == "Here is the full essay the model produced."
    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "Here is the full essay the model produced."

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.ended_at is not None
    assert execution.state == TurnExecution.COMPLETED
    assert bool(execution.cancel_requested) is False
