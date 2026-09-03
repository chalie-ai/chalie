"""Feature test: ``ContextLimit`` — the compact-then-continue recovery loop.

One exception carries the two ways a request stops fitting the model's window,
because both mean the same thing and take the same recovery:

- **pre-send**, when the provider's OWN reported token count for the *previous*
  request on this channel came back at or past 90% of the window, so the next
  call is withheld before it is made;
- **mid-inference**, when the provider itself rejects the payload for length and
  the thin client raises.

Either way the turn compacts its history and re-enters the step loop, so the
retry is a *smaller* request rather than the same one sent twice.

Nothing in this path estimates anything. The only number the gate ever reads is
the one the provider itself reported and ``ProviderService`` wrote to the ledger.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real, fully-migrated SQLite DB, with the real
``ProviderService``, ``MessageProcessor`` step loop, dispatcher and
``TurnExecution`` state machine all running. The only substitution is the LLM
network boundary — ``services.provider_service.build_client`` (the same seam
``test_context_usage_signal.py`` uses) — because the whole point is to script
what the provider says about size.

Five claims, one per branch the recovery actually carries:

1. a previous request the provider reported at the 90% line withholds the next
   one, compacts, and the *retry* is what completes the turn;
2. the same turn under the line does NOT compact — which is what proves claim 1
   fired on the reported number rather than on anything incidental;
3. a provider that rejects the payload mid-inference takes the same path — the
   client raises without a turn attached, and ``ProviderService`` attaches its
   own before it reaches the handler;
4. a payload that will never fit is let out after a bounded number of attempts
   instead of recursing forever;
5. the window every one of the above is measured against comes from
   ``providers.context_window`` and nowhere else — ``pin_context_window`` reads
   the column, probes and persists it once when unset, clamps it to 200k, and
   raises rather than substituting a default when it cannot be determined.
"""

import sqlite3
import threading
import time
from typing import cast
from unittest.mock import patch

import pytest

from abilities.chat_history_compactor import ChatHistoryCompactionConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from exceptions import ContextLimit, ProviderError
from models.provider import Provider as ProviderModel
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.database import Database
from services.provider_api import MAX_CONTEXT_WINDOW
from services.provider_db_service import ProviderDbService

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"

# Deliberately below MAX_CONTEXT_WINDOW so an asserted window can only have come
# from the source under test, never from a silently-substituted ceiling.
_CLIENT_WINDOW = 120_000
#: 90% of _CLIENT_WINDOW is 108_000 — a reported usage at that line gates the NEXT
#: request on the same channel.
_OVER_CAP = 108_000
_UNDER_CAP = 1_000


class _ReportingProvider:
    """A real functional double at the network boundary: implements the thin
    client protocol and scripts what the provider *reports* about the request it
    just served.

    ``send`` answers with a ``tokens_input`` drawn from ``reports`` — one entry
    per call, the last repeating. That number is the provider's own count, and it
    is the ONLY size signal anywhere in this path: ``ProviderService`` writes it
    to the ledger, and the next request on that channel is gated against it.
    Nothing measures the outgoing payload, here or in production.

    ``raise_on_send`` models the other half — a provider that accepts the request
    and only then rejects it for length, exactly as a real 413 /
    ``context_length_exceeded`` arrives."""

    def __init__(
        self, reports: list[int], raise_on_send: int = 0, cached: int = 0,
    ) -> None:
        self._reports = reports
        self._raise_on_send = raise_on_send
        self._cached = cached
        self.sends = 0

    def get_context_limit(self) -> int:
        return _CLIENT_WINDOW

    def send(self, _dto: object) -> ProviderResponse:
        self.sends += 1
        if self.sends <= self._raise_on_send:
            # No MessageProcessor: a client has no turn. ProviderService is the
            # layer that knows whose turn it was and must attach it.
            raise ContextLimit("payload exceeds the model's maximum context length")
        return ProviderResponse(
            text="answered after making room",
            model="scripted-context-limit",
            tool_calls=None,
            tokens_input=self._reports[min(self.sends - 1, len(self._reports) - 1)],
            # Disjoint from tokens_input, exactly as the real providers report
            # it: a cached token is counted HERE and not there.
            tokens_cache_read=self._cached or None,
        )


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed turn spawns
    so they run to completion inside THIS test's provider patch and never leak
    into the next — left running, each would call ``build_client`` while the
    NEXT test holds the patch."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=deadline - time.monotonic())


def _terminal_state(mp: MessageProcessor) -> str:
    """The turn's stamped terminal state, read back off the real row — the same
    record the UI and the crash toast are driven from."""
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None, "the turn never stamped an execution row"
    return execution.state


def _run(provider: _ReportingProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def test_previous_reported_usage_at_the_line_compacts_the_next_turn(
    db: sqlite3.Connection,
) -> None:
    """A turn the provider reported at the 90% line gates the NEXT one: that
    request is withheld, the turn compacts, and the *retry* is what answers.

    Two real turns on the same channel. The first is served normally and the
    provider reports it at the cap — the number ``ProviderService`` writes to the
    ledger. The second reads that ledger row before it sends anything, so it
    compacts without ever measuring its own payload.

    The gate is *reactive* by design: it can only know a request was too big
    after one came back. That is why the first turn completes untouched and the
    second is the one that pays."""
    assert db is not None  # fixture taken for its binding side effect (real DB gateway)
    provider = _ReportingProvider(reports=[_OVER_CAP])

    first = _run(provider, "a first turn the provider serves normally")
    assert _terminal_state(first) == TurnExecution.COMPLETED
    assert not first.turn_handover, "nothing should have compacted before any usage was reported"
    sends_after_first = provider.sends

    second = _run(provider, "the next turn on the same channel")

    assert _terminal_state(second) == TurnExecution.COMPLETED
    # Compaction erases the in-flight turn's act trail, so the turn writes itself
    # a hand-over first — otherwise it continues blind to what it just lost.
    assert second.turn_handover, "the second turn never compacted"
    # More than one send: the withheld request forced a hand-over pass and a
    # retry. A turn that sailed through the gate would have cost exactly one.
    assert provider.sends > sends_after_first + 1, (
        f"the gated request was not withheld — only "
        f"{provider.sends - sends_after_first} send(s) on the second turn"
    )


def test_previous_reported_usage_under_the_line_does_not_compact(
    db: sqlite3.Connection,
) -> None:
    """The identical two turns, with only the reported number changed, do NOT
    compact — which is what proves the test above fired on that number.

    Without this pair the first test would pass just as happily if compaction
    ran on every turn regardless of usage."""
    assert db is not None
    provider = _ReportingProvider(reports=[_UNDER_CAP])

    _run(provider, "a first turn the provider serves normally")
    second = _run(provider, "the next turn on the same channel")

    assert _terminal_state(second) == TurnExecution.COMPLETED
    assert not second.turn_handover, (
        "a turn whose predecessor came in under the line must not compact"
    )


def test_cached_prompt_tokens_count_against_the_window(db: sqlite3.Connection) -> None:
    """A prompt that is mostly CACHED still fills the window, and still gates.

    The three prompt-side counters are disjoint slices of one prompt: a cached
    token is reported under ``tokens_cache_read`` and is NOT also in
    ``tokens_input``. Here ``tokens_input`` alone is far under the line and only
    the sum crosses it — so a gate reading ``tokens_input`` would sail straight
    past, on precisely the long, cache-heavy conversations it exists to catch.

    This is the regression with teeth: caching makes the uncached remainder
    SMALLER the longer a conversation runs, so the naive reading fails hardest
    exactly where the window is tightest."""
    assert db is not None
    cached = _OVER_CAP - _UNDER_CAP
    provider = _ReportingProvider(reports=[_UNDER_CAP], cached=cached)

    _run(provider, "a long, heavily cached conversation")
    second = _run(provider, "the next turn on the same channel")

    assert _terminal_state(second) == TurnExecution.COMPLETED
    assert second.turn_handover, (
        f"a prompt of {_UNDER_CAP} uncached + {cached} cached tokens fills the "
        f"{_CLIENT_WINDOW}-token window past the line, but nothing compacted — "
        f"the cached slice was not counted against the window"
    )


def test_a_one_shot_channel_is_never_gated_on_its_predecessor(
    db: sqlite3.Connection,
) -> None:
    """A ``suppress_history`` channel is exempt: a huge previous pass on it must
    not withhold the next one.

    Those channels are one-shots — a vision pass, a browse summary, the compactor
    itself — sized by the payload they were handed, never by a history that
    grows, so the predecessor's cost predicts nothing. They are also the exact
    channels ``ContextLimit.recover()`` re-raises on (there is no history to
    compact), so a gate here could only turn a request that might well have fit
    into a certain failure with no recovery behind it.

    Driven through the real compaction config, which is the one-shot that sits
    directly under the gate: the compactor runs as a sub-turn of every
    compaction, so gating it would deadlock the recovery it exists to perform."""
    assert db is not None
    provider = _ReportingProvider(reports=[_OVER_CAP])

    with patch(_BUILD_CLIENT, return_value=provider):
        first = MessageProcessor.process(
            ChatHistoryCompactionConfig(), raw_input="fold this history",
        ).result()
        sends_after_first = provider.sends
        # The predecessor reported at the cap on this very channel. A gated call
        # would raise ContextLimit here, and recover() would re-raise it rather
        # than compact — so reaching an answer at all is the assertion.
        second = MessageProcessor.process(
            ChatHistoryCompactionConfig(), raw_input="fold this history too",
        ).result()

    assert first and second, "the one-shot passes must both have produced an answer"

    assert provider.sends == sends_after_first + 1, (
        f"a one-shot channel must send exactly once regardless of what the "
        f"previous pass cost — got {provider.sends - sends_after_first} sends"
    )


def test_provider_side_rejection_takes_the_same_recovery(db: sqlite3.Connection) -> None:
    """A provider that accepts the request and only then rejects it for length
    lands on the same compact-then-continue path.

    The client raises with no turn attached (it has none); ``ProviderService``
    attaches its own before re-raising, which is the only reason the handler can
    reach a MessageProcessor to compact. The ledger is empty on this first turn
    and every report is under the cap, so the pre-send gate is provably NOT what
    fired — the send did."""
    assert db is not None
    provider = _ReportingProvider(reports=[_UNDER_CAP], raise_on_send=1)

    mp = _run(provider, "a request the provider refuses for length")

    assert provider.sends >= 2, "the rejected request was never retried"
    assert _terminal_state(mp) == TurnExecution.COMPLETED
    assert mp.turn_handover, "the turn was compacted with no hand-over captured"


def test_unfittable_payload_is_let_out_not_spun_on(db: sqlite3.Connection) -> None:
    """A payload the provider rejects however often it is compacted crashes the
    turn loudly instead of recursing forever.

    Compaction is the only lever the handler has, so a rejection that survives it
    will survive the next one too — a window smaller than the prompt itself, or
    one oversized turn. The bound is what makes that a CRASHED turn (visible,
    with a reason) rather than a thread spinning until the process dies.

    Driven through the provider-side rejection because that is the branch that
    can repeat: the pre-send gate disarms itself after one fire, precisely so a
    ledger reading that compaction has just made stale cannot re-trigger it."""
    assert db is not None
    provider = _ReportingProvider(reports=[_UNDER_CAP], raise_on_send=99)  # never accepted

    mp = _run(provider, "a prompt that cannot be made to fit")

    assert _terminal_state(mp) == TurnExecution.CRASHED
    # Bounded, not infinite: a handful of attempts, not thousands.
    assert provider.sends < 10, f"recovery was not bounded — {provider.sends} attempts"


#: An official-OpenAI config needs no host and no network to be probed — the
#: published table answers directly — which makes it the honest vehicle for
#: exercising pin_context_window's real code path with nothing stubbed out.
def _openai_config(**overrides: object) -> dict[str, object]:
    return {
        "platform": "openai", "model": "gpt-4o", "api_key": "k", **overrides,
    }


def test_pin_context_window_prefers_the_column_and_clamps_it() -> None:
    """A row that carries a window answers from the column, hard-capped at 200k.

    The clamp is applied on the way out as well as on the way in, so a row that
    predates the ceiling (or was written by hand) still cannot lift the cap."""
    service = ProviderDbService()

    assert service.pin_context_window(_openai_config(context_window=64_000)) == 64_000
    assert service.pin_context_window(
        _openai_config(context_window=5_000_000)
    ) == MAX_CONTEXT_WINDOW


@pytest.mark.parametrize("unpinned", [{}, {"context_window": None}, {"context_window": 0}])
def test_pin_context_window_probes_an_unpinned_row_and_stamps_it(
    unpinned: dict[str, object],
) -> None:
    """Missing, NULL and 0 all mean *unpinned* — none of them is a window.

    The probed answer is written back onto the config it was handed, which is
    what stops a single turn probing twice for the same provider."""
    config = _openai_config(**unpinned)

    assert ProviderDbService().pin_context_window(config) == 128_000
    assert config["context_window"] == 128_000


def test_pin_context_window_persists_so_the_row_converges(db: sqlite3.Connection) -> None:
    """The probe is a one-off: it lands in the column, not just the return value.

    ``ProviderService.send`` builds a fresh client per call, so a client's own
    probe cache dies with it — without the write-back every send would re-probe
    the host forever."""
    assert db is not None
    with Database.transaction():
        row = ProviderModel(
            name="pin-convergence", platform="openai", model="gpt-4o",
            host=None, api_key=None, dimensions=None, timeout=120,
            supports_vision=0, context_window=None,
        ).save()
    provider_id = cast(int, row.id)
    service = ProviderDbService()

    assert service.pin_context_window(_openai_config(id=provider_id)) == 128_000

    stored = service.get_provider_by_id(provider_id) or {}
    assert stored["context_window"] == 128_000, "the probe must converge the row"


def test_pin_context_window_raises_when_the_host_cannot_be_reached() -> None:
    """An unreachable provider is a loud failure, never a default.

    This is now the only case that raises, and the distinction is deliberate: a
    host that answers — even to refuse — is sized, so the fault it reported
    surfaces at send time where the message names it. A host that answers
    nothing leaves the window genuinely unknown, and inventing one here would
    silently mis-size compaction for as long as the row survives.

    Port 1 is closed, so this resolves without touching the network. Asserting
    the raise through an unknown *model* instead would have leaned on
    api.openai.com answering, and passed offline for the wrong reason."""
    with pytest.raises(ProviderError):
        ProviderDbService().pin_context_window(
            _openai_config(model="not-a-real-model", host="http://127.0.0.1:1/v1"),
        )
