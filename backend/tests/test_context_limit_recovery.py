"""Feature test: ``ContextLimit`` — the compact-then-continue recovery loop.

One exception carries the two ways a request stops fitting the model's window,
because both mean the same thing and take the same recovery:

- **pre-flight**, when ``ProviderService.send`` measures the payload at or past
  90% of the window and refuses to make the call at all;
- **mid-inference**, when the provider itself rejects the payload for length and
  the thin client raises.

Either way the turn compacts its history and re-enters the step loop, so the
retry is a *smaller* request rather than the same one sent twice.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real, fully-migrated SQLite DB, with the real
``ProviderService``, ``MessageProcessor`` step loop, dispatcher and
``TurnExecution`` state machine all running. The only substitution is the LLM
network boundary — ``services.provider_service.build_client`` (the same seam
``test_context_usage_signal.py`` uses) — because the whole point is to script
what the provider says about size.

Four claims, one per branch the recovery actually carries:

1. a pre-flight measurement at the 90% line compacts and retries, and the
   *retry* is what completes the turn;
2. a provider that rejects the payload mid-inference takes the same path — the
   client raises without a turn attached, and ``ProviderService`` attaches its
   own before it reaches the handler;
3. a payload that will never shrink is let out after a bounded number of
   attempts instead of recursing forever;
4. the window every one of the above is measured against comes from
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
#: 90% of _CLIENT_WINDOW is 108_000 — one token past the line is a refusal.
_OVER_CAP = 108_000
_UNDER_CAP = 1_000


class _SizedProvider:
    """A real functional double at the network boundary: implements the thin
    client protocol and scripts what the provider says about *size*.

    ``estimate_request_tokens`` answers from ``sizes`` — one entry per call, the
    last repeating — which is how a shrinking payload (compaction worked) or a
    stubborn one (compaction cannot help) is expressed. ``raise_on_send`` models
    the other half: a provider that accepts the request and only then rejects it
    for length, exactly as a real 413 / ``context_length_exceeded`` arrives."""

    def __init__(
        self, sizes: list[int], raise_on_send: int = 0,
    ) -> None:
        self._sizes = sizes
        self._raise_on_send = raise_on_send
        self.measurements = 0
        self.sends = 0

    def get_context_limit(self) -> int:
        return _CLIENT_WINDOW

    def estimate_request_tokens(self, _dto: object) -> int:
        size = self._sizes[min(self.measurements, len(self._sizes) - 1)]
        self.measurements += 1
        return size

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
            tokens_input=None,
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


def _run(provider: _SizedProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def test_preflight_over_cap_compacts_then_completes(db: sqlite3.Connection) -> None:
    """A payload measured at the 90% line never reaches the provider: the turn
    compacts, re-enters the step loop, and the *smaller* retry is what answers.

    The first measurement is over the cap and the next is under it — the shape a
    working compaction produces. ``sends`` proves the first request was withheld
    rather than sent and rejected: had the cap not fired, the turn would have
    completed on send #1 with no compaction at all."""
    assert db is not None  # fixture taken for its binding side effect (real DB gateway)
    provider = _SizedProvider(sizes=[_OVER_CAP, _UNDER_CAP])

    mp = _run(provider, "a very long conversation")

    assert provider.measurements >= 2, "the turn never re-measured — no retry happened"
    # Exactly two sends, and neither is the over-cap payload: the hand-over pass
    # then the retry. Had the cap not fired the turn would have completed on send
    # #1 with no compaction at all (1); had the over-cap request been sent and
    # rejected instead of withheld, there would be three.
    assert provider.sends == 2, (
        f"expected the over-cap request to be withheld and only the hand-over "
        f"pass plus the retry sent, got {provider.sends} sends"
    )
    assert _terminal_state(mp) == TurnExecution.COMPLETED
    # Compaction erases the in-flight turn's act trail, so the turn writes itself
    # a hand-over first — otherwise it continues blind to what it just lost.
    assert mp.turn_handover, "the turn was compacted with no hand-over captured"


def test_provider_side_rejection_takes_the_same_recovery(db: sqlite3.Connection) -> None:
    """A provider that accepts the request and only then rejects it for length
    lands on the same compact-then-continue path.

    The client raises with no turn attached (it has none); ``ProviderService``
    attaches its own before re-raising, which is the only reason the handler can
    reach a MessageProcessor to compact. Every measurement here is under the cap,
    so the pre-flight branch is provably NOT what fired — the send did."""
    assert db is not None
    provider = _SizedProvider(sizes=[_UNDER_CAP], raise_on_send=1)

    mp = _run(provider, "a request the provider refuses for length")

    assert provider.sends >= 2, "the rejected request was never retried"
    assert _terminal_state(mp) == TurnExecution.COMPLETED
    assert mp.turn_handover, "the turn was compacted with no hand-over captured"


def test_unshrinkable_payload_is_let_out_not_spun_on(db: sqlite3.Connection) -> None:
    """A payload that stays over the cap however often it is compacted crashes
    the turn loudly instead of recursing forever.

    Compaction is the only lever the handler has, so a hit that survives it will
    survive the next one too — a window smaller than the prompt itself, or one
    oversized turn. The bound is what makes that a CRASHED turn (visible, with a
    reason) rather than a thread spinning until the process dies."""
    assert db is not None
    provider = _SizedProvider(sizes=[_OVER_CAP])  # never shrinks

    mp = _run(provider, "a prompt that cannot be made to fit")

    assert _terminal_state(mp) == TurnExecution.CRASHED
    assert provider.sends == 0, "an over-cap request must never be sent"
    # Bounded, not infinite: a handful of attempts, not thousands.
    assert provider.measurements < 10, (
        f"recovery was not bounded — {provider.measurements} attempts"
    )


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
