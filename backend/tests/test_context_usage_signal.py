"""Feature test: the ``context_usage`` TurnSignal — the context meter's live feed.

``ProviderService.send`` is the one place both halves of the meter's fraction
exist together (the ``window`` it computed to enforce the pre-flight cap, and the
``tokens_input`` the reply came back with), so it reports the same reading it
already enforces. One CHAT call is exactly one move of the meter.

Drives the real production entry point (construct inertly, ``begin()``,
``result()`` — what ``MessageProcessor.process()`` does) against the real,
fully-migrated SQLite DB, with the real ``ProviderService``, ``TurnSignal``,
``MessageProcessor.push_websocket`` gate and ``Websocket`` fan-out all running.
The frames are observed through the REAL socket registry via an in-process client
implementing the broker's ``send(str)`` Protocol — the same seam the WS route uses
— so a dropped frame or a missing payload key fails loud. The only substitution is
the LLM network boundary: ``services.provider_service.build_client`` (the exact
seam ``test_message_processor_runaway_loop.py`` uses).

Three claims, one per branch the emit site actually carries:

1. a ``user`` turn emits ``context_usage`` carrying ``tokens_input`` +
   ``context_window`` (and the ``turn_id``/``type`` the store keys off);
2. a background turn (``DiscoveryConfig`` — a thin split of UserConfig whose ONLY
   relevant difference is ``BROADCASTS_STATE = False``) emits NONE, while its CHAT
   call demonstrably still happened — proving the silence is the
   ``push_websocket`` gate, not a turn that never ran;
3. a provider that reports no token count is SILENT rather than emitting a
   fabricated 0, which would render as a real, empty context.
"""

import json
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from configs.channels.discovery import DiscoveryConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from services.processor_config import ProcessorConfig
from services.websocket import Websocket

pytestmark = pytest.mark.unit

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py).
_BUILD_CLIENT = "services.provider_service.build_client"

# Deliberately BELOW MAX_CONTEXT_WINDOW (200_000) so an asserted window of
# 120_000 can only have come from this client's real declared limit — a
# hard-coded constant or a silently-substituted cap would read 200_000 and fail.
_CLIENT_WINDOW = 120_000
_TOKENS_IN = 4_321


class _RecordingClient:
    """A real socket-registry subscriber: implements the registry's ``send(str)``
    Protocol and captures every broadcast frame — the inspectable surface the
    context meter renders from."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))

    def usage_frames(self) -> list[dict[str, object]]:
        return [f for f in self.frames if f.get("status") == "context_usage"]


class _TokenReportingProvider:
    """A real functional double at the network boundary: implements the thin
    client protocol (``get_context_limit``/``estimate_request_tokens``/``send``)
    and answers with one benign terminal response (empty text, no tool calls) so
    the turn completes in a single step — one CHAT call, hence one expected move
    of the meter. ``tokens_input=None`` models a provider whose native payload
    exposes no usage counter."""

    def __init__(self, tokens_input: int | None = _TOKENS_IN) -> None:
        self._tokens_input = tokens_input
        self.sends = 0

    def get_context_limit(self) -> int:
        return _CLIENT_WINDOW

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        self.sends += 1
        return ProviderResponse(
            text="",
            model="scripted-context-usage",
            tool_calls=None,
            tokens_input=self._tokens_input,
        )


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed turn spawns
    (skill suggestion, thread gist) so they run to completion inside THIS test's
    provider+DB patch and never leak into the next — the same cross-test
    corruption guard as test_message_processor_runaway_loop.py's helper of the
    same name. Each is a whole MessageProcessor turn on its own daemon; left
    running it would call ``build_client`` while the NEXT test holds the patch."""
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


def _run(
    config: ProcessorConfig, provider: _TokenReportingProvider, raw_input: str,
) -> tuple[MessageProcessor, _RecordingClient]:
    """Drive a real turn on *config* to termination against *provider* with a live
    socket registered, then quiesce any post-turn daemon turns inside the patch so
    the test stays hermetic. Returns the turn and the socket that watched it."""
    client = _RecordingClient()
    Websocket._connect(client)
    try:
        mp = MessageProcessor(config, raw_input=raw_input)  # inert (I2)
        with patch(_BUILD_CLIENT, return_value=provider):
            mp.begin()
            mp.result()
            _drain_background_turns()
    finally:
        Websocket._disconnect(client)
    return mp, client


def test_user_turn_emits_context_usage_with_tokens_and_window(db: sqlite3.Connection) -> None:
    """A user turn's CHAT call reports the meter's reading on the wire: the
    measured ``tokens_input`` from the reply and the ``context_window`` the cap
    was enforced against, addressed by the ``type``/``turn_id`` pair the store
    keys off (``stores/contextUsage.ts``). Carries the counts, not a ratio — the
    surface owns the rendering."""
    assert db is not None  # fixture taken for its binding side effect (real DB gateway)
    provider = _TokenReportingProvider()

    mp, client = _run(UserConfig(), provider, "how full is my context")

    usage = client.usage_frames()
    assert len(usage) == 1, f"expected exactly one meter move per CHAT call, got {usage!r}"
    frame = usage[0]
    assert frame["tokens_input"] == _TOKENS_IN
    assert frame["context_window"] == _CLIENT_WINDOW  # the client's real limit, under the cap
    assert frame["turn_id"] == mp.turn_id
    assert frame["type"] == "user"


def test_background_turn_emits_no_context_usage(db: sqlite3.Connection) -> None:
    """A background turn stays silent — the ``push_websocket`` gate
    (``BROADCASTS_STATE``) is the only thing between it and the wire.

    ``DiscoveryConfig`` is a thin split of UserConfig: same prompt assembly, same
    CHAT provider call, and its ``type_value()`` is a real addressable type
    (``discovery``) — so ``BROADCASTS_STATE = False`` is the ONE difference from
    the emitting case above. The call demonstrably happened (``sends``), which is
    what makes this silence the gate rather than a turn that never reached a
    provider."""
    assert db is not None
    provider = _TokenReportingProvider()

    _mp, client = _run(DiscoveryConfig(), provider, "quietly research something")

    assert provider.sends >= 1, "background turn never reached the provider — silence proves nothing"
    assert client.usage_frames() == []


def test_provider_reporting_no_tokens_is_silent_not_zero(db: sqlite3.Connection) -> None:
    """A provider whose reply carries no token count emits NOTHING rather than a
    fabricated ``tokens_input=0``: the surface hides a meter it has no reading
    for, whereas a 0 would render as a real, empty context. The CHAT call still
    happened — the emit site's ``is not None`` check is what withholds the
    frame."""
    assert db is not None
    provider = _TokenReportingProvider(tokens_input=None)

    _mp, client = _run(UserConfig(), provider, "how full is my context")

    assert provider.sends >= 1, "turn never reached the provider — silence proves nothing"
    assert client.usage_frames() == []
    # A user turn is otherwise live on the wire: proving frames DID flow rules out
    # a dead socket faking the silence above.
    assert client.frames != [], "no frames at all — the socket, not the guard, was silent"
    assert all(f.get("tokens_input") is None for f in client.frames)
