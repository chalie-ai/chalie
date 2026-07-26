# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for AsyncDelegateRunner — the Processes-panel delegate lifecycle.

Drives the runner's real daemon-thread path with no mocks of in-process code:
spawn returns the placeholder immediately (non-blocking), builds a dedicated
inert MessageProcessor off the ORIGINATING turn's real config (``_dedicated_mp``),
and runs the gated ability for real through that dedicated MP's
``dispatch_service.dispatch(name, params)`` — the dispatch-BY-NAME seam the
rewritten runner actually uses, not the retired direct ``ability.run()`` call.
Both lifecycle events (``subagent_start``/``subagent_end``) are observed
through the REAL socket-registry fan-out via an in-process client implementing
the broker's send Protocol — the same seam the WS route uses — so a dropped
event or a missing payload key fails the test loud.

``cancel()`` sets the cooperative cancel event; on completion ``_run`` replaces
the now-unwanted result with a "cancelled by the user" notice and ALWAYS routes
it to ``_deliver``. Because ``_dedicated_mp`` and ``_deliver`` key off the SAME
originating ``mp.config``, giving that mp a real config (so dispatch runs for
real) also makes delivery run for real: these tests supply an originating mp
built from a real, policy-seeded ``ProcessorConfig`` on the ``user`` channel —
the only channel AsyncDelegateRunner is ever spawned from in production — and
the delivered synthesis turn's own provider call is substituted at the ONE
sanctioned network boundary (``services.provider_service.build_client``, the
same seam ``test_message_processor_runaway_loop.py`` uses). The delivered
content is asserted on the recorded provider request, not on log lines.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar, cast
from unittest.mock import patch

import pytest

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.policy_channel import PolicyChannel
from models.policy import Policy
from models.provider_response import ProviderResponse
from services.async_delegate_runner import AsyncDelegateRunner
from services.processor_config import ProcessorConfig
from services.time_utils import parse_utc
from services.websocket import Websocket
from tests.helpers import make_stub_config

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py). The
# delivered notice always lands as a real MessageProcessor.process() turn, and
# that turn's own provider call must not reach a real LLM.
_BUILD_CLIENT = "services.provider_service.build_client"


class _RecordingClient:
    """A real socket-registry subscriber: implements the registry's send(str)
    Protocol and captures every broadcast frame as the inspectable surface the
    Processes panel renders from."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))

    def frames_for(self, event_type: str, sub_id: str) -> list[dict[str, object]]:
        return [f for f in self.frames if f.get("type") == event_type and f.get("sub_id") == sub_id]


class _RecordingProvider:
    """A real functional double at the network boundary: implements the thin
    client protocol (``get_context_limit``/``estimate_request_tokens``/``send``)
    and RECORDS every request it receives, always answering with one benign
    terminal response (empty text, no tool calls) so the delivered synthesis
    turn completes in a single step. Mirrors ``_ScriptedProvider`` in
    test_message_processor_runaway_loop.py — the harness runs for real; only
    the far side of the network call is replaced."""

    _TERMINAL = ProviderResponse(text="", model="scripted-runner-test", tool_calls=None)

    def __init__(self) -> None:
        self.requests: list[object] = []

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, dto: object) -> ProviderResponse:
        self.requests.append(dto)
        return self._TERMINAL

    def delivered_content(self) -> str:
        """Every recorded request's user-message content, concatenated — the
        observable proof of what text actually reached the model."""
        chunks: list[str] = []
        for dto in self.requests:
            for message in cast("list[dict[str, object]]", getattr(dto, "messages", [])):
                content = message.get("content")
                if isinstance(content, str):
                    chunks.append(content)
        return "\n".join(chunks)


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join every fire-and-forget daemon turn a completed delivery can spawn
    (its own drive thread, plus any post-turn skill-suggestion/thread-gist
    turn), so nothing leaks past this test's provider+DB patch into the next
    test — the same cross-test-corruption guard as
    test_message_processor_runaway_loop.py's helper of the same name."""
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


@dataclass
class _OriginMp:
    """The slice of the originating turn's mp that AsyncDelegateRunner actually
    reads (via getattr): ``.config`` builds the dedicated execution MP
    (``_dedicated_mp``); ``.turn_id``/``.metadata`` anchor the delivered
    synthesis reply on the right channel/thread (``_deliver``). A plain
    attribute carrier, not a mock of any method — the same duck-typed shape
    ``object()`` stands in for below when these fields must be ABSENT."""

    config: ProcessorConfig
    turn_id: int = -1
    metadata: dict[str, object] = field(default_factory=dict)


class _GatedAbility(Ability):
    """A real, registry-dispatchable ability that blocks until released.

    Not ``_SYNTHETIC``, and no custom ``__init__``: ``DispatchService._bind()``
    always builds a FRESH instance via ``type(template)(mp=...)`` for the real
    dispatch-by-name seam, so this fixture's synchronization primitives live on
    the CLASS (armed immediately before ``spawn``, read by whichever instance's
    ``run()`` actually executes) rather than on a per-instance constructor the
    binder can no longer supply. Leaves ``DISCOVERABLE`` at its default
    (``True``): a module-level test class is permanently visible to
    ``Ability.__subclasses__()`` for the rest of the process once collected, so
    it cannot be kept off the registry. ``test_find_tools_channel_isolation.py``
    pins the NON-discoverable roster by exact equality; staying DISCOVERABLE
    keeps this fixture off that pinned set."""

    _release: ClassVar["threading.Event | None"] = None
    _started: ClassVar["threading.Event | None"] = None
    _done: ClassVar["list[bool] | None"] = None

    def get_name(self) -> str:
        return "test_runner_gated"

    def get_search_tooltip(self) -> str:
        return "block until released — runner feature-test fixture"

    def get_summary(self) -> str:
        return "Block until released — feature-test fixture only."

    def get_examples(self) -> list[str]:
        return [
            "block until released",
            "wait for the gate",
            "hold until signalled",
            "pause for the test",
            "stay blocked",
            "wait then return",
        ]

    def get_parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, params: dict[str, object]) -> ToolResult:
        assert self._release is not None and self._started is not None and self._done is not None
        self._started.set()
        self._release.wait(timeout=5)
        self._done.append(True)
        return ToolResult.ok("gated done")


def _arm_gate() -> tuple[threading.Event, threading.Event, list[bool]]:
    """Fresh synchronization primitives for one test, wired onto the class the
    registry's per-call ``_bind()`` will instantiate."""
    release: threading.Event = threading.Event()
    started: threading.Event = threading.Event()
    done: list[bool] = []
    _GatedAbility._release = release
    _GatedAbility._started = started
    _GatedAbility._done = done
    return release, started, done


def _origin_mp() -> _OriginMp:
    """A real originating-turn stand-in with a genuine ``.config``:
    ``_dedicated_mp`` builds a real dispatch-capable MP off it, and
    ``_deliver`` lands the notice as a real ``channel="user"`` synthesis turn —
    the only channel AsyncDelegateRunner is ever spawned from in production."""
    return _OriginMp(config=make_stub_config(channel="user", role="user", policy_channel=PolicyChannel.CHAT))


def _allow_gated_tool(policy_channel: PolicyChannel) -> None:
    """Pre-seed the policy gate 'allow' so dispatch runs the ability
    immediately instead of parking on the interactive ask-gate — the fixture's
    tool name has no seeded policy row by default (would default to 'ask')."""
    Policy.upsert(policy_channel.value, "test_runner_gated", "allow")


def test_spawn_pushes_rich_snapshot_then_emits_end_on_deregister(db: sqlite3.Connection) -> None:
    """spawn() builds a real dedicated MP off the originating config and runs
    the gated ability through the real dispatch-by-name seam on a daemon
    thread (non-blocking); the panel's snapshot — polled and pushed live —
    carries everything a row renders. Cancelling then releasing lets the tool
    finish for real, deregisters it, and emits subagent_end."""
    release, started, done = _arm_gate()
    origin = _origin_mp()
    _allow_gated_tool(origin.config.policy_channel)

    client = _RecordingClient()
    Websocket._connect(client)
    provider = _RecordingProvider()
    runner = AsyncDelegateRunner()
    try:
        with patch(_BUILD_CLIENT, return_value=provider):
            placeholder = runner.spawn(_GatedAbility(), {}, origin, "Drafting the weekly digest")

            # Non-blocking: the placeholder returns while run() is still gated.
            assert placeholder.startswith("test_runner_gated dispatched (id: ")
            assert started.wait(timeout=5), "background run() never started"
            assert not done, "spawn blocked until run() finished — not backgrounded"

            # The panel's snapshot carries everything a row renders.
            rows = runner.active()
            assert len(rows) == 1
            row = rows[0]
            delegate_id = cast(str, row["sub_id"])
            assert delegate_id.startswith("test_runner_gated_")
            assert row["tool_name"] == "test_runner_gated"
            assert row["summary"] == "Drafting the weekly digest"
            assert parse_utc(cast(str, row["started_at"])).year >= 2025  # real timestamp, not datetime.min

            # The same data was pushed live to the connected client as subagent_start.
            starts = client.frames_for("subagent_start", delegate_id)
            assert len(starts) == 1
            assert starts[0]["tool_name"] == "test_runner_gated"
            assert starts[0]["summary"] == "Drafting the weekly digest"
            assert starts[0]["started_at"] == row["started_at"]  # snapshot == pushed frame

            # Stop control: flip the cancel event → run() finishes for real (the
            # ability actually dispatched) → _run delivers the cancel notice as a
            # real synthesis turn on the originating channel → deregisters +
            # emits subagent_end in `finally`.
            assert runner.cancel(delegate_id) is True
            release.set()
            for _ in range(100):
                if not runner.active():
                    break
                time.sleep(0.05)
            assert runner.active() == []
            assert done == [True]

            ends = client.frames_for("subagent_end", delegate_id)
            assert len(ends) == 1

            _drain_background_turns()
    finally:
        Websocket._disconnect(client)


def test_cancel_delivers_notice_to_model_instead_of_dropping(db: sqlite3.Connection) -> None:
    """A user-cancelled delegate whose tool still finishes must NOT silently
    drop the result: ``_run`` builds a "cancelled by the user" notice and
    ALWAYS routes it to ``_deliver``, which lands it as a real hidden-input
    synthesis turn back on the originating channel. Proven by inspecting what
    the delivered turn's own provider call actually received — the delivered
    prompt's user line — not by asserting on log lines."""
    release, started, done = _arm_gate()
    origin = _origin_mp()
    _allow_gated_tool(origin.config.policy_channel)

    provider = _RecordingProvider()
    runner = AsyncDelegateRunner()
    with patch(_BUILD_CLIENT, return_value=provider):
        runner.spawn(_GatedAbility(), {}, origin, "Researching the Maltese election")
        assert started.wait(timeout=5), "background run() never started"
        delegate_id = cast(str, runner.active()[0]["sub_id"])

        # User stops the delegate; the tool then finishes cooperatively.
        assert runner.cancel(delegate_id) is True
        release.set()
        for _ in range(100):
            if not runner.active():
                break
            time.sleep(0.05)
        assert runner.active() == [], "delegate never deregistered"
        assert done == [True], "the gated tool did not finish"

        _drain_background_turns()

    notice = "`test_runner_gated` was cancelled by the user. Ask the user for follow-up actions."
    assert notice in provider.delivered_content(), (
        f"cancel notice never reached the delivered synthesis turn; saw {provider.requests!r}"
    )


def test_cancel_unknown_id_returns_false() -> None:
    runner = AsyncDelegateRunner()
    assert runner.cancel("nope_12345678") is False
    assert runner.active() == []
