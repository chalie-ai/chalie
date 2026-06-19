# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — per-channel MessageProcessor serialisation.

Two turns on the SAME channel must run one at a time: the second turn cannot
open its turn (write its input row, advance the channel's turn cursor, fire a
compaction) until the first has fully finished — otherwise a delegate finishing
into the user channel while the user's own turn is still synthesising would let
both advance the cursor and both kick off a compaction. Two turns on DIFFERENT
channels run fully in parallel — the lock is per-channel, never global.

Both drive the real top-level entry point ``MessageProcessor.process`` on real
worker threads against the real file-backed SQLite db. The only stand-in is the
single sanctioned LLM boundary ``Providers._resolve`` — here a provider whose
``send`` blocks (or rendezvouses on a barrier) so the timing window the lock
governs is observable. Zero internal mocks.
"""

import threading
import time
from unittest.mock import patch

import pytest

from services.message_processor import MessageProcessor
from services.provider_api import ProviderApiResponse
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"
_INPUT_ROLE = "agent"


def _count_input_rows(db, channel):
    return db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND role = ?",
        (channel, _INPUT_ROLE),
    ).fetchone()[0]


def _turn_ids(db, channel):
    return [
        r[0]
        for r in db.execute(
            "SELECT turn_id FROM transcript WHERE channel = ? AND role = ? "
            "ORDER BY id ASC",
            (channel, _INPUT_ROLE),
        ).fetchall()
    ]


class _BlockingProvider:
    """Resolved-provider stand-in whose ``send`` blocks until released, so one
    turn can be pinned 'inside the loop, holding the channel lock' while we
    observe whether a second same-channel turn is kept out."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._guard = threading.Lock()

    def get_context_limit(self):
        return 200000

    def estimate_request_tokens(self, dto):
        return 1

    def send(self, dto):
        with self._guard:
            self.calls += 1
        self.entered.set()
        self.release.wait(timeout=10)
        return ProviderApiResponse(text="ok", model="blocker", tool_calls=None)


class _BarrierProvider:
    """Resolved-provider stand-in whose ``send`` rendezvouses two callers on a
    barrier. Both turns must reach ``send`` concurrently for the barrier to trip;
    a global lock would keep the second out, breaking the barrier on timeout."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=5)
        self.broken = False

    def get_context_limit(self):
        return 200000

    def estimate_request_tokens(self, dto):
        return 1

    def send(self, dto):
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            self.broken = True
            raise
        return ProviderApiResponse(text="ok", model="barrier", tool_calls=None)


def test_same_channel_turns_serialise_and_never_double_advance(db):
    """A second same-channel turn is blocked at the lock until the first finishes,
    so the channel's turn cursor advances by exactly one per turn — never twice
    from two compactions racing on one channel."""
    channel = "agent:lockcheck"
    provider = _BlockingProvider()

    def _run_turn(text):
        MessageProcessor.process(text, make_stub_config(channel=channel, role=_INPUT_ROLE))

    t_a = threading.Thread(target=_run_turn, args=("first",), daemon=True)
    t_b = threading.Thread(target=_run_turn, args=("second",), daemon=True)

    with patch(_PROVIDERS_RESOLVE, return_value=provider):
        try:
            # A reaches the provider: its input row is written and it now holds
            # the channel lock for the rest of the turn.
            t_a.start()
            assert provider.entered.wait(timeout=5), "turn A never reached the provider"
            assert _count_input_rows(db, channel) == 1

            # B contends for the SAME channel lock. While A holds it, B must not
            # be able to run _setup / open its turn / reach the provider.
            provider.entered.clear()
            t_b.start()
            time.sleep(0.5)
            assert _count_input_rows(db, channel) == 1, (
                "a second same-channel turn opened its turn while the first still "
                "held the lock — per-channel serialisation is not in force"
            )
            assert not provider.entered.is_set(), (
                "the second same-channel turn reached the provider before the first "
                "released the lock"
            )

            provider.release.set()
        finally:
            provider.release.set()
            t_a.join(timeout=10)
            t_b.join(timeout=10)

    assert not t_a.is_alive() and not t_b.is_alive(), "a locked turn never drained"
    # Two turns opened; the cursor advanced by exactly one each (no double-advance).
    assert _turn_ids(db, channel) == [1, 2]
    assert provider.calls == 2


def test_different_channel_turns_run_in_parallel(db):
    """Two turns on distinct channels both reach the provider at the same time —
    the lock is per-channel, so neither blocks the other."""
    provider = _BarrierProvider()
    errors: list = []

    def _run_turn(channel):
        try:
            MessageProcessor.process("hi", make_stub_config(channel=channel, role=_INPUT_ROLE))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the assertion
            errors.append(exc)

    t_a = threading.Thread(target=_run_turn, args=("agent:par-a",), daemon=True)
    t_b = threading.Thread(target=_run_turn, args=("agent:par-b",), daemon=True)

    with patch(_PROVIDERS_RESOLVE, return_value=provider):
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

    assert not provider.broken, (
        "two different-channel turns could not both reach the provider at once — "
        "the lock is global, not per-channel"
    )
    assert not errors, f"different-channel turns raised: {errors}"
    assert not t_a.is_alive() and not t_b.is_alive()
