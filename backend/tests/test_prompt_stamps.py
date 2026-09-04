# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the stamp contract every model-facing line PromptService
builds honours: ``[Ddd YYYY-MM-DD HH:MM]`` in the user's local timezone,
computed from the row's own stored UTC ``created_at`` (never wall-clock
"now"), plus the World State block's position relative to the history and
its exclusion of ``local_time``.

The World State block used to sit above the conversation history; it now
sits directly between the history and the stamped input line, and that
input line's stamp — not a telemetry ``local_time`` field — is the model's
only time source. A regression in either the stamp's source or the block's
position is invisible to a substring check against a canned prompt: these
tests pin real past timestamps into real rows and assert on the literal
rendered line, so a stamp silently reverting to "now" or the block sliding
back above the history both fail loudly, and a hardcoded past value in the
assertion means a stamp read from wall-clock "now" cannot accidentally
satisfy it.

Driven against the real :class:`PromptService` on real
:class:`MessageProcessor` instances, built the same way
``test_threads_serialize_turn.py`` builds a scoped turn: the synchronous
portion of ``MessageProcessor.begin()`` (turn-id allocation, input row,
execution row) replicated by hand, without its DB transaction wrapper or
background drive thread. Zero mocks.
"""

from __future__ import annotations

import sqlite3

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_WORLD_BLOCK_HEADER = "### Background Telemetry,Processes"


def _seed_telemetry(ctx: dict[str, object]) -> None:
    """Persist a heartbeat snapshot the way POST /health does (the ``db``
    fixture redirects the snapshot path into this test's tmp dir)."""
    from services.telemetry_service import TelemetryService
    TelemetryService.write(ctx)


def _open_turn(config: UserConfig, raw_input: str) -> MessageProcessor:
    """Replicate ``MessageProcessor.begin()``'s synchronous portion — turn-id
    allocation, the input row (skipped for a ``skip_input_row`` config, e.g.
    ``UserConfig(metadata={"hidden_input": True})``), then the execution
    row — without its DB transaction wrapper or background drive thread.
    The same construction every scoped-state test in this suite uses instead
    of ``begin()`` (precedent: ``test_threads_serialize_turn.py``)."""
    mp = MessageProcessor(config, raw_input=raw_input)
    mp.turn_id = mp.transcript_service.allocate_turn()
    if not mp.config.skip_input_row:
        mp.uid = mp.transcript_service.append_input(mp.raw_input)
        mp.current_transcript_id = mp.uid
    mp.turn_execution_service.open()
    return mp


# ---------------------------------------------------------------------------
# 1. The current input line is stamped from the input row's own created_at.
# ---------------------------------------------------------------------------


def test_user_line_is_stamped_from_the_input_rows_stored_utc_time(db: sqlite3.Connection) -> None:
    """The current input line's stamp comes from THIS turn's anchoring
    transcript row, converted to the user's local timezone — never from
    wall-clock "now". ``created_at`` is pinned to a fixed past moment; a
    stamp read from ``datetime.now()`` instead could never produce this
    exact line."""
    _seed_telemetry({"timezone": "Europe/Malta", "locale": "en-GB"})
    raw_input = "what time is my flight tomorrow"

    mp = _open_turn(UserConfig(), raw_input)
    db.execute("UPDATE transcript SET created_at = ? WHERE id = ?", ("2026-06-15 10:30:00", mp.uid))
    db.commit()

    prompt = mp.prompt_service.user_prompt()
    expected_line = f"[Mon 2026-06-15 12:30] user: {raw_input}"
    assert expected_line in prompt, (
        "the current input line must be stamped from the input row's stored "
        "UTC created_at converted to the user's Europe/Malta timezone, not "
        f"wall-clock now. expected_line={expected_line!r} prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 2. History rows carry the same stamp shape, from their own created_at.
# ---------------------------------------------------------------------------


def test_history_rows_carry_the_same_stamp_shape(db: sqlite3.Connection) -> None:
    """A prior, settled turn renders inside ``## Previous Messages`` stamped
    from ITS OWN stored ``created_at`` — in the same
    ``[Ddd YYYY-MM-DD HH:MM]`` shape the current input line uses — not the
    current turn's clock."""
    _seed_telemetry({"timezone": "Europe/Malta", "locale": "en-GB"})
    prior_content = "what's the weather like today"

    prior = _open_turn(UserConfig(), prior_content)
    prior.transcript_service.append_assistant("Sunny, 28C.")  # settles the prior turn
    db.execute("UPDATE transcript SET created_at = ? WHERE id = ?", ("2026-06-14 09:00:00", prior.uid))
    db.commit()

    mp = _open_turn(UserConfig(), "and what about tomorrow")
    prompt = mp.prompt_service.user_prompt()

    expected_line = f"[Sun 2026-06-14 11:00] user: {prior_content}"
    assert expected_line in prompt, (
        "a settled prior row must render in Previous Messages stamped from "
        f"its own created_at. expected_line={expected_line!r} prompt={prompt!r}"
    )
    assert prompt.index("## Previous Messages") < prompt.index(expected_line), (
        "the settled prior row must render inside the Previous Messages "
        f"block, not merely somewhere in the prompt. prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 3. The world block sits between the history and the stamped input line.
# ---------------------------------------------------------------------------


def test_world_block_sits_between_history_and_the_user_line(db: sqlite3.Connection) -> None:
    """The world-state block now sits directly above the stamped input
    line — pre-rewrite it sat above the history instead. Order must be:
    Previous Messages, then the world block, then the current input line."""
    _seed_telemetry({"timezone": "Europe/Malta", "locale": "en-GB"})

    prior = _open_turn(UserConfig(), "remind me to call the dentist")
    prior.transcript_service.append_assistant("Noted, I'll remind you.")

    mp = _open_turn(UserConfig(), "anything else on my list today")
    prompt = mp.prompt_service.user_prompt()

    history_idx = prompt.index("## Previous Messages")
    world_idx = prompt.index(_WORLD_BLOCK_HEADER)
    user_idx = prompt.index(f"] user: {mp.raw_input}")
    assert history_idx < world_idx < user_idx, (
        "the world block must sit between the history and the stamped input "
        f"line (got history={history_idx}, world={world_idx}, "
        f"user_line={user_idx}). prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 4. The world block never renders local_time.
# ---------------------------------------------------------------------------


def test_world_block_never_renders_local_time(db: sqlite3.Connection) -> None:
    """The world block renders the persisted heartbeat verbatim except
    ``local_time``, which stays hidden — the input-line stamp is the
    model's only time source, never a second, potentially stale clock."""
    _seed_telemetry({
        "timezone": "Europe/Malta", "locale": "en-GB", "local_time": "10:47",
    })

    mp = _open_turn(UserConfig(), "what's on my calendar")
    prompt = mp.prompt_service.user_prompt()

    assert _WORLD_BLOCK_HEADER in prompt, (
        f"the world block itself must still render. prompt={prompt!r}"
    )
    assert "timezone:Europe/Malta" in prompt, (
        f"the surviving telemetry fields must still render. prompt={prompt!r}"
    )
    assert "local_time" not in prompt, (
        "local_time must never reach the model even though it was "
        f"persisted — the input-line stamp is its only clock. prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 5. A row-less turn stamps from the execution's started_at instead.
# ---------------------------------------------------------------------------


def test_row_less_turn_is_stamped_from_the_execution_start(db: sqlite3.Connection) -> None:
    """A channel with no anchoring input row (``UserConfig``'s
    ``hidden_input`` metadata — ``mp.uid`` stays ``None``) stamps its input
    line from the turn execution's ``started_at`` instead, per
    ``_input_stamp``'s documented fallback. ``_input_stamp`` reads whatever
    object is on ``mp.execution`` directly rather than re-querying — so
    after updating the row we re-fetch it, the same way ``anchor_row()``
    always re-queries fresh instead of trusting a cached instance."""
    _seed_telemetry({"timezone": "Europe/Malta", "locale": "en-GB"})
    raw_input = "silent scheduled check-in"

    mp = _open_turn(UserConfig(metadata={"hidden_input": True}), raw_input)
    assert mp.uid is None, "sanity: a hidden_input turn must never get an anchoring input row"
    assert mp.execution is not None, "sanity: the execution row must have opened"

    db.execute(
        "UPDATE turn_executions SET started_at = ? WHERE id = ?",
        ("2026-06-15T10:30:00+00:00", mp.execution.id),
    )
    db.commit()
    mp.execution = TurnExecution.filter("id", mp.execution.id).first()

    prompt = mp.prompt_service.user_prompt()
    expected_line = f"[Mon 2026-06-15 12:30] user: {raw_input}"
    assert expected_line in prompt, (
        "a row-less turn must stamp its input line from turn_executions."
        "started_at when there is no anchoring transcript row. "
        f"expected_line={expected_line!r} prompt={prompt!r}"
    )
