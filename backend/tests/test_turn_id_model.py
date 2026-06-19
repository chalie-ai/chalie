# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the turn_id conversation model and the MP chain.

turn_id is a per-channel monotonic turn counter living ONLY on transcript — the
single boundary that replaces the old ``tool_calls.transcript_id == mp.uid``
anchor. These tests drive the REAL production functions (transcript_service
writers, ActTrail, MessageProcessor turn methods, the static ``process`` entry
point) against the REAL ``db`` fixture with ZERO mocks except the one sanctioned
provider seam (``Providers._resolve``), and assert every downstream DB effect.

Contracts under test:
  * write_input_row allocates a monotonic per-channel turn_id ATOMICALLY inside
    the INSERT — no caller ever supplies one (an async re-entry opens a NEW
    turn). write_assistant_row grounds the turn's end message to a supplied
    turn_id, or allocates one for the rare standalone caller.
  * ActTrail rows anchor only on transcript_id and carry the ability's
    act_summary; fetch_by_turn groups a turn by JOINing transcript on
    (channel, turn_id) — tool_calls carries no turn_id / channel column.
  * The MP runs as a recursive chain (no while-loop): each tool-bearing step
    writes its OWN assistant row, the tools it emitted anchor to THAT row, and
    the chain ends on the first text-only response. N tool-call steps therefore
    produce N+1 assistant rows in one turn.
  * A mid-turn compaction splits the turn_id: the continuation MP runs on the
    fresh compaction turn, is hidden-input, and is told to recover continuity.
  * _render_act_trail renders a turn's tool calls only.
  * get_previous_messages excludes the current turn (turn_id boundary).
  * _cleanup_cancelled scopes its delete to every turn_id the chain opened,
    removing the turns' tool calls (via the input-row join) before the
    transcript rows they reference.
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

# The single sanctioned mock seam: the provider client behind Providers.send.
_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


# ── helpers ──────────────────────────────────────────────────────────────────

def _turn_id(db, row_id):
    return db.execute(
        "SELECT turn_id FROM transcript WHERE id = ?", (row_id,)
    ).fetchone()[0]


def _bind_mp(channel, turn_id, uid):
    """A real MessageProcessor bound to a channel/turn — the same object the
    production loop carries.  Built via the production constructor path
    (object.__new__ + __init__), exactly like api.chat does."""
    from services.message_processor import MessageProcessor
    from tests.helpers import make_stub_config

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "raw input", {})
    mp.config = make_stub_config(channel=channel)
    mp.uid = uid
    mp.turn_id = turn_id
    return mp


class _ScriptedProvider:
    """Provider stub returning a fixed (text, tool_calls) sequence per send.

    The ONLY sanctioned seam (patched onto ``Providers._resolve``). Drives a
    full MP chain hermetically: every request fits (estimate well under the
    cap), and the script names UNKNOWN tools so the dispatcher records a
    tool_calls row carrying its act_summary WITHOUT touching the network or
    running a real ability."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def get_context_limit(self):
        return 200_000

    def estimate_request_tokens(self, dto):
        return 1

    def send(self, dto):
        from services.provider_api import ProviderApiResponse  # noqa: PLC0415
        text, tool_calls = self._script[self._i]
        self._i += 1
        return ProviderApiResponse(text=text, model="scripted-stub", tool_calls=tool_calls)


class _CompactionTriggerProvider:
    """Forces exactly one mid-turn compaction, then lets the turn finish.

    The main-job request over-caps on its first send (estimate == window),
    which drives ``_dispatch_compaction``. The nested compaction job always
    fits and emits the checkpoint text; sending it flips ``_compacted`` so the
    continuation's main-job request now fits and produces the final answer."""

    def __init__(self):
        self._compacted = False

    def get_context_limit(self):
        return 200_000

    def estimate_request_tokens(self, dto):
        job = getattr(dto, "_job_name", "") or ""
        if job.startswith("compaction"):
            return 1
        return 1 if self._compacted else 200_000

    def send(self, dto):
        from services.provider_api import ProviderApiResponse  # noqa: PLC0415
        job = getattr(dto, "_job_name", "") or ""
        if job.startswith("compaction"):
            self._compacted = True
            return ProviderApiResponse(
                text="CHECKPOINT: the user was asking about the prior topic.",
                model="stub", tool_calls=None,
            )
        return ProviderApiResponse(
            text="Final answer after compaction.", model="stub", tool_calls=None,
        )


# ── turn_id allocation ─────────────────────────────────────────────────────────

def test_write_input_row_allocates_monotonic_turn_id_per_channel(db):
    """Every input row opens the next turn for its channel — allocated atomically
    inside the INSERT, never supplied by a caller (async re-entry = new turn)."""
    from services import transcript_service as ts

    a1 = ts.write_input_row("user", "user", "hello")
    a2 = ts.write_input_row("user", "user", "again")
    b1 = ts.write_input_row("dmn", "proactive_thought", "tick")

    assert _turn_id(db, a1) == 1          # first turn on the channel
    assert _turn_id(db, a2) == 2          # bumps per channel
    assert _turn_id(db, b1) == 1          # independent counter per channel


# ── act-trail keyed on (channel, turn_id) via the transcript JOIN ──────────────

def test_act_trail_records_and_fetches_by_turn(db):
    """A turn's tool calls are anchored ONLY by transcript_id; fetch_by_turn
    groups them by joining transcript on (channel, turn_id). The ability's
    act_summary is persisted on the row and round-trips through the read."""
    from services import transcript_service as ts
    from services.act_trail import ActTrail

    uid = ts.write_input_row("user", "user", "do stuff")  # turn 1
    trail = ActTrail()
    trail.record(tool_name="weather", params={"location": "Malta"},
                 result="sunny", transcript_id=uid, summary="Checking Malta weather")
    trail.record(tool_name="search", params={"q": "x"},
                 result="hits", transcript_id=uid, summary="Searching the web")

    rows = trail.fetch_by_turn("user", 1)
    assert [r["tool_name"] for r in rows] == ["weather", "search"]
    assert rows[0]["result"] == "sunny"
    # The blue act-summary the ability returned is persisted and read back.
    assert rows[0]["summary"] == "Checking Malta weather"
    assert rows[1]["summary"] == "Searching the web"
    # isolation across turn and channel — the JOIN finds no transcript row there
    assert trail.fetch_by_turn("user", 2) == []
    assert trail.fetch_by_turn("dmn", 1) == []


def test_render_act_trail_shows_tool_calls_only(db):
    """The act-trail renders the turn's tool calls (joined via transcript_id) and
    NOT any assistant transcript text on the turn — the trail is the tool log,
    not the chain of assistant step rows."""
    from services import transcript_service as ts
    from services.act_trail import ActTrail

    uid = ts.write_input_row("user", "user", "plan a trip")          # turn 1
    # An assistant step row on the same turn must NOT leak into the trail.
    ts.write_assistant_row("user", "Let me check the weather first.", turn_id=1)
    ActTrail().record(tool_name="weather", params={"location": "Malta"},
                      result="sunny", transcript_id=uid, summary="check weather")

    mp = _bind_mp("user", turn_id=1, uid=uid)
    rendered = mp._render_act_trail()

    assert "[weather]" in rendered and "sunny" in rendered
    assert "Let me check the weather first." not in rendered


# ── turn_id is the previous-messages boundary ──────────────────────────────────

def test_previous_messages_exclude_the_current_turn(db):
    from services import transcript_service as ts

    # a complete prior exchange (turn 1) — input rows auto-allocate monotonically
    ts.write_input_row("user", "user", "what's the weather?")        # turn 1
    ts.write_assistant_row("user", "It's sunny.", turn_id=1)
    # the current turn (turn 2): input + the turn's reply-in-progress
    uid2 = ts.write_input_row("user", "user", "and tomorrow?")       # turn 2
    ts.write_assistant_row("user", "Checking the forecast...", turn_id=2)

    mp = _bind_mp("user", turn_id=2, uid=uid2)
    prev = mp.get_previous_messages()

    assert "what's the weather?" in prev                  # prior turn shown
    assert "It's sunny." in prev
    assert "and tomorrow?" not in prev                    # current turn excluded
    assert "Checking the forecast..." not in prev


# ── cancellation cleanup scoped to the turn ────────────────────────────────────

def test_cleanup_cancelled_deletes_only_the_current_turn(db):
    from services import transcript_service as ts
    from services.act_trail import ActTrail

    # a prior turn that MUST survive cancellation
    ts.write_input_row("user", "user", "keep me")                    # turn 1
    ts.write_assistant_row("user", "kept", turn_id=1)
    # the current turn (turn 2): input + a tool call anchored to its input row
    uid2 = ts.write_input_row("user", "user", "cancel me")           # turn 2
    ts.write_assistant_row("user", "partial answer", turn_id=2)
    ActTrail().record(tool_name="search", params={}, result="r",
                      transcript_id=uid2)

    mp = _bind_mp("user", turn_id=2, uid=uid2)
    mp.cancel_event.set()
    mp._cleanup_cancelled()

    # turn 2 is gone from transcript, and its tool calls (which referenced the
    # now-deleted input row) were deleted FIRST so no orphan row dangles.
    assert db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel='user' AND turn_id=2"
    ).fetchone()[0] == 0
    assert db.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE transcript_id = ?", (uid2,)
    ).fetchone()[0] == 0
    # turn 1 untouched
    assert db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel='user' AND turn_id=1"
    ).fetchone()[0] == 2


def test_cleanup_cancelled_deletes_every_chain_turn_id(db):
    """When a chain spanned multiple turn_ids (a mid-turn compaction split it),
    cancellation must remove ALL of them — both the pre- and post-compaction
    turns — while leaving an unrelated prior turn intact."""
    from services import transcript_service as ts
    from services.act_trail import ActTrail

    # an unrelated prior turn (turn 1) — survives
    ts.write_input_row("agent:chain2", "agent", "survivor")          # turn 1
    ts.write_assistant_row("agent:chain2", "kept", turn_id=1)
    # the chain's first leg (turn 2): input + a step row + its tool call
    uid2 = ts.write_input_row("agent:chain2", "agent", "step a")     # turn 2
    step2 = ts.write_assistant_row("agent:chain2", "preamble", turn_id=2)
    ActTrail().record(tool_name="x", params={}, result="r", transcript_id=step2)
    # the chain's post-compaction leg (turn 3): a checkpoint row + final row
    ts.write_input_row("agent:chain2", "compaction", "ckpt")         # turn 3
    ts.write_assistant_row("agent:chain2", "final", turn_id=3)

    mp = _bind_mp("agent:chain2", turn_id=3, uid=uid2)
    mp._chain_turn_ids = {2, 3}
    mp.cancel_event.set()
    mp._cleanup_cancelled()

    assert db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel='agent:chain2' AND turn_id IN (2, 3)"
    ).fetchone()[0] == 0
    assert db.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE transcript_id = ?", (step2,)
    ).fetchone()[0] == 0
    # the unrelated prior turn is untouched
    assert db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel='agent:chain2' AND turn_id=1"
    ).fetchone()[0] == 2


# ── the recursive MP chain (no while-loop) ─────────────────────────────────────

def test_tool_chain_writes_a_step_row_per_iteration_and_anchors_tools(db):
    """Two tool-call steps then a text-only response → one input row + two step
    rows + one final row, all on the SAME turn. Each step's tools anchor to the
    assistant step row that emitted them (never the input row), carrying their
    act_summary; the final text-only response ends the chain."""
    from services.message_processor import MessageProcessor
    from tests.helpers import make_stub_config

    script = [
        ("Looking into alpha.", [{"name": "alpha_probe", "input": {"act_summary": "probe alpha"}}]),
        ("Looking into beta.",  [{"name": "beta_probe",  "input": {"act_summary": "probe beta"}}]),
        ("Final synthesis.",    None),
    ]
    with patch(_PROVIDERS_RESOLVE, return_value=_ScriptedProvider(script)):
        result = MessageProcessor.process(
            "kick off", make_stub_config(channel="agent:chain", role="agent")
        )

    assert result == "Final synthesis."

    rows = db.execute(
        "SELECT id, role, content, turn_id FROM transcript "
        "WHERE channel='agent:chain' ORDER BY id"
    ).fetchall()
    # 1 input + 2 step rows + 1 final = 3 assistant rows, all on one turn.
    assert [r[1] for r in rows] == ["agent", "assistant", "assistant", "assistant"]
    assert {r[3] for r in rows} == {1}            # tool continuation never splits the turn
    input_id = rows[0][0]
    step1_id, step2_id = rows[1][0], rows[2][0]
    assert rows[1][2] == "Looking into alpha."
    assert rows[2][2] == "Looking into beta."
    assert rows[3][2] == "Final synthesis."

    tcs = db.execute(
        "SELECT tool_name, transcript_id, summary FROM tool_calls ORDER BY id"
    ).fetchall()
    by_tool = {t[0]: (t[1], t[2]) for t in tcs}
    assert by_tool["alpha_probe"] == (step1_id, "probe alpha")
    assert by_tool["beta_probe"] == (step2_id, "probe beta")
    # The tools anchored to their emitting step row, NOT the input row.
    assert input_id not in (step1_id, step2_id)


def test_tool_step_writes_its_row_even_when_model_emits_no_text(db):
    """A tool-bearing step ALWAYS writes a step row — even with empty model text —
    so its tools have a distinct anchor row and the interleave stays intact."""
    from services.message_processor import MessageProcessor
    from tests.helpers import make_stub_config

    script = [
        ("", [{"name": "silent_probe", "input": {"act_summary": "quiet"}}]),
        ("Done.", None),
    ]
    with patch(_PROVIDERS_RESOLVE, return_value=_ScriptedProvider(script)):
        result = MessageProcessor.process(
            "go", make_stub_config(channel="agent:silent", role="agent")
        )

    assert result == "Done."
    rows = db.execute(
        "SELECT id, role, content FROM transcript WHERE channel='agent:silent' ORDER BY id"
    ).fetchall()
    assert [r[1] for r in rows] == ["agent", "assistant", "assistant"]  # input + empty step + final
    input_id, step_id = rows[0][0], rows[1][0]
    assert rows[1][2] == ""                       # the empty-text step row is still persisted
    anchored = db.execute(
        "SELECT transcript_id FROM tool_calls WHERE tool_name='silent_probe'"
    ).fetchone()[0]
    assert anchored == step_id
    assert anchored != input_id


# ── mid-turn compaction splits the turn_id; continuation recovers ──────────────

def test_mid_turn_compaction_splits_the_turn_id(db):
    """An over-cap request mid-turn fires compaction, which writes a checkpoint on
    a FRESH turn_id; the continuation MP runs on that new turn and produces the
    final answer. The user's input row and the final answer therefore live on
    different, ordered turn_ids — the split."""
    from services.message_processor import MessageProcessor
    from services import transcript_service as ts
    from tests.helpers import make_stub_config

    # A compaction needs a backlog past the watermark, so seed a prior turn.
    ts.write_input_row("agent:compact", "agent", "prior question")   # turn 1
    ts.write_assistant_row("agent:compact", "prior answer", turn_id=1)

    provider = _CompactionTriggerProvider()
    with patch(_PROVIDERS_RESOLVE, return_value=provider):
        result = MessageProcessor.process(
            "trigger", make_stub_config(channel="agent:compact", role="agent")
        )

    assert result == "Final answer after compaction."
    assert provider._compacted is True             # the compaction send really fired

    rows = db.execute(
        "SELECT role, content, turn_id FROM transcript "
        "WHERE channel='agent:compact' ORDER BY id"
    ).fetchall()
    # the checkpoint row exists and opened a new turn
    checkpoint = [r for r in rows if r[0] == "compaction"]
    assert len(checkpoint) == 1
    assert checkpoint[0][1].startswith("CHECKPOINT")

    main_input = next(r for r in rows if r[0] == "agent" and r[1] == "trigger")
    final = next(r for r in rows if r[0] == "assistant" and r[1] == "Final answer after compaction.")
    assert main_input[2] == 2                       # the user's input opened turn 2
    assert final[2] == checkpoint[0][2]             # the final answer lives on the checkpoint's turn
    assert final[2] > main_input[2]                 # ...which is strictly later — the split


# ── the continuation MP factory ────────────────────────────────────────────────

def test_continue_inherits_turn_state_and_is_hidden_input(db):
    """The ACT-iteration continuation MP (no compaction) is hidden-input, inherits
    the turn's identity, and SHARES the chain's mutable state objects so cancel
    and cleanup see the whole chain."""
    from services import transcript_service as ts

    uid = ts.write_input_row("agent:cont", "agent", "do a thing")
    turn = ts.turn_id_of_row(uid)
    mp = _bind_mp("agent:cont", turn_id=turn, uid=uid)
    mp.active_tools = ["find_tools"]
    mp._chain_turn_ids.add(turn)
    mp.anchor = 999                                 # a step row had set this mid-chain

    child = mp._continue()

    assert child.config.skip_input_row is True      # HiddenInput=True
    assert child.uid == uid                         # same turn identity
    assert child.turn_id == turn
    assert child.anchor == uid                      # reset to the input row for the next step
    assert child.active_tools is mp.active_tools    # SAME list object
    assert child._chain_turn_ids is mp._chain_turn_ids   # SAME set object
    assert child.cancel_event is mp.cancel_event    # SAME cancel Event
    assert child.post_compaction_continuation is False
    assert child.continuation_user_query is None


def test_post_compaction_continuation_carries_query_and_review_tool(db):
    """The post-compaction continuation MP is hidden-input, surfaces the
    transcript-reading tool the model needs to recover context, and carries the
    latest user input (the latest non-assistant, non-compaction row)."""
    from services import transcript_service as ts

    uid = ts.write_input_row("agent:pc", "agent", "the original request")
    turn = ts.turn_id_of_row(uid)
    mp = _bind_mp("agent:pc", turn_id=turn, uid=uid)
    mp.active_tools = []

    child = mp._continue(post_compaction=True)

    assert child.post_compaction_continuation is True
    assert child.continuation_user_query == "the original request"
    assert "review_transcript" in child.active_tools
    assert child.config.skip_input_row is True


def test_user_prompt_renders_post_compaction_banner(db):
    """UserConfig.get_user_prompt prepends the continuity banner — restating the
    request, pointing at the Checkpoint section and review_transcript — and it
    sits BEFORE the rendered user input line."""
    from configs.channels import UserConfig
    from services.message_processor import MessageProcessor

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "continue please", {})
    mp.config = UserConfig({"channel": "user"})
    mp.uid = None
    mp.turn_id = None
    mp.post_compaction_continuation = True
    mp.continuation_user_query = "plan my trip to Rome"

    prompt = mp.config.get_user_prompt(mp)

    assert "plan my trip to Rome" in prompt          # the restated request
    assert "Checkpoint" in prompt                    # points at the Checkpoint section
    assert "review_transcript" in prompt             # surfaces the transcript-reading tool
    # the banner is part of the Current State block, above the input line
    assert prompt.index("review_transcript") < prompt.index("continue please")
