"""Feature tests for the settle-triggered memory step.

After any turn settles on an in-scope channel, ``MemoryStepService.on_settle``
spawns a daemon ``MessageProcessor`` on the SAME channel with a cloned config
so it runs the four memory tools against the settled turn's act trail. This
suite drives that end-to-end: the step fires, it sees its own tool trail, the
next turn is never polluted, rapid settles coalesce into at most two runs, and
the REST thread listing never surfaces the step's synthetic input row.

Every test drives a real user turn through the production ``MessageProcessor``
entry point against the real SQLite ``db``, with the LLM network boundary
swapped for ``_ScriptedProvider`` — the same seam
``test_context_limit_recovery.py`` uses — because the whole point is to verify
the orchestration, not the model. The provider patch is held across the drive
AND the step wait: the settle spawns a daemon that builds its own client, so
releasing the patch between the two would race the step against the real
network boundary.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from abilities.chat_history_compactor import ChatHistoryCompactionConfig
from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from controllers.message_processor import MessageProcessor
from models.memory_graph import MemoryGraphRow
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from services.memory_step_service import (
    MEMORY_STEP_PROMPT,
    MemoryStepService,
    _in_scope,
    memory_step_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "real_memory_step")]

_BUILD_CLIENT = "services.provider_service.build_client"

_MEMORY_TOOLS = {Recall.NAME, SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME}

#: Message payloads are compared as ``json.dumps`` strings, which escapes the
#: prompt's newlines — so containment checks use its (newline-free) first line.
_PROMPT_HEAD = MEMORY_STEP_PROMPT.splitlines()[0]


@dataclass(frozen=True)
class _Say:
    """Script entry: answer with prose (settles the exchange)."""

    text: str


@dataclass(frozen=True)
class _Call:
    """Script entry: answer with one tool call."""

    name: str
    input: dict[str, str]


def _is_step_request(request: ProviderRequest) -> bool:
    """A step request is recognised by its tool set: exactly the four memory
    tools, which no other channel config carries."""
    return {t["name"] for t in (request.tools or [])} == _MEMORY_TOOLS


def _texts(request: ProviderRequest) -> str:
    """The request's message list as one searchable JSON string."""
    return json.dumps(request.messages)


class _ScriptedProvider:
    """A scripted double at the ``services.provider_service.build_client``
    seam. Serves STEP requests (recognised by the four-memory-tool set) from
    ``step_script`` and all other requests from ``turn_script``. Each script
    is consumed one entry per call, the last entry repeating.

    ``step_gate`` (optional) parks EVERY step request until the caller sets
    it. Before the set only one step request can exist — the first run's
    first iteration, since each following iteration needs the previous one
    served — so the gate holds exactly that request while the caller
    interleaves a second settle.
    """

    def __init__(
        self,
        turn_script: list[_Say | _Call],
        step_script: list[_Say | _Call],
        step_gate: threading.Event | None = None,
    ) -> None:
        self._turn_script = list(turn_script)
        self._step_script = list(step_script)
        self._step_gate = step_gate
        self._turn_idx = 0
        self._step_idx = 0
        self.requests: list[ProviderRequest] = []

    def get_context_limit(self) -> int:
        return 120_000

    def send(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if _is_step_request(request):
            if self._step_gate is not None:
                self._step_gate.wait(timeout=10)
            entry = self._step_script[min(self._step_idx, len(self._step_script) - 1)]
            self._step_idx += 1
        else:
            entry = self._turn_script[min(self._turn_idx, len(self._turn_script) - 1)]
            self._turn_idx += 1
        if isinstance(entry, _Say):
            return ProviderResponse(text=entry.text, model="scripted", tokens_input=1000)
        return ProviderResponse(
            # Non-empty prose alongside the call: an empty completion would
            # trigger the empty-completion steering path instead.
            text="calling tool",
            model="scripted",
            tool_calls=[{"id": "tc1", "name": entry.name, "input": entry.input}],
            tokens_input=1000,
        )


def _step_requests(provider: _ScriptedProvider) -> list[ProviderRequest]:
    return [r for r in provider.requests if _is_step_request(r)]


@pytest.fixture(autouse=True)
def _fresh_step_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the singleton so every test starts with a clean service."""
    monkeypatch.setattr(MemoryStepService, "_instance", None)


@pytest.fixture(autouse=True)
def _quiet_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress voice synthesis — its ONNX model spawns threads that
    outlive the test and pollute the step-wait loop."""
    monkeypatch.setattr(MessageProcessor, "_voice_presynthesis", lambda self: None)


def _await_step(timeout_s: float = 15.0) -> None:
    """Block until no memory-step thread is running and no pending work
    remains. Raises ``AssertionError`` on timeout naming the stuck state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        svc = MemoryStepService.instance()
        if svc._running or svc._pending:
            time.sleep(0.02)
            continue
        if any(t.name.startswith("memory-step-") for t in threading.enumerate()):
            time.sleep(0.02)
            continue
        return
    svc = MemoryStepService.instance()
    hints = []
    if svc._running:
        hints.append(f"_running={svc._running!r}")
    if svc._pending:
        hints.append(f"_pending keys={list(svc._pending)}")
    threads = [t.name for t in threading.enumerate() if t.name.startswith("memory-step-")]
    if threads:
        hints.append(f"live threads={threads}")
    raise AssertionError(f"step did not settle within {timeout_s}s: {'; '.join(hints)}")


def _drive_turn(raw_input: str = "hello") -> MessageProcessor:
    """Construct, begin, and block until the turn settles. Call INSIDE the
    provider patch: the settle spawns the step daemon, which builds its own
    client, so the patch must outlive this call."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)
    mp.begin()
    mp.result()
    return mp


# ── tests ────────────────────────────────────────────────────────────────────

def test_settled_user_turn_fires_one_step_with_turn_provenance() -> None:
    """A settled user turn fires exactly one memory step that writes a
    graph row carrying the turn's transcript ids as provenance.

    The provider returns a single prose answer for the user turn, then the
    step calls ``save_graph`` with the expected subject/contents and answers
    with a confirming prose line."""
    provider = _ScriptedProvider(
        turn_script=[_Say("the answer")],
        step_script=[
            _Call(SaveGraph.NAME, {"subject": "user.residence", "contents": "Lisbon"}),
            _Say("recorded"),
        ],
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        mp = _drive_turn("I live in Lisbon")
        _await_step()

    steps = _step_requests(provider)
    assert len(steps) >= 1, "no step requests were served"

    row = MemoryGraphRow.by_subject("user.residence")
    assert row is not None, "save_graph never wrote the row"
    turn_ids = Transcript.ids_for_turn(Channel.USER.value, mp.turn_id)
    assert turn_ids, "turn had no transcript rows"
    assert json.loads(row.sourced_from) == turn_ids


def test_step_iteration_two_sees_its_own_act_trail() -> None:
    """The step's second provider request carries both the tool call and
    its result — proving the step is not blind to the tool it just called.

    The scripted provider answers the first step iteration with a
    ``save_graph`` tool call; the dispatcher runs it and feeds the result
    back. The second iteration's message list must contain both the call
    and the success response (which includes ``saved``)."""
    provider = _ScriptedProvider(
        turn_script=[_Say("the answer")],
        step_script=[
            _Call(SaveGraph.NAME, {"subject": "user.residence", "contents": "Lisbon"}),
            _Say("recorded"),
        ],
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        _drive_turn("I live in Lisbon")
        _await_step()

    steps = _step_requests(provider)
    assert len(steps) >= 2, f"expected at least 2 step requests, got {len(steps)}"
    text = _texts(steps[1])
    assert SaveGraph.NAME in text, f"step 2 did not carry the tool call: {text!r}"
    assert "saved" in text, f"step 2 did not carry the tool result: {text!r}"


def test_completed_step_does_not_fire_a_second_step(db: sqlite3.Connection) -> None:
    """After the step finishes, no additional step is scheduled.

    We await the step, sleep briefly, await again, and assert the step
    request count is stable. We also assert the DB carries exactly one
    role='memory' row (the step's synthetic input) and zero assistant
    rows in that turn — the step never settles an assistant row."""
    provider = _ScriptedProvider(
        turn_script=[_Say("the answer")],
        step_script=[
            _Call(SaveGraph.NAME, {"subject": "user.residence", "contents": "Lisbon"}),
            _Say("recorded"),
        ],
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        _drive_turn("I live in Lisbon")
        _await_step()
        n = len(_step_requests(provider))
        time.sleep(0.3)
        _await_step()

    assert len(_step_requests(provider)) == n, "a second step was scheduled after the first finished"

    memory_count = db.execute("SELECT COUNT(*) FROM transcript WHERE role='memory'").fetchone()[0]
    assert memory_count == 1, f"expected 1 role='memory' row, got {memory_count}"
    memory_turn = db.execute("SELECT turn_id FROM transcript WHERE role='memory'").fetchone()[0]
    assistant_count = db.execute(
        "SELECT COUNT(*) FROM transcript WHERE turn_id = ? AND role='assistant'",
        (memory_turn,),
    ).fetchone()[0]
    assert assistant_count == 0, f"memory turn had {assistant_count} assistant rows"


def test_rapid_settles_coalesce_into_at_most_two_runs() -> None:
    """Two settles on the same channel in quick succession coalesce into
    at most two step runs: the first run plus one trailing run.

    Turn 1's step is parked on ``step_gate`` before it can be served; the
    channel is already marked running by then, so turn 2's settle must land
    in the pending slot rather than spawn a third run. Releasing the gate
    lets run 1 finish and the trailing run absorb turn 2. Each run is a
    2-iteration script (save + prose), so exactly two runs serve exactly
    four step requests, and the trailing run's ``save_graph`` upsert merges
    turn 2's provenance into the row run 1 wrote."""
    gate = threading.Event()
    provider = _ScriptedProvider(
        turn_script=[_Say("answer")],
        step_script=[
            _Call(SaveGraph.NAME, {"subject": "probe", "contents": "x"}),
            _Say("done"),
            _Call(SaveGraph.NAME, {"subject": "probe", "contents": "x"}),
            _Say("done"),
        ],
        step_gate=gate,
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        turn1 = _drive_turn("first")
        # Turn 1's settle marked the channel running before result() returned,
        # so this second settle coalesces regardless of step-thread progress.
        turn2 = _drive_turn("second")
        gate.set()
        _await_step()

    assert turn1.turn_id != turn2.turn_id, "the two settles must be distinct turns"
    assert len(_step_requests(provider)) == 4, (
        f"expected exactly 4 step requests (2 runs x 2 iterations), "
        f"got {len(_step_requests(provider))}"
    )
    row = MemoryGraphRow.by_subject("probe")
    assert row is not None, "save_graph never wrote the row"
    merged = json.loads(row.sourced_from)
    for turn in (turn1, turn2):
        turn_ids = Transcript.ids_for_turn(Channel.USER.value, turn.turn_id)
        assert turn_ids, f"turn {turn.turn_id} had no transcript rows"
        for tid in turn_ids:
            assert tid in merged, f"turn {turn.turn_id} id {tid} missing from provenance {merged!r}"


def test_out_of_scope_settles_fire_nothing(db: sqlite3.Connection) -> None:
    """Channels outside the in-scope set never fire a memory step.

    Directly asserts ``_in_scope`` for several fixed and dynamic channel
    names, then drives a real out-of-scope one-shot (``compaction``) and
    verifies that no step requests were served and no role='memory' rows
    were written."""
    assert _in_scope("dmn") is False
    assert _in_scope("delegate:pim") is False
    assert _in_scope("compaction") is False
    assert _in_scope("external-agent:crm") is True
    assert _in_scope("user") is True

    provider = _ScriptedProvider(
        turn_script=[_Say("folded")],
        step_script=[_Say("should not run")],
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        MessageProcessor.process(
            ChatHistoryCompactionConfig(), raw_input="fold this history",
        ).result()
        _await_step()

    steps = _step_requests(provider)
    assert steps == [], f"out-of-scope channel fired {len(steps)} step(s)"
    memory_count = db.execute("SELECT COUNT(*) FROM transcript WHERE role='memory'").fetchone()[0]
    assert memory_count == 0, f"expected 0 role='memory' rows, got {memory_count}"


def test_step_request_carries_exactly_the_four_memory_tools_and_the_prompt() -> None:
    """The first step request carries exactly the four memory-tool names and
    the memory instruction in its message list.

    This pins the contract: no more, no fewer tools, and the prompt is
    visible in the messages so a regression that drops it is caught."""
    provider = _ScriptedProvider(
        turn_script=[_Say("the answer")],
        step_script=[
            _Call(SaveGraph.NAME, {"subject": "user.residence", "contents": "Lisbon"}),
            _Say("recorded"),
        ],
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        _drive_turn("I live in Lisbon")
        _await_step()

    steps = _step_requests(provider)
    assert len(steps) >= 1, "no step requests were served"
    step1 = steps[0]
    tool_names = {t["name"] for t in (step1.tools or [])}
    assert tool_names == _MEMORY_TOOLS, f"step carried wrong tools: {tool_names!r}"
    assert _PROMPT_HEAD in _texts(step1), "memory instruction not found in step messages"


def test_step_is_invisible_to_the_next_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The memory step's synthetic input row, tool call and tool result must
    never leak into the next user turn's message history or the REST thread
    listing.

    Turn 2's request that carries turn 1's answer proves history flows, and
    must not contain the memory instruction, the saved contents, or the
    subject. The thread feed must list both real turns (their openers appear
    as previews) while the step's turn — whose only row is role='memory' —
    never forms a thread."""
    client, _db, _store = authed_client
    provider = _ScriptedProvider(
        turn_script=[_Say("first answer"), _Say("second answer")],
        step_script=[
            _Call(SaveGraph.NAME, {"subject": "user.residence", "contents": "Lisbon"}),
            _Say("recorded"),
        ],
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        _drive_turn("hello")
        _await_step()
        _drive_turn("again")

    turn2_first = next(
        (r for r in provider.requests if not _is_step_request(r) and "first answer" in _texts(r)),
        None,
    )
    assert turn2_first is not None, "turn 2 request carrying turn 1's history was not found"
    t2_text = _texts(turn2_first)
    assert _PROMPT_HEAD not in t2_text, "memory step prompt leaked into turn 2"
    assert "Lisbon" not in t2_text, "memory step content leaked into turn 2"
    assert "user.residence" not in t2_text, "memory step subject leaked into turn 2"

    resp = client.get("/api/threads/all")
    assert resp.status_code == 200
    body_str = json.dumps(resp.get_json())
    assert "hello" in body_str, "turn 1's opener missing from the thread feed"
    assert "again" in body_str, "turn 2's opener missing from the thread feed"
    assert _PROMPT_HEAD not in body_str, "memory step prompt leaked into the thread feed"
    assert "Lisbon" not in body_str, "memory step content leaked into the thread feed"


def test_memory_step_config_overrides_and_leaves_the_source_untouched() -> None:
    """``memory_step_config`` must return a clone with all the expected
    overrides and must NOT mutate the original config — a shared-instance
    mutation would poison every later turn on the channel."""
    cfg = UserConfig()
    original_role = cfg.role
    original_skip = cfg.skip_transcript
    original_always = list(cfg.always_available)

    clone = memory_step_config(cfg, [1, 2, 3])

    assert clone.role == "memory"
    assert clone.skip_transcript is True
    assert clone.skip_input_row is False
    assert clone.memory_seed is False
    assert clone.recall_k == 10
    assert clone.BROADCASTS_STATE is False
    assert clone.USAGE_TYPE == "background"
    assert getattr(clone, "_source_transcript_ids", None) == [1, 2, 3]
    assert clone.always_available == [Recall.NAME, SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME]

    assert cfg.role == original_role
    assert cfg.role != "memory"
    assert cfg.skip_transcript is original_skip
    assert cfg.skip_transcript is False
    assert cfg.always_available == original_always
    assert cfg.always_available != [Recall.NAME, SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME]
