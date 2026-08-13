"""Feature tests for the compaction-fired UserSynthesisGenerator vertical.

Real MessageProcessor turns against a real SQLite database — only the provider
transport (``build_client``) is patched, per the test_turn_handover /
test_memory_step conventions. The scripted double routes each request by the
opening line of its system prompt (content-addressed, never by call ordinal:
post-turn daemons such as skill-suggest and thread-gist send through the same
transport, so ordinals are racy), and the main-channel refusal that triggers a
real ContextLimit recovery is likewise addressed by content — the first main
send whose body carries the marker text, refused exactly once.

Covered:
- Trigger e2e: refused user turn → recovery → compactor → generator; the
  synthesis request carries BOTH the prior version and the folded window
  (captured pre-watermark), a scripted ``save_synthesis`` lands version 2, and
  the next user turn reads it back through ``user_definition`` (loop closed).
- Negative: the same dance on the schedule channel writes a compaction row but
  never wakes the generator and never writes a synthesis row.
- Ability contract: params-bag validation, versioned writes, INTERNAL policy.
- No-op exit: a prose-only generator turn completes without persisting
  anything — no synthesis row, no transcript residue.
- Containment: a generator turn that yields only empty completions crashes in
  isolation while the parent user turn completes untouched.
"""

import sqlite3
import threading
import time
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from abilities._result import ToolResult
from abilities._turn_handover_config import TurnHandoverConfig
from abilities.chat_history_compactor import ChatHistoryCompactionConfig
from abilities.recall import Recall
from abilities.save_synthesis import SaveSynthesis
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.channels.user_synthesis import UserSynthesisConfig
from configs.enums.channels import Channel
from configs.enums.param_key import Keys
from contracts.params.save_synthesis_params_bag import SaveSynthesisParamsBag
from controllers.message_processor import MessageProcessor
from exceptions import ContextLimit
from models.compaction import Compaction
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from models.user_synthesis import UserSynthesisRow
from services.policy_manager import INTERNAL
from services.user_synthesis_generator import UserSynthesisGenerator
from tests._tool_result_harness import built

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "real_user_synthesis")]

_BUILD_CLIENT = "services.provider_service.build_client"

#: System-prompt opening lines — the content addresses the double routes on.
_SYNTHESIS_MARK = UserSynthesisConfig().system_prompt.splitlines()[0]
_HANDOVER_MARK = TurnHandoverConfig().system_prompt.splitlines()[0]
_COMPACTION_MARK = ChatHistoryCompactionConfig().system_prompt.splitlines()[0]

#: Background threads a test must outwait before asserting on shared state.
_BACKGROUND_THREAD_NAMES = ("skill-suggest", "thread-gist", "user-synthesis-generator")


def _system_of(dto: object) -> str:
    system = getattr(dto, "system", "")
    return system if isinstance(system, str) else ""


def _body_of(dto: object) -> str:
    messages = getattr(dto, "messages", None) or []
    return "\n".join(str(m.get("content") or "") for m in messages)


def _tools_of(dto: object) -> set[str]:
    return {str(t.get("name")) for t in (getattr(dto, "tools", None) or [])}


@dataclass(frozen=True)
class _Say:
    text: str


@dataclass(frozen=True)
class _Call:
    name: str
    input: "dict[str, str]"


class _ScriptedProvider:
    """Transport double: synthesis turns follow a script, everything else gets
    canned prose — except the one content-addressed main-channel refusal."""

    def __init__(
        self,
        synthesis_script: "list[_Say | _Call] | None" = None,
        refuse_body_marker: "str | None" = None,
    ) -> None:
        self.requests: "list[object]" = []
        self.synthesis_requests: "list[object]" = []
        self._script = list(synthesis_script or [])
        self._script_i = 0
        self._refuse_marker = refuse_body_marker
        self._refused = False

    def get_context_limit(self) -> int:
        return 200_000

    def send(self, request: object) -> ProviderResponse:
        self.requests.append(request)
        system = _system_of(request)
        if _SYNTHESIS_MARK in system:
            self.synthesis_requests.append(request)
            return self._next_scripted()
        if _HANDOVER_MARK in system:
            return ProviderResponse(text="I was mid-task.", model="scripted")
        if _COMPACTION_MARK in system:
            return ProviderResponse(text="## Now\nthe compacted checkpoint", model="scripted")
        if (
            self._refuse_marker is not None
            and not self._refused
            and self._refuse_marker in _body_of(request)
        ):
            self._refused = True
            raise ContextLimit("payload exceeds the model's maximum context length")
        return ProviderResponse(text="answered after making room", model="scripted")

    def _next_scripted(self) -> ProviderResponse:
        # Past-the-end synthesis sends repeat the last entry so a steering
        # loop (empty completions) sees a stable script, not an IndexError.
        entry = self._script[min(self._script_i, len(self._script) - 1)]
        self._script_i += 1
        if isinstance(entry, _Call):
            return ProviderResponse(
                text="calling tool",
                model="scripted-synthesis",
                tool_calls=[{"id": f"tc{self._script_i}", "name": entry.name, "input": entry.input}],
            )
        return ProviderResponse(text=entry.text, model="scripted-synthesis")


@pytest.fixture(autouse=True)
def _fresh_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own singleton — no single-flight state bleeds over."""
    monkeypatch.setattr(UserSynthesisGenerator, "_instance", None)


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Wait for post-turn daemons and background MPs to finish inside the patch."""
    deadline = time.monotonic() + timeout_s
    stragglers: "list[threading.Thread]" = []
    while time.monotonic() < deadline:
        stragglers = [
            t for t in threading.enumerate()
            if t.name.startswith(_BACKGROUND_THREAD_NAMES) or t.name.startswith("turn-")
        ]
        if not stragglers:
            return
        time.sleep(0.02)
    raise AssertionError(f"background turns never drained: {[t.name for t in stragglers]}")


def _await_generator(timeout_s: float = 15.0) -> None:
    """Block until the generator is idle — no live run, no pending buffer."""
    generator = UserSynthesisGenerator.instance()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        alive = any(t.name == "user-synthesis-generator" for t in threading.enumerate())
        if not generator._running and not generator._pending and not alive:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"generator never went idle: running={generator._running} "
        f"pending={len(generator._pending)}"
    )


def _terminal_state(mp: MessageProcessor) -> str:
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None, "no TurnExecution row for the driven turn"
    return execution.state


def test_user_compaction_generates_and_next_turn_reads_the_new_version() -> None:
    """Refused user turn → real recovery → compactor → generator → version 2,
    and the following user turn reads version 2 back in its own request."""
    UserSynthesisRow.write("- Known: user answers to Alex")
    provider = _ScriptedProvider(
        synthesis_script=[
            _Call(SaveSynthesis.NAME, {Keys.summary: "- Answers to Alex\n- Adopted a corgi named Biscuit"}),
            _Say("Synthesis updated and saved."),
        ],
        refuse_body_marker="carry on with the planning",
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        first = MessageProcessor.process(UserConfig(), "I adopted a corgi named Biscuit")
        first.result()
        _drain_background_turns()
        second = MessageProcessor.process(UserConfig(), "carry on with the planning for the weekend")
        second.result()
        _await_generator()
        _drain_background_turns()

        before = len(provider.requests)
        third = MessageProcessor.process(UserConfig(), "one more planning question")
        third.result()
        _drain_background_turns()

    assert _terminal_state(first) == TurnExecution.COMPLETED
    assert _terminal_state(second) == TurnExecution.COMPLETED
    assert Compaction.latest_main(Channel.USER.value) is not None

    # The generator turn ran on exactly the scripted two round-trips, with
    # exactly the two pinned tools.
    assert len(provider.synthesis_requests) == 2
    synthesis_request = provider.synthesis_requests[0]
    assert _tools_of(synthesis_request) == {Recall.NAME, SaveSynthesis.NAME}

    # Payload proof: the prior version AND the folded window both rode in —
    # the window was captured before Compaction.write advanced the watermark,
    # after which those rows can no longer be read.
    synthesis_body = _body_of(synthesis_request)
    assert "user answers to Alex" in synthesis_body
    assert "corgi named Biscuit" in synthesis_body

    row = UserSynthesisRow.latest()
    assert row is not None
    assert row.version == 2
    assert "Adopted a corgi named Biscuit" in row.content

    # Reader loop closed: the third turn's own main request carries version 2 —
    # user_definition opens the user-prompt body. The capital-A phrasing is
    # unique to the saved synthesis (turn 1 said "adopted", and its rows are
    # folded below the watermark), so only the new version can satisfy this.
    reader = next(
        (r for r in provider.requests[before:] if "one more planning question" in _body_of(r)),
        None,
    )
    assert reader is not None
    assert "Adopted a corgi named Biscuit" in _body_of(reader)


def test_a_schedule_compaction_never_wakes_the_generator() -> None:
    """The same refusal dance on the schedule channel folds a window and
    writes its checkpoint, but the generator gate (user channel only) holds."""
    provider = _ScriptedProvider(refuse_body_marker="follow-up status update")
    with patch(_BUILD_CLIENT, return_value=provider):
        first = MessageProcessor.process(ScheduledConfig(), "write a status update")
        first.result()
        _drain_background_turns()
        assert first.turn_id is not None
        second = MessageProcessor.process(
            ScheduledConfig(), "write the follow-up status update", turn_id=first.turn_id
        )
        second.result()
        _await_generator()
        _drain_background_turns()

    assert _terminal_state(second) == TurnExecution.COMPLETED
    # The compaction really happened — on whichever axis the schedule view
    # keys (fork when the second pass continues the turn, main otherwise).
    assert (
        Compaction.latest_main(Channel.SCHEDULE.value) is not None
        or Compaction.latest_fork(Channel.SCHEDULE.value, first.turn_id) is not None
    )
    assert provider.synthesis_requests == []
    assert UserSynthesisRow.latest() is None


def test_save_synthesis_contract() -> None:
    """Params-bag validation, versioned writes, and the INTERNAL policy pin."""
    assert SaveSynthesis.NAME in INTERNAL

    missing = SaveSynthesisParamsBag.from_params({})
    assert isinstance(missing, ToolResult)
    assert missing.status == "error"
    assert missing.code == "missing-params"
    assert UserSynthesisRow.latest() is None

    blank = SaveSynthesisParamsBag.from_params({Keys.summary: "   "})
    assert isinstance(blank, ToolResult)
    assert blank.code == "missing-params"

    first = SaveSynthesis().run(built(SaveSynthesisParamsBag.from_params({Keys.summary: "- Answers to Alex"})))
    assert first.status == "success"
    assert "version 1" in str(first.body)

    second = SaveSynthesis().run(
        built(SaveSynthesisParamsBag.from_params({Keys.summary: "- Answers to Alex\n- Lives in Valletta"}))
    )
    assert second.status == "success"
    assert "version 2" in str(second.body)

    row = UserSynthesisRow.latest()
    assert row is not None
    assert row.version == 2
    assert "Valletta" in row.content
    assert UserSynthesisRow.latest_content() == row.content


def test_a_no_op_exit_persists_nothing(db: sqlite3.Connection) -> None:
    """The prompt's "just exit" path: prose, no tool calls → COMPLETED turn,
    no synthesis row, no transcript residue on the generator channel."""
    provider = _ScriptedProvider(synthesis_script=[_Say("Synthesis unchanged — exiting.")])
    with patch(_BUILD_CLIENT, return_value=provider):
        UserSynthesisGenerator.instance().on_compaction(
            Channel.USER.value, "user said the sky was clear today"
        )
        _await_generator()
        _drain_background_turns()

    assert len(provider.synthesis_requests) == 1
    execution = TurnExecution.filter("channel", Channel.USER_SYNTHESIS.value).order_by("id DESC").first()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    assert UserSynthesisRow.latest() is None
    # skip_transcript holds: at most the input-anchor row, never the reply.
    rows = db.execute(
        "SELECT content FROM transcript WHERE channel = ?",
        (Channel.USER_SYNTHESIS.value,),
    ).fetchall()
    assert len(rows) <= 1
    assert all("Synthesis unchanged" not in (r["content"] or "") for r in rows)


def test_an_empty_completion_generator_turn_crashes_contained() -> None:
    """Truly-empty completions steer then crash the isolated generator turn;
    the parent user turn completes and no synthesis row lands."""
    provider = _ScriptedProvider(
        synthesis_script=[_Say("")],
        refuse_body_marker="carry on with the planning",
    )
    with patch(_BUILD_CLIENT, return_value=provider):
        first = MessageProcessor.process(UserConfig(), "I adopted a corgi named Biscuit")
        first.result()
        _drain_background_turns()
        second = MessageProcessor.process(UserConfig(), "carry on with the planning for the weekend")
        second.result()
        _await_generator()
        _drain_background_turns()

    assert _terminal_state(second) == TurnExecution.COMPLETED
    # The generator fired (steering re-sent at least once) and crashed alone.
    assert len(provider.synthesis_requests) >= 2
    execution = TurnExecution.filter("channel", Channel.USER_SYNTHESIS.value).order_by("id DESC").first()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert UserSynthesisRow.latest() is None
