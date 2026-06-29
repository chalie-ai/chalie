"""Feature tests for the thread context & compaction model.

Zero mocks. Every test drives the real context-assembly hot path against the
migrated test DB using the SAME factories production uses: the transcript
writers (``write_input_row`` / ``write_assistant_row``), the tool-call writer
(``ActTrail.record``), the compaction writer
(``compaction_persistence.write_compaction``), the real read+render seam
(``MessageProcessor.get_previous_messages`` → ``_previous_rows``), the real
checkpoint-envelope seam (``_wrap_with_checkpoint``), the real cancellation
cleanup (``_cleanup_cancelled``) and the real production config (``UserConfig``).

The model has two views, both keyed on ``turn_id``:

* MAIN spine (a fresh message): every settled turn's opening exchange (rows
  ``id <= settle0`` — a turn's first assistant row carrying no model-driven
  tool), across turns, cut on the *turn_id* axis by the MAIN watermark.
* THREAD read (a genuine reply into a thread): that turn's WHOLE row set — the
  opening user message, pre-settle interims, settle0 and every continuation —
  cut on the *transcript.id* axis by the THREAD watermark, dropping the live
  reply. No settle0 floor.

Both watermark axes live in the dedicated ``compactions`` table and never
collide: a MAIN compaction can't hide a thread reply, a THREAD compaction can't
hide the spine.

Coverage boundary: the compactor's ``run()`` fires a real LLM to author the
summary text, so summary *generation* is out of scope for these deterministic
tests. These tests pin the deterministic half — the watermark cut and the
envelope — by driving ``write_compaction`` (the exact persistence call ``run()``
makes once the LLM returns) and the real read seams.
"""
from typing import cast

import pytest

from api.threads import serialize_turn
from configs.channels import UserConfig
from services import compaction_persistence
from services.act_trail import ActTrail
from services.message_processor import MessageProcessor, _wrap_with_checkpoint
from services.transcript_service import Transcript

pytestmark = pytest.mark.unit

_CH = "user"


def _settled_turn(question: str, *, answer: str,
                  steps: tuple[tuple[str, str], ...] = ()) -> tuple[int, int, int]:
    """Seed one settled turn; return ``(turn_id, input_id, settle_id)``.

    ``steps`` are pre-settle tool-bearing assistant rows — each ``(text, tool)``
    records a real (non-internal) tool call, which demotes the row below
    settle0. ``answer`` is the final no-tool assistant row = the turn's settle0.
    """
    input_id = Transcript.write_input_row(_CH, "user", question)
    turn_id = Transcript.turn_id_of_row(input_id)
    for text, tool in steps:
        step_id = Transcript.write_assistant_row(_CH, text, turn_id=turn_id)
        ActTrail().record(tool_name=tool, params={}, result="ok", transcript_id=step_id)
    settle_id = Transcript.write_assistant_row(_CH, answer, turn_id=turn_id)
    return turn_id, input_id, settle_id


def _reply(turn_id: int, question: str, *, answer: str) -> tuple[int, int]:
    """Append a settled reply continuation into an existing thread; return
    ``(reply_input_id, reply_settle_id)`` — both above the thread's settle0."""
    reply_in = Transcript.write_input_row(_CH, "user", question, turn_id=turn_id)
    reply_settle = Transcript.write_assistant_row(_CH, answer, turn_id=turn_id)
    return reply_in, reply_settle


def _main_mp() -> MessageProcessor:
    """A fresh-message processor — MAIN view (``_forked`` False, ``turn_id`` None)."""
    mp = MessageProcessor("a fresh message", None)
    mp.config = UserConfig()
    return mp


def _fork_mp(turn_id: int, live_input_id: int) -> MessageProcessor:
    """A genuine reply into ``turn_id`` — FORK view, with the live reply's own
    input id as the upper bound (``self.uid``) so it renders separately."""
    mp = MessageProcessor("a reply into the thread", {"is_thread_reply": True, "thread_id": turn_id})
    mp.config = UserConfig()
    mp.uid = live_input_id
    return mp


class TestMainSpine:

    def test_spine_emits_opening_exchanges_and_excludes_fork_continuations(self, db: object) -> None:
        """The MAIN spine renders each settled turn's opening exchange (input +
        pre-settle steps + settle0) across turns, and excludes rows past settle0
        — a fork continuation never leaks onto the spine."""
        _settled_turn("first question", answer="first answer",
                      steps=(("let me look that up", "search_files"),))
        t2, _, _ = _settled_turn("second question", answer="second answer")
        _reply(t2, "a later reply into turn two", answer="the reply answer")

        rendered = _main_mp().get_previous_messages()

        assert "first question" in rendered
        assert "let me look that up" in rendered, "pre-settle step is part of the opening exchange"
        assert "first answer" in rendered
        assert "second question" in rendered
        assert "second answer" in rendered
        assert "a later reply into turn two" not in rendered, "fork continuation must not enter the spine"
        assert "the reply answer" not in rendered

    def test_compaction_drops_absorbed_turns_and_is_channel_isolated(self, db: object) -> None:
        """A MAIN compaction advances the turn-axis watermark: turns at/below it
        are absorbed (dropped from the spine), later turns remain. Another
        channel's rows + compaction never enter the user spine."""
        _settled_turn("question one", answer="answer one")
        t2, _, _ = _settled_turn("question two", answer="answer two")
        _settled_turn("question three", answer="answer three")
        # Noise on another channel — must never reach the user spine or its watermark.
        other_in = Transcript.write_input_row("dmn", "user", "other channel question")
        other_t = Transcript.turn_id_of_row(other_in)
        Transcript.write_assistant_row("dmn", "other channel answer", turn_id=other_t)
        compaction_persistence.write_compaction("dmn", None, other_t, "OTHER-CKPT")

        before = _main_mp().get_previous_messages()
        assert "question one" in before and "question three" in before

        compaction_persistence.write_compaction(_CH, None, t2, "USER-CKPT")
        after = _main_mp().get_previous_messages()

        assert "question one" not in after, "turn 1 absorbed by the watermark"
        assert "question two" not in after, "turn 2 absorbed (watermark inclusive on the turn axis)"
        assert "question three" in after, "turn 3 is past the watermark — still on the spine"
        assert "other channel question" not in after, "another channel never enters the user spine"

    def test_cold_spine_renders_every_un_checkpointed_turn(self, db: object) -> None:
        """The watermark is the only spine bound: with no checkpoint yet, every
        settled turn renders — nothing has been folded into a summary, so nothing
        may be silently dropped. Compaction fires on the 90% context trip and
        advances the watermark, which is what then trims the tail."""
        for i in range(15):
            _settled_turn(f"question {i:02d}", answer=f"answer {i:02d}")

        rendered = _main_mp().get_previous_messages()

        assert "question 00" in rendered, "the oldest un-checkpointed turn still renders"
        assert "question 07" in rendered, "a middle turn renders"
        assert "question 14" in rendered, "the newest turn renders"


class TestForkRead:

    def test_thread_read_is_the_whole_turn_minus_the_live_reply(self, db: object) -> None:
        """The THREAD read is the entire turn — opening user message, pre-settle
        interims, settle0 and every settled continuation — with no settle0 floor;
        only the live reply's own rows are excluded (they render separately)."""
        t, _, _ = _settled_turn("thread question", answer="thread answer",
                                steps=(("thinking it through", "search_files"),))
        _reply(t, "first follow-up", answer="first follow-up answer")
        live_in = Transcript.write_input_row(_CH, "user", "the live reply", turn_id=t)

        rendered = _fork_mp(t, live_in).get_previous_messages()

        assert "thread question" in rendered, "the opening user message is part of the whole turn"
        assert "thinking it through" in rendered, "pre-settle interim is part of the whole turn"
        assert "thread answer" in rendered, "settle0 is included"
        assert "first follow-up" in rendered, "prior settled continuation is included"
        assert "first follow-up answer" in rendered
        assert "the live reply" not in rendered, "the live reply renders separately, not in history"

    def test_watermark_cuts_fork_read_without_touching_the_spine(self, db: object) -> None:
        """A FORK compaction (transcript.id axis) drops the continuation up to its
        watermark for the fork read, but does NOT advance the spine's turn-axis
        watermark — the spine still shows both turns' opening exchanges."""
        _settled_turn("alpha question", answer="alpha answer")
        t2, _, _ = _settled_turn("beta question", answer="beta answer")
        _, r_settle = _reply(t2, "beta follow-up", answer="beta follow-up answer")
        _reply(t2, "beta second follow-up", answer="beta second answer")
        live_in = Transcript.write_input_row(_CH, "user", "beta live reply", turn_id=t2)

        compaction_persistence.write_compaction(_CH, t2, r_settle, "FORK checkpoint")

        fork_rendered = _fork_mp(t2, live_in).get_previous_messages()
        assert "beta second follow-up" in fork_rendered, "rows above the fork watermark survive"
        assert "beta second answer" in fork_rendered
        assert "beta follow-up answer" not in fork_rendered, "row at the fork watermark is folded"
        assert "beta answer" not in fork_rendered, "settle0 anchor below the watermark is folded"

        main_rendered = _main_mp().get_previous_messages()
        assert "alpha question" in main_rendered, "fork checkpoint did not move the spine watermark"
        assert "beta question" in main_rendered


class TestTwoAxisIndependence:

    def test_main_watermark_hides_turn_from_spine_but_not_from_its_fork_read(self, db: object) -> None:
        """A MAIN compaction absorbs a turn from the spine (turn-axis) yet the SAME
        turn read as a FORK is untouched — its own id-axis watermark is 0, so the
        anchor + continuation are all there. The two axes do not collide."""
        _settled_turn("turn one question", answer="turn one answer")
        t2, _, _ = _settled_turn("turn two question", answer="turn two answer")
        _reply(t2, "turn two follow-up", answer="turn two follow-up answer")
        live_in = Transcript.write_input_row(_CH, "user", "live reply into turn two", turn_id=t2)
        _settled_turn("turn three question", answer="turn three answer")

        compaction_persistence.write_compaction(_CH, None, t2, "MAIN checkpoint")

        main_rendered = _main_mp().get_previous_messages()
        assert "turn three question" in main_rendered
        assert "turn one question" not in main_rendered
        assert "turn two question" not in main_rendered, "turn 2 absorbed from the spine"

        fork_rendered = _fork_mp(t2, live_in).get_previous_messages()
        assert "turn two answer" in fork_rendered, "settle0 anchor survives — fork axis is independent"
        assert "turn two follow-up" in fork_rendered, "continuation survives the MAIN watermark"
        assert "turn two follow-up answer" in fork_rendered

    def test_checkpoint_envelope_reads_the_views_own_axis(self, db: object) -> None:
        """``_wrap_with_checkpoint`` wraps with the compaction on the SAME axis the
        view reads: ``None`` (spine) sees the MAIN checkpoint, ``turn_id`` (fork)
        sees the FORK checkpoint — never each other's."""
        t, _, settle = _settled_turn("envelope question", answer="envelope answer")
        compaction_persistence.write_compaction(_CH, None, t, "MAIN-CKPT-TEXT")
        compaction_persistence.write_compaction(_CH, t, settle, "FORK-CKPT-TEXT")

        main_wrapped = _wrap_with_checkpoint(_CH, "the current body", None)
        fork_wrapped = _wrap_with_checkpoint(_CH, "the current body", t)

        assert "### Checkpoint" in main_wrapped
        assert "MAIN-CKPT-TEXT" in main_wrapped
        assert "FORK-CKPT-TEXT" not in main_wrapped, "MAIN envelope must read only the for_turn_id IS NULL axis"

        assert "FORK-CKPT-TEXT" in fork_wrapped
        assert "MAIN-CKPT-TEXT" not in fork_wrapped, "FORK envelope must read only its own thread axis"
        assert "the current body" in fork_wrapped


class TestForkReplyCancellation:

    def test_cancel_purges_only_its_own_rows_by_id_floor(self, db: object) -> None:
        """A cancelled fork reply purges only ITS own rows (``id >= self.uid``) —
        the pre-existing thread it replied into survives untouched."""
        t, _, _ = _settled_turn("pre-existing question", answer="pre-existing answer")
        reply_in = Transcript.write_input_row(_CH, "user", "the cancelled reply", turn_id=t)
        Transcript.write_assistant_row(_CH, "half-written response", turn_id=t)

        _fork_mp(t, reply_in)._cleanup_cancelled()

        surviving = [cast("str", r["content"]) for r in Transcript.by_turn(_CH, t)]
        assert "pre-existing question" in surviving, "the thread predating the reply must survive"
        assert "pre-existing answer" in surviving
        assert "the cancelled reply" not in surviving, "the cancelled reply's own input row is purged"
        assert "half-written response" not in surviving, "the cancelled reply's own step is purged"


class TestSerializeTurnWatermark:

    def test_serialize_turn_returns_full_turn_ignoring_compaction(self, db: object) -> None:
        """serialize_turn (REST GET /api/thread/<id>) returns EVERY row of the turn
        regardless of any thread compaction checkpoint — the REST feed has full
        fidelity; the watermark only governs the LLM context window (FORK read)."""
        t, _, _ = _settled_turn("opening question", answer="opening answer")
        _, r_settle = _reply(t, "first follow-up", answer="first follow-up answer")
        Transcript.write_input_row(_CH, "user", "second follow-up", turn_id=t)
        Transcript.write_assistant_row(_CH, "second follow-up answer", turn_id=t)

        # Write a thread compaction checkpoint — REST must ignore it.
        compaction_persistence.write_compaction(_CH, t, r_settle, "THREAD-CKPT")

        result = serialize_turn(_CH, t)
        messages = cast("list[dict[str, object]]", result["messages"])
        contents = [cast("str", m["content"]) for m in messages]
        assert "opening question" in contents, "rows below the watermark must still be present"
        assert "opening answer" in contents
        assert "first follow-up answer" in contents, "row at the watermark must still be present"
        assert "second follow-up" in contents, "rows above the watermark must be present"
        assert "second follow-up answer" in contents
