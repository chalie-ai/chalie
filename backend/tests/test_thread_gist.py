"""Feature tests for thread-gist generation + encoding.

Zero mocks. Drives the real production writers (Transcript.write_input_row /
write_assistant_row), the real ThreadGistService, and the real conversation API
endpoints against the real test DB, asserting the cross-step contract:

1. The gist input is EXACTLY two non-assistant messages — the thread's opening
   message and the first message beyond ``settle0`` (the assistant settle row is
   never sent, nor any later reply).
2. A gist is written once per thread; the ``(channel, turn_id)`` primary key
   makes a repeat write idempotent.
3. The gist rides BOTH read endpoints: GET /api/thread/<turn_id> and the
   GET /api/threads feed.
4. The trigger boundary: a turn has no non-assistant row beyond ``settle0`` until
   a reply lands — the exact condition that makes a turn a thread.

The LLM-produced gist TEXT itself is not asserted here (it is non-deterministic
delegate output, covered by the nightly scenarios); these tests pin every
deterministic surface around it.
"""
import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from services.thread_gist_service import get_thread_gist_service
from services.transcript_service import Transcript


def _seed_thread_with_reply(opening: str, settle: str, reply: str) -> int:
    """A real thread: opening input → assistant settle0 (no tools) → reply past settle0."""
    input_id = Transcript.write_input_row('user', 'user', opening)
    turn_id = Transcript.turn_id_of_row(input_id)
    Transcript.write_assistant_row('user', settle, turn_id=turn_id)          # settle0
    Transcript.write_input_row('user', 'user', reply, turn_id=turn_id)        # beyond settle0
    return turn_id


@pytest.mark.unit
class TestThreadGist:

    def test_gist_prompt_is_exactly_two_non_assistant_messages(self, db: sqlite3.Connection) -> None:
        """get_user_prompt sends ONLY the opening + the first message beyond settle0 —
        never the assistant settle, never any later reply."""
        from configs.channels.thread_gist import ThreadGistConfig
        from services.message_processor import MessageProcessor

        turn_id = _seed_thread_with_reply(
            "OPENING researching a Mac Mini for my homelab",
            "ASSISTANTSETTLE here are the M2 vs M4 trade-offs",
            "FIRSTREPLY what about the M4 Pro power draw",
        )
        Transcript.write_input_row('user', 'user', "SECONDREPLY and the price", turn_id=turn_id)

        mp = object.__new__(MessageProcessor)
        setattr(mp, "_trigger_channel", 'user')
        setattr(mp, "_trigger_turn_id", turn_id)
        prompt = ThreadGistConfig().get_user_prompt(mp)

        assert "OPENING" in prompt
        assert "FIRSTREPLY" in prompt
        assert "ASSISTANTSETTLE" not in prompt, "the assistant settle row must never be sent"
        assert "SECONDREPLY" not in prompt, "only the FIRST message beyond settle0 is sent"
        message_lines = [ln for ln in prompt.splitlines() if ln.startswith("[")]
        assert len(message_lines) == 2, "exactly two messages go to the gist model"

    def test_gist_written_once_per_thread(self, db: sqlite3.Connection) -> None:
        """bulk_get is the once-per-thread gate; the (channel, turn_id) PK keeps a
        repeat write to a single row."""
        svc = get_thread_gist_service()
        turn_id = _seed_thread_with_reply("q", "a", "follow-up")

        assert svc.bulk_get('user', [turn_id]) == {}, "no gist before the delegate fires"
        svc.upsert('user', turn_id, "Mac Mini Research")
        assert svc.bulk_get('user', [turn_id]) == {turn_id: "Mac Mini Research"}

        svc.upsert('user', turn_id, "Different Label")  # would never happen behind the gate
        row = db.execute(
            "SELECT COUNT(*) FROM thread_gist WHERE channel = 'user' AND turn_id = ?",
            (turn_id,),
        ).fetchone()
        assert row[0] == 1, "one gist row per thread"
        assert svc.bulk_get('user', [turn_id]) == {turn_id: "Different Label"}

    def test_gist_rides_per_thread_endpoint(self, authed_client: tuple[object, sqlite3.Connection, object]) -> None:
        """GET /api/thread/<turn_id> carries the gist (None when absent)."""
        client = cast("FlaskClient", authed_client[0])
        turn_id = _seed_thread_with_reply(
            "Researching a Mac Mini purchase", "Here are the options", "What about the Pro?",
        )

        before = cast("dict[str, object]", client.get(f'/api/thread/{turn_id}').get_json())
        assert before['turn_id'] == turn_id
        assert before['gist'] is None, "gist key present and None before any gist is written"

        get_thread_gist_service().upsert('user', turn_id, "Mac Mini Research")
        after = cast("dict[str, object]", client.get(f'/api/thread/{turn_id}').get_json())
        assert after['gist'] == "Mac Mini Research"

    def test_gist_rides_thread_feed(self, authed_client: tuple[object, sqlite3.Connection, object]) -> None:
        """GET /api/threads attaches each thread's gist."""
        client = cast("FlaskClient", authed_client[0])
        turn_id = _seed_thread_with_reply(
            "Drafting an HN post", "Here's a structure", "Tighten the opening line?",
        )
        get_thread_gist_service().upsert('user', turn_id, "HN Post Feedback")

        feed = cast("dict[str, object]", client.get('/api/threads').get_json())
        threads = cast("list[dict[str, object]]", feed['threads'])
        mine = next(t for t in threads if t.get('turn_id') == turn_id)
        assert mine['gist'] == "HN Post Feedback"

    def test_trigger_boundary_no_message_beyond_settle0_until_reply(self, db: sqlite3.Connection) -> None:
        """A turn becomes a thread only once a non-assistant row exists beyond settle0.
        Before the reply that condition is False; the reply flips it True."""
        input_id = Transcript.write_input_row('user', 'user', "opening question")
        turn_id = Transcript.turn_id_of_row(input_id)
        Transcript.write_assistant_row('user', "the answer", turn_id=turn_id)

        settle = Transcript.settle0('user', turn_id)
        assert settle is not None, "the opening exchange has settled"

        def _has_beyond() -> bool:
            return any(
                r.get('role') != 'assistant' and cast("int", r.get('id')) > settle
                for r in Transcript.by_turn('user', turn_id)
            )

        assert _has_beyond() is False, "opening exchange alone → no thread"
        Transcript.write_input_row('user', 'user', "a follow-up", turn_id=turn_id)
        assert _has_beyond() is True, "reply beyond settle0 → thread (gist fires)"
