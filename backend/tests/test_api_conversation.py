import sqlite3

import pytest

from api.conversation import get_recent_history
from services.transcript_service import Transcript


def _seed_turn(question: str, *answers: str) -> int:
    """Open a real turn via the production writers and append its step rows.

    write_input_row allocates the turn boundary atomically; every assistant step
    shares that turn_id — the exact multi-row shape the chain model (TKT-1070)
    persists. Returns the turn_id so the test can assert grouping.
    """
    input_id = Transcript.write_input_row('user', 'user', question)
    turn_id = Transcript.turn_id_of_row(input_id)
    for ans in answers:
        Transcript.write_assistant_row('user', ans, turn_id=turn_id)
    return turn_id


@pytest.mark.unit
class TestChatHistory:

    def test_pagination_returns_whole_turns_not_rows(self, db: sqlite3.Connection) -> None:
        """Regression (feed history): a turn is many rows under the chain model,
        so the feed must paginate by TURN, never by row. The old row-LIMIT
        collapsed a page onto the latest turn's tail; here a 2-turn page must
        return every row of both turns, the offset must advance by turns, and
        each message must carry its turn_id."""
        t1 = _seed_turn('Q1', 'A1a', 'A1b')          # 1 input + 2 steps = 3 rows
        t2 = _seed_turn('Q2', 'A2')                   # 2 rows
        t3 = _seed_turn('Q3', 'A3a', 'A3b', 'A3c')   # 4 rows
        assert t1 < t2 < t3

        # Page 1: the two newest WHOLE turns (t2 then t3), oldest-first within each.
        page1, has_more, turns_returned = get_recent_history(limit=2, offset=0)
        assert turns_returned == 2
        assert has_more is True                       # t1 still unpaged
        assert [m["content"] for m in page1] == ['Q2', 'A2', 'Q3', 'A3a', 'A3b', 'A3c']
        # Every row of t3 survived — the regression dropped all but the last rows.
        assert [m["turn_id"] for m in page1] == [t2, t2, t3, t3, t3, t3]

        # Page 2: advancing by turns_returned reaches the oldest turn whole.
        page2, has_more, turns_returned = get_recent_history(limit=2, offset=2)
        assert turns_returned == 1
        assert has_more is False
        assert [m["content"] for m in page2] == ['Q1', 'A1a', 'A1b']
        assert {m["turn_id"] for m in page2} == {t1}

    def test_boot_and_scroll(self, db: sqlite3.Connection) -> None:
        for i in range(30):
            role = 'user' if i % 2 == 0 else 'assistant'
            db.execute(
                "INSERT INTO transcript (channel, role, content, created_at) "
                "VALUES ('user', ?, ?, ?)",
                (role, f'Message {i}', f'2026-01-01T00:{i:02d}:00+00:00'),
            )
        db.commit()

        # Boot — last 12 messages
        messages, has_more, _ = get_recent_history(limit=12, offset=0)
        assert len(messages) == 12
        assert has_more is True
        assert messages[0]["content"] == "Message 18"
        assert messages[-1]["content"] == "Message 29"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Scroll up — next 12 older messages
        messages, has_more, _ = get_recent_history(limit=12, offset=12)
        assert len(messages) == 12
        assert has_more is True
        assert messages[0]["content"] == "Message 6"
        assert messages[-1]["content"] == "Message 17"

    def test_subagent_return_role_hidden_from_recent_history(self, db: sqlite3.Connection) -> None:
        """subagent_return rows must be hidden from user-visible chat history.

        Plan §Q-spec-1: 'subagent_return' role is internal async-delivery — only the
        synthesised response reaches the chat, not the raw subagent envelope.
        """
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', 'hello', '2026-01-01T00:01:00+00:00')",
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'assistant', 'hi', '2026-01-01T00:02:00+00:00')",
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'subagent_return', "
            "'[subagent.complete(type=web_surfer)]\\nresult\\n[end:subagent.complete]', "
            "'2026-01-01T00:03:00+00:00')",
        )
        db.commit()

        messages, _, _ = get_recent_history(limit=50, offset=0)

        roles = [m["role"] for m in messages]
        assert "subagent_return" not in roles, (
            f"subagent_return row leaked into chat history: {roles}"
        )
        assert "user" in roles, "user row missing from history"
        assert "assistant" in roles, "assistant row missing from history"
        assert len(messages) == 2, (
            f"Expected 2 rows (user + assistant), got {len(messages)}: {roles}"
        )
