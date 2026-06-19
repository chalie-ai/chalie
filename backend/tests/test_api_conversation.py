import sqlite3

import pytest

from api.conversation import get_recent_history


@pytest.mark.unit
class TestChatHistory:

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
        messages, has_more = get_recent_history(limit=12, offset=0)
        assert len(messages) == 12
        assert has_more is True
        assert messages[0]["content"] == "Message 18"
        assert messages[-1]["content"] == "Message 29"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Scroll up — next 12 older messages
        messages, has_more = get_recent_history(limit=12, offset=12)
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

        messages, _ = get_recent_history(limit=50, offset=0)

        roles = [m["role"] for m in messages]
        assert "subagent_return" not in roles, (
            f"subagent_return row leaked into chat history: {roles}"
        )
        assert "user" in roles, "user row missing from history"
        assert "assistant" in roles, "assistant row missing from history"
        assert len(messages) == 2, (
            f"Expected 2 rows (user + assistant), got {len(messages)}: {roles}"
        )
