"""Test chat history retrieval — calls the same function the API uses."""

import pytest

from api.conversation import get_recent_history


@pytest.mark.unit
class TestChatHistory:

    def test_boot_and_scroll(self, db):
        """Seed 30 messages, verify boot=12 and scroll-up=next 12."""
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
