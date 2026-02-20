"""Tests for WorkingMemoryService — FIFO buffer with fakeredis."""

import pytest
from services.working_memory_service import WorkingMemoryService


pytestmark = pytest.mark.unit


class TestWorkingMemory:

    def test_append_and_retrieve_turn(self, mock_redis):
        svc = WorkingMemoryService(max_turns=4)
        svc.append_turn("topic-a", "user", "Hello")
        svc.append_turn("topic-a", "assistant", "Hi there")

        turns = svc.get_recent_turns("topic-a")
        assert len(turns) == 2
        assert turns[0]['role'] == 'user'
        assert turns[0]['content'] == 'Hello'
        assert turns[1]['role'] == 'assistant'
        assert turns[1]['content'] == 'Hi there'

    def test_fifo_eviction_at_max_turns(self, mock_redis):
        """5th turn pair evicts 1st (max_entries=8, 2 per turn)."""
        svc = WorkingMemoryService(max_turns=4)
        # Add 5 turn pairs (10 entries, but max is 8)
        for i in range(5):
            svc.append_turn("topic-a", "user", f"q{i}")
            svc.append_turn("topic-a", "assistant", f"a{i}")

        turns = svc.get_recent_turns("topic-a")
        assert len(turns) == 8  # max_turns * 2
        # First turn should be evicted — oldest remaining is q1
        assert turns[0]['content'] == 'q1'

    def test_formatted_context_output(self, mock_redis):
        """get_formatted_context returns 'Human: ... / Assistant: ...' format."""
        svc = WorkingMemoryService(max_turns=4)
        svc.append_turn("topic-a", "user", "What is Python?")
        svc.append_turn("topic-a", "assistant", "A programming language.")

        context = svc.get_formatted_context("topic-a")
        assert "## Recent Conversation" in context
        assert "User: What is Python?" in context
        assert "Assistant: A programming language." in context

    def test_clear_removes_all(self, mock_redis):
        svc = WorkingMemoryService(max_turns=4)
        svc.append_turn("topic-a", "user", "Hello")
        svc.clear("topic-a")

        turns = svc.get_recent_turns("topic-a")
        assert len(turns) == 0

    def test_buffer_size_tracking(self, mock_redis):
        svc = WorkingMemoryService(max_turns=4)
        assert svc.get_buffer_size("topic-a") == 0

        svc.append_turn("topic-a", "user", "Hello")
        assert svc.get_buffer_size("topic-a") == 1

        svc.append_turn("topic-a", "assistant", "Hi")
        assert svc.get_buffer_size("topic-a") == 2
