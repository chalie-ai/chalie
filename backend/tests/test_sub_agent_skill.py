"""Tests for sub_agent_skill — spawn a focused sub-agent for a specific task.

Note: TestHandleSubAgentHappyPath, TestHandleSubAgentOrchestration, and
TestHandleSubAgentErrors have been replaced with stub tests because
ACTOrchestrator has been deleted. sub_agent_skill returns a migration
stub for now. Tests will be updated when the skill is rewired to
MessageProcessor.send() tool loop.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.sub_agent_skill import (
    handle_sub_agent,
    TOOL_SCHEMA,
    _SUB_AGENT_SKILLS,
)


pytestmark = pytest.mark.unit


# ── Schema / constants ────────────────────────────────────────────────

class TestSubAgentSkillConstants:

    def test_tool_name(self):
        assert TOOL_SCHEMA['name'] == 'sub_agent'

    def test_goal_is_required(self):
        assert 'goal' in TOOL_SCHEMA['input_schema']['required']

    def test_sub_agent_skills_excludes_sub_agent(self):
        """Prevent recursive sub_agent spawning."""
        assert 'sub_agent' not in _SUB_AGENT_SKILLS

    def test_sub_agent_skills_excludes_persistent_task(self):
        """Prevent sub-agents from creating background tasks."""
        assert 'persistent_task' not in _SUB_AGENT_SKILLS

    def test_sub_agent_skills_includes_core_tools(self):
        """Sub-agents must have access to memory, find_tools, document, read."""
        for skill in ('memory', 'find_tools', 'document', 'read'):
            assert skill in _SUB_AGENT_SKILLS


# ── handle_sub_agent — parameter validation ───────────────────────────

class TestHandleSubAgentValidation:

    def test_missing_goal_returns_error(self):
        """Missing goal parameter returns an error string immediately."""
        result = handle_sub_agent('topic', {})

        assert 'error' in result.lower()
        assert 'goal' in result.lower()

    def test_empty_goal_returns_error(self):
        """Empty string goal returns an error string."""
        result = handle_sub_agent('topic', {'goal': '   '})

        assert 'error' in result.lower()
        assert 'goal' in result.lower()

    def test_none_goal_returns_error(self):
        """None goal returns an error string."""
        result = handle_sub_agent('topic', {'goal': None})

        assert 'error' in result.lower()


# ── handle_sub_agent — migration stub ────────────────────────────────
# TODO: replace with real tests when sub_agent is rewired to MessageProcessor

class TestHandleSubAgentStub:

    def test_returns_string_for_valid_goal(self):
        """Valid goal returns a string (migration stub message)."""
        result = handle_sub_agent('topic', {'goal': 'Find papers on LLMs'})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stub_message_is_informative(self):
        """Migration stub message mentions unavailability."""
        result = handle_sub_agent('topic', {'goal': 'Do research'})
        assert 'unavailable' in result.lower() or 'migrat' in result.lower()
