"""Unit tests for services/innate_skills/_tag.py::tag().

Pure-function: no IO, no collaborators, no state.
"""

import pytest
from services.innate_skills._tag import tag

pytestmark = pytest.mark.unit


class TestTagFormatter:
    """tag() produces the canonical [name(...)] body [end:name] format."""

    def test_empty_content_omits_body_line(self):
        """Empty content produces two-line output: opener + terminator only."""
        result = tag("read")
        assert result == "[read()]\n[end:read]"

    def test_none_content_omits_body_line(self):
        """None content is treated the same as empty — no body line."""
        result = tag("read", None)
        assert result == "[read()]\n[end:read]"

    def test_content_with_newlines_is_preserved(self):
        """Multi-line content appears verbatim between markers."""
        body = "line one\nline two\nline three"
        result = tag("memory", body, action="recall")
        assert result == f"[memory(action=recall)]\n{body}\n[end:memory]"

    def test_error_only_args_no_body(self):
        """Error args live on the opener; body is omitted when content is empty."""
        result = tag("memory", error="key-required", action="store")
        assert result == "[memory(error=key-required, action=store)]\n[end:memory]"
        # No blank line between opener and terminator
        assert "\n\n" not in result

    def test_mixed_args_and_content(self):
        """Args appear in opener; content appears between opener and terminator."""
        result = tag("list", '{"status": "success"}', action="add")
        assert result.startswith("[list(action=add)]\n")
        assert result.endswith("\n[end:list]")
        assert '{"status": "success"}' in result

    def test_arg_insertion_order_preserved(self):
        """kwargs appear in the opener in insertion order (Python 3.7+ dict order)."""
        result = tag("schedule", action="create", status="success", found=3)
        opener_line = result.split("\n")[0]
        assert opener_line == "[schedule(action=create, status=success, found=3)]"

    def test_arg_values_are_str_coerced(self):
        """Non-string arg values are coerced via str()."""
        result = tag("find_tools", query="weather", found=5)
        assert "found=5" in result
        assert "query=weather" in result

    def test_single_content_line(self):
        """Single-line content produces three-line output."""
        result = tag("list", "Groceries: Milk, Eggs", action="view")
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "[list(action=view)]"
        assert lines[1] == "Groceries: Milk, Eggs"
        assert lines[2] == "[end:list]"

    def test_name_appears_in_both_opener_and_terminator(self):
        """The tag name must match exactly in both the opener and terminator."""
        result = tag("goal_pursuit", "body here", pursuit_id="abc")
        assert result.startswith("[goal_pursuit(")
        assert result.endswith("[end:goal_pursuit]")

    def test_no_args_produces_empty_parens(self):
        """When no kwargs are supplied, the opener is [name()]."""
        result = tag("read", "content")
        assert result.startswith("[read()]")
