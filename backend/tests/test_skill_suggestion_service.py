"""Feature tests for TKT-597: skill_suggestion_service pure functions.

All functions under test are pure: deterministic input → output,
no IO, no collaborators, no LLM calls, no database. Zero mocks.

Behaviours covered:

1. _extract_tool_name   — bracket-prefix parsing, edge cases
2. _extract_result_snippet — result extraction and 200-char cap
3. _summarise_trail     — compression of act_trail list
4. _parse_analysis      — JSON parsing, code-fence stripping, SKIP semantics
5. _build_suggestion_message — output format and key content
6. _build_analysis_prompt — prompt structure contains the right sections
"""

import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. _extract_tool_name
# ---------------------------------------------------------------------------


class TestExtractToolName:
    """Parses the tool name from a '[tool_name(…)] result' act_trail entry."""

    def test_standard_bracket_format_returns_tool_name(self):
        from services.skill_suggestion_service import _extract_tool_name

        entry = "[search(query='Malta weather')] Result text here"
        assert _extract_tool_name(entry) == "search"

    def test_tool_name_with_no_args_bracket_only(self):
        from services.skill_suggestion_service import _extract_tool_name

        entry = "[memory] Some result"
        assert _extract_tool_name(entry) == "memory"

    def test_entry_without_leading_bracket_returns_unknown(self):
        from services.skill_suggestion_service import _extract_tool_name

        entry = "No bracket here — just plain text"
        assert _extract_tool_name(entry) == "unknown"

    def test_empty_string_returns_unknown(self):
        from services.skill_suggestion_service import _extract_tool_name

        assert _extract_tool_name("") == "unknown"


# ---------------------------------------------------------------------------
# 2. _extract_result_snippet
# ---------------------------------------------------------------------------


class TestExtractResultSnippet:
    """Extracts result text after the closing bracket, capped at 200 chars."""

    def test_result_extracted_after_closing_bracket(self):
        from services.skill_suggestion_service import _extract_result_snippet

        entry = "[search(q='foo')] Here is the result"
        snippet = _extract_result_snippet(entry)
        assert snippet == "Here is the result"

    def test_result_capped_at_200_chars(self):
        from services.skill_suggestion_service import _extract_result_snippet

        long_result = "x" * 500
        entry = f"[tool()] {long_result}"
        snippet = _extract_result_snippet(entry)
        assert len(snippet) == 200

    def test_entry_with_no_bracket_returns_full_entry_capped(self):
        from services.skill_suggestion_service import _extract_result_snippet

        text = "plain entry with no bracket at all"
        snippet = _extract_result_snippet(text)
        # No bracket found — falls back to whole string, capped
        assert snippet == text[:200]


# ---------------------------------------------------------------------------
# 3. _summarise_trail
# ---------------------------------------------------------------------------


class TestSummariseTrail:
    """Compresses a list of act_trail entries into (tool, result) dicts."""

    def test_each_entry_produces_one_dict(self):
        from services.skill_suggestion_service import _summarise_trail

        trail = [
            "[search(q='news')] Some news results here",
            "[memory(action='recall')] Recalled three facts",
        ]
        summary = _summarise_trail(trail)
        assert len(summary) == 2

    def test_each_dict_has_tool_and_result_keys(self):
        from services.skill_suggestion_service import _summarise_trail

        trail = ["[weather(loc='London')] Sunny, 22°C"]
        summary = _summarise_trail(trail)
        assert set(summary[0].keys()) == {"tool", "result"}

    def test_tool_name_parsed_correctly_in_summary(self):
        from services.skill_suggestion_service import _summarise_trail

        trail = ["[news(topic='tech')] Top stories today"]
        summary = _summarise_trail(trail)
        assert summary[0]["tool"] == "news"

    def test_empty_trail_returns_empty_list(self):
        from services.skill_suggestion_service import _summarise_trail

        assert _summarise_trail([]) == []


# ---------------------------------------------------------------------------
# 4. _parse_analysis
# ---------------------------------------------------------------------------


class TestParseAnalysis:
    """JSON parser for the LLM analysis response. Returns dict or None."""

    def test_reusable_true_with_all_fields_returns_dict(self):
        from services.skill_suggestion_service import _parse_analysis

        payload = json.dumps({
            "reusable": True,
            "title": "Research and summarise topic",
            "use_for": "When the user asks to research any topic",
            "tags": "research,summary",
            "content": "1. Search\n2. Summarise",
        })
        result = _parse_analysis(payload)
        assert result is not None
        assert result["title"] == "Research and summarise topic"

    def test_reusable_false_returns_none(self):
        from services.skill_suggestion_service import _parse_analysis

        payload = json.dumps({"reusable": False})
        assert _parse_analysis(payload) is None

    def test_code_fence_stripped_before_parsing(self):
        from services.skill_suggestion_service import _parse_analysis

        fenced = (
            "```json\n"
            '{"reusable": true, "title": "Do thing", '
            '"use_for": "When you need it", "content": "1. Step one"}\n'
            "```"
        )
        result = _parse_analysis(fenced)
        assert result is not None
        assert result["title"] == "Do thing"

    def test_malformed_json_returns_none(self):
        from services.skill_suggestion_service import _parse_analysis

        assert _parse_analysis("{not: valid json}") is None

    def test_empty_string_returns_none(self):
        from services.skill_suggestion_service import _parse_analysis

        assert _parse_analysis("") is None

    def test_reusable_true_but_missing_required_field_returns_none(self):
        from services.skill_suggestion_service import _parse_analysis

        # 'content' is missing — must be rejected even though reusable=true
        payload = json.dumps({
            "reusable": True,
            "title": "Do a thing",
            "use_for": "When needed",
        })
        assert _parse_analysis(payload) is None


# ---------------------------------------------------------------------------
# 5. _build_suggestion_message
# ---------------------------------------------------------------------------


class TestBuildSuggestionMessage:
    """Builds the natural-language message delivered to the user via UMP."""

    def test_output_contains_skill_title(self):
        from services.skill_suggestion_service import _build_suggestion_message

        msg = _build_suggestion_message(
            title="Research and summarise",
            use_for="When the user asks to research anything",
            content="1. Search\n2. Summarise",
            iteration_count=5,
        )
        assert "Research and summarise" in msg

    def test_output_contains_iteration_count(self):
        from services.skill_suggestion_service import _build_suggestion_message

        msg = _build_suggestion_message(
            title="Do a thing",
            use_for="When needed",
            content="1. Step",
            iteration_count=7,
        )
        assert "7" in msg

    def test_output_contains_save_prompt(self):
        from services.skill_suggestion_service import _build_suggestion_message

        msg = _build_suggestion_message(
            title="Any title",
            use_for="Any use",
            content="1. Step",
            iteration_count=4,
        )
        # Must invite the user to save — "yes" cue triggers approval flow
        assert "yes" in msg.lower()


# ---------------------------------------------------------------------------
# 6. _build_analysis_prompt
# ---------------------------------------------------------------------------


class TestBuildAnalysisPrompt:
    """Prompt builder from user query + summarised trail."""

    def test_prompt_contains_user_request_section(self):
        from services.skill_suggestion_service import _build_analysis_prompt

        summary = [{"tool": "search", "result": "Some results"}]
        prompt = _build_analysis_prompt("find me tech news", summary)
        assert "find me tech news" in prompt

    def test_prompt_contains_tool_execution_trail(self):
        from services.skill_suggestion_service import _build_analysis_prompt

        summary = [{"tool": "weather", "result": "sunny skies"}]
        prompt = _build_analysis_prompt("what is the weather", summary)
        # Trail must appear as JSON so the LLM can read tool names + snippets
        assert "weather" in prompt
        assert "sunny skies" in prompt
