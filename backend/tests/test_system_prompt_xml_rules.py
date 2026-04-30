import pytest

from services.system_message_prompt import UnifiedSystemMessagePrompt


@pytest.mark.unit
class TestSystemPromptXmlRules:
    def test_prompt_mentions_response_format(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        assert "Response format" in prompt or "response format" in prompt.lower()

    def test_prompt_says_html(self):
        # New whitelist phrasing: "format your response as HTML" — keep the
        # word HTML present so the model is told the output language.
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        assert "HTML" in prompt

    def test_prompt_lists_all_eight_llm_tags(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        for tag in ["<b>", "<i>", "<u>", "<h1>", "<code>", "<p>", "<ul>", "<li>"]:
            assert tag in prompt, f"Missing tag in prompt: {tag}"

    def test_prompt_does_not_list_anchor_tag(self):
        """``<a>`` is not in the LLM allowlist. The whitelist approach
        guarantees this implicitly: the model only sees the allowed tag list,
        and ``<a>`` is absent. Backend nh3 sanitisation strips any anchor
        that slips through anyway."""
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        assert "<a>" not in prompt

    def test_prompt_does_not_mention_programmatic_tags(self):
        # The harness emits these; the LLM must never be primed to do so.
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        for forbidden in ["<actions", "<action "]:
            assert forbidden not in prompt, f"Programmatic tag leaked into prompt: {forbidden}"
