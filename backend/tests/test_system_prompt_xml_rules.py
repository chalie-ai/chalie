import pytest

from services.system_message_prompt import UnifiedSystemMessagePrompt


@pytest.mark.unit
class TestSystemPromptXmlRules:
    def test_prompt_mentions_xml_format(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        assert "RESPONSE FORMAT" in prompt or "Response format" in prompt

    def test_prompt_lists_all_eight_llm_tags(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        # ``<a>`` was removed from the LLM allowlist so the model cannot emit
        # arbitrary anchors. Plain-text URLs are auto-linkified by the FE.
        for tag in ["<b>", "<i>", "<u>", "<h1>", "<code>", "<p>", "<ul>", "<li>"]:
            assert tag in prompt, f"Missing tag in prompt: {tag}"

    def test_prompt_forbids_markdown(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT.lower()
        assert "do not use markdown" in prompt or "no markdown" in prompt

    def test_prompt_forbids_anchor_tag(self):
        """``<a>`` is no longer in the LLM allowlist; backend strips it on
        sanitisation. The system prompt must instruct the model accordingly
        so it does not waste tokens emitting tags that get dropped."""
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        assert "Do NOT emit <a>" in prompt or "auto-linkified" in prompt

    def test_prompt_does_not_mention_programmatic_tags(self):
        # The harness emits these; the LLM must never be primed to do so.
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        for forbidden in ["<actions", "<action "]:
            assert forbidden not in prompt, f"Programmatic tag leaked into prompt: {forbidden}"
