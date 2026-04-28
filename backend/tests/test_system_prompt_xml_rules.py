import pytest

from services.system_message_prompt import UnifiedSystemMessagePrompt


@pytest.mark.unit
class TestSystemPromptXmlRules:
    def test_prompt_mentions_xml_format(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        assert "RESPONSE FORMAT" in prompt or "Response format" in prompt

    def test_prompt_lists_all_eight_llm_tags(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        for tag in ["<b>", "<i>", "<u>", "<h1>", "<code>", "<p>", "<ul>", "<li>", "<a "]:
            assert tag in prompt, f"Missing tag in prompt: {tag}"

    def test_prompt_forbids_markdown(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT.lower()
        assert "do not use markdown" in prompt or "no markdown" in prompt

    def test_prompt_does_not_mention_forbidden_tags(self):
        prompt = UnifiedSystemMessagePrompt._SYSTEM_PROMPT
        for forbidden in ["<img", "<actions", "<action "]:
            assert forbidden not in prompt, f"Programmatic tag leaked into prompt: {forbidden}"
