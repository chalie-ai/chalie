"""Unit tests for _extract_inline_tool_calls and its integration with
_parse_chat_response.

Covers Qwen-style <tool_call>...</tool_call> markup extraction when the
structured tool_calls channel is empty or null.
"""

import pytest

from abilities._registry import AbilityRegistry

pytestmark = pytest.mark.unit


class TestExtractInlineToolCalls:
    def _extract(self, text):
        from services.ollama_service import _extract_inline_tool_calls
        return _extract_inline_tool_calls(text)

    def _real_ability(self):
        """Return the NAME of any registered ability to use as a real name."""
        abilities = AbilityRegistry.all()
        assert abilities, "Registry must contain at least one ability"
        return abilities[0].NAME

    def test_path_a_explicit_name_with_arguments(self):
        """Path A: dict has 'name' key recognised by registry + 'arguments' dict."""
        name = self._real_ability()
        text = f'<tool_call>{{"name": "{name}", "arguments": {{"source": "https://example.com"}}}}</tool_call>'
        cleaned, calls = self._extract(text)
        assert len(calls) == 1
        assert calls[0]['name'] == name
        assert calls[0]['input'] == {'source': 'https://example.com'}
        assert '<tool_call>' not in cleaned

    def test_path_b_action_as_name(self):
        """Path B — 039 case: dict has 'action' key whose value is a registered name.
        The 'action' key must NOT appear in the extracted input."""
        name = self._real_ability()
        text = f'<tool_call>{{"source": "https://example.com", "action": "{name}"}}</tool_call>'
        cleaned, calls = self._extract(text)
        assert len(calls) == 1
        assert calls[0]['name'] == name
        assert 'action' not in calls[0]['input']
        assert calls[0]['input'] == {'source': 'https://example.com'}
        assert '<tool_call>' not in cleaned

    def test_unregistered_action_skipped(self):
        """An action value not in the registry must be silently skipped.
        The block is still stripped from the text."""
        text = '<tool_call>{"action": "check", "items": ["milk"]}</tool_call>'
        cleaned, calls = self._extract(text)
        assert calls == []
        assert '<tool_call>' not in cleaned

    def test_multiple_blocks_both_extracted(self):
        """Two valid <tool_call> blocks produce two entries in order."""
        name = self._real_ability()
        text = (
            f'<tool_call>{{"action": "{name}", "query": "foo"}}</tool_call>\n'
            f'<tool_call>{{"action": "{name}", "query": "bar"}}</tool_call>'
        )
        cleaned, calls = self._extract(text)
        assert len(calls) == 2
        assert calls[0]['input']['query'] == 'foo'
        assert calls[1]['input']['query'] == 'bar'
        assert '<tool_call>' not in cleaned

    def test_malformed_json_skipped_block_still_stripped(self):
        """Non-parseable JSON is skipped silently; markup is still removed."""
        text = '<tool_call>not json{</tool_call>'
        cleaned, calls = self._extract(text)
        assert calls == []
        assert '<tool_call>' not in cleaned

    def test_no_inline_blocks_passthrough(self):
        """Plain text with no blocks returns the original text and empty list."""
        text = 'Hello, I have no tool calls here.'
        cleaned, calls = self._extract(text)
        assert calls == []
        assert cleaned == text

    def test_structured_channel_wins_inline_ignored(self):
        """When raw_tool_calls is present, _parse_chat_response must not call
        _extract_inline_tool_calls — inline blocks stay in the text unchanged."""
        name = self._real_ability()
        from services.ollama_service import _parse_chat_response

        data = {
            'message': {
                'content': f'<tool_call>{{"action": "{name}"}}</tool_call>',
                'tool_calls': [
                    {'function': {'name': 'structured_tool', 'arguments': {'k': 'v'}}}
                ],
            },
            'model': 'qwen2.5:7b',
        }
        resp = _parse_chat_response(data, 'qwen2.5:7b')
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]['name'] == 'structured_tool'
        # Inline block untouched — content preserved as-is
        assert '<tool_call>' in resp.text
        assert resp.stop_reason == 'tool_use'
