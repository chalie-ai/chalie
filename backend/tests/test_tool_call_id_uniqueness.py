"""Tool-call ids must be globally unique across a turn.

Regression guard: the ACT-trail keys tool pills by ``call_id``.
Ollama minted ids as ``ollama_<name>_<index>`` where the index resets every
LLM response, so the same tool at the same position in two ACT iterations
collided on one id. The frontend then rendered a duplicate pill and left the
new spinner stuck (the end event resolved the older, already-done pill).
Gemini had the same class of bug using a millisecond timestamp.

The contract these tests pin: any two distinct tool calls — whether in the
same response or across responses — must receive distinct ids.
"""
from types import SimpleNamespace
from typing import cast

import pytest

pytestmark = pytest.mark.unit


def _ollama_response_with(*tool_names: str) -> dict[str, object]:
    return {
        'message': {
            'content': '',
            'tool_calls': [
                {'function': {'name': n, 'arguments': {}}} for n in tool_names
            ],
        }
    }


class TestOllamaToolCallIds:
    def test_same_tool_across_responses_gets_unique_ids(self) -> None:
        from services.llm_clients.ollama import _parse_chat_response

        first = _parse_chat_response(_ollama_response_with('find_tools'), 'm')
        second = _parse_chat_response(_ollama_response_with('find_tools'), 'm')

        id_a = cast(list[dict[str, object]], first.tool_calls)[0]['id']
        id_b = cast(list[dict[str, object]], second.tool_calls)[0]['id']
        assert id_a != id_b, f"colliding ids across responses: {id_a!r}"

    def test_same_tool_twice_in_one_response_unique(self) -> None:
        from services.llm_clients.ollama import _parse_chat_response

        resp = _parse_chat_response(
            _ollama_response_with('find_tools', 'find_tools'), 'm'
        )
        ids = [tc['id'] for tc in cast(list[dict[str, object]], resp.tool_calls)]
        assert len(set(ids)) == len(ids), f"duplicate ids in one response: {ids}"


class TestGeminiToolCallIds:
    def test_same_tool_twice_gets_unique_ids(self) -> None:
        from services.llm_clients.gemini import _accumulate_part as _gemini_accumulate_part

        def make_part(name: str) -> object:
            return SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(name=name, args={}),
            )

        tool_calls: list[dict[str, object]] = []
        _gemini_accumulate_part(make_part('find_tools'), [], tool_calls)
        _gemini_accumulate_part(make_part('find_tools'), [], tool_calls)

        ids = [tc['id'] for tc in tool_calls]
        assert len(set(ids)) == len(ids), f"colliding gemini ids: {ids}"
