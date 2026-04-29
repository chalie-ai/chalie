import pytest

from services.message_processor import _sanitize_llm_args


@pytest.mark.unit
class TestSanitizeLLMArgs:
    def test_observed_qwen_list_items(self):
        assert _sanitize_llm_args('<|"|milk<|"|') == 'milk'

    def test_well_formed_qwen_sentinels(self):
        assert _sanitize_llm_args('<|im_start|>foo<|im_end|>') == 'foo'

    def test_nested_dict_with_sentinel_items(self):
        result = _sanitize_llm_args({'items': ['<|"|milk<|"|', 'eggs']})
        assert result == {'items': ['milk', 'eggs']}

    def test_clean_inputs_untouched(self):
        value = {'action': 'add', 'items': ['milk']}
        assert _sanitize_llm_args(value) == {'action': 'add', 'items': ['milk']}

    def test_non_string_scalars_untouched(self):
        value = {'limit': 10, 'active': True, 'ratio': 0.5}
        assert _sanitize_llm_args(value) == {'limit': 10, 'active': True, 'ratio': 0.5}

    def test_empty_string_after_strip(self):
        result = _sanitize_llm_args({'x': '<|"|<|"|'})
        assert result == {'x': ''}
