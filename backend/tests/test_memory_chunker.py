"""Tests for memory_chunker_worker — LLM extraction pipeline."""

import json
import pytest
from unittest.mock import patch, MagicMock
from workers.memory_chunker_worker import memory_chunker_worker, load_config


pytestmark = pytest.mark.unit


def _make_job_data(**overrides):
    base = {
        'topic': 'test-topic',
        'exchange_id': 'abc12345-6789-0000-0000-000000000000',
        'prompt_message': 'What is Python?',
        'response_message': 'Python is a programming language.',
        'thread_id': 'test:user1:chan1:1',
    }
    base.update(overrides)
    return base


def _make_llm_mock(response_text: str):
    """Create a mock LLM service whose send_message().text returns response_text."""
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_llm = MagicMock()
    mock_llm.send_message.return_value = mock_response
    return mock_llm


class TestMemoryChunker:

    @patch('workers.memory_chunker_worker.enqueue_episodic_memory')
    @patch('workers.memory_chunker_worker.ThreadConversationService')
    @patch('workers.memory_chunker_worker.create_llm_service')
    @patch('workers.memory_chunker_worker.WorldStateService')
    @patch('workers.memory_chunker_worker.ConfigService')
    @patch('workers.memory_chunker_worker.GistStorageService')
    @patch('workers.memory_chunker_worker.FactStoreService')
    def test_gists_stored_after_extraction(
        self, mock_fact_cls, mock_gist_cls, mock_config_cls,
        mock_ws_cls, mock_llm_factory, mock_conv_cls, mock_enqueue,
    ):
        """Valid LLM JSON response → gists stored via GistStorageService."""
        # Config
        mock_config_cls.resolve_agent_config.return_value = {
            'model': 'test', 'attention_span_minutes': 30,
            'min_gist_confidence': 7, 'max_gists': 8,
            'gist_similarity_threshold': 0.7, 'max_gists_per_type': 2,
            'min_confidence': 0.5, 'ttl_minutes': 1440, 'max_facts_per_topic': 50,
        }
        mock_config_cls.get_agent_config.return_value = {
            'min_confidence': 0.5, 'ttl_minutes': 1440, 'max_facts_per_topic': 50,
        }
        mock_config_cls.get_agent_prompt.return_value = 'prompt {{world_state}}'

        # World state
        ws_instance = MagicMock()
        ws_instance.get_world_state.return_value = ''
        mock_ws_cls.return_value = ws_instance

        # Conversation service
        conv_instance = MagicMock()
        conv_instance.get_conversation_history.return_value = []
        mock_conv_cls.return_value = conv_instance

        # LLM returns valid gists
        llm_response = json.dumps({
            'gists': [
                {'content': 'User asked about Python', 'type': 'observation', 'confidence': 8},
            ],
            'scope': 'test',
        })
        mock_llm_factory.return_value = _make_llm_mock(llm_response)

        # Gist storage
        gist_instance = MagicMock()
        gist_instance.store_gists.return_value = 1
        mock_gist_cls.return_value = gist_instance

        # Fact store
        fact_instance = MagicMock()
        mock_fact_cls.return_value = fact_instance

        result = memory_chunker_worker(_make_job_data())

        assert 'Memory chunk generated' in result
        gist_instance.store_gists.assert_called_once()

    @patch('workers.memory_chunker_worker.enqueue_episodic_memory')
    @patch('workers.memory_chunker_worker.ThreadConversationService')
    @patch('workers.memory_chunker_worker.create_llm_service')
    @patch('workers.memory_chunker_worker.WorldStateService')
    @patch('workers.memory_chunker_worker.ConfigService')
    @patch('workers.memory_chunker_worker.GistStorageService')
    @patch('workers.memory_chunker_worker.FactStoreService')
    def test_facts_extracted_and_stored(
        self, mock_fact_cls, mock_gist_cls, mock_config_cls,
        mock_ws_cls, mock_llm_factory, mock_conv_cls, mock_enqueue,
    ):
        """Facts extracted and stored via FactStoreService."""
        mock_config_cls.resolve_agent_config.return_value = {
            'model': 'test', 'attention_span_minutes': 30,
            'min_gist_confidence': 7, 'max_gists': 8,
            'gist_similarity_threshold': 0.7, 'max_gists_per_type': 2,
            'min_confidence': 0.5, 'ttl_minutes': 1440, 'max_facts_per_topic': 50,
        }
        mock_config_cls.get_agent_config.return_value = {
            'min_confidence': 0.5, 'ttl_minutes': 1440, 'max_facts_per_topic': 50,
        }
        mock_config_cls.get_agent_prompt.return_value = 'prompt {{world_state}}'

        ws_instance = MagicMock()
        ws_instance.get_world_state.return_value = ''
        mock_ws_cls.return_value = ws_instance

        conv_instance = MagicMock()
        conv_instance.get_conversation_history.return_value = []
        mock_conv_cls.return_value = conv_instance

        # LLM returns memory chunk with facts included
        chunk_response = json.dumps({
            'gists': [],
            'scope': 'test',
            'facts': [
                {'key': 'language', 'value': 'Python', 'confidence': 9},
            ],
        })
        mock_llm_factory.return_value = _make_llm_mock(chunk_response)

        gist_instance = MagicMock()
        mock_gist_cls.return_value = gist_instance

        fact_instance = MagicMock()
        mock_fact_cls.return_value = fact_instance

        memory_chunker_worker(_make_job_data())

        fact_instance.store_fact.assert_called_once_with(
            topic='test-topic',
            key='language',
            value='Python',
            confidence=0.9,
            source=_make_job_data()['exchange_id'],
        )

    @patch('workers.memory_chunker_worker.enqueue_episodic_memory')
    @patch('workers.memory_chunker_worker.ThreadConversationService')
    @patch('workers.memory_chunker_worker.create_llm_service')
    @patch('workers.memory_chunker_worker.WorldStateService')
    @patch('workers.memory_chunker_worker.ConfigService')
    def test_episodic_job_enqueued(
        self, mock_config_cls, mock_ws_cls, mock_llm_factory,
        mock_conv_cls, mock_enqueue,
    ):
        """Episodic memory job enqueued after processing."""
        mock_config_cls.resolve_agent_config.return_value = {
            'model': 'test', 'attention_span_minutes': 30,
            'min_gist_confidence': 7, 'max_gists': 8,
            'min_confidence': 0.5, 'ttl_minutes': 1440, 'max_facts_per_topic': 50,
        }
        mock_config_cls.get_agent_config.return_value = {
            'min_confidence': 0.5, 'ttl_minutes': 1440, 'max_facts_per_topic': 50,
        }
        mock_config_cls.get_agent_prompt.return_value = 'prompt {{world_state}}'

        ws_instance = MagicMock()
        ws_instance.get_world_state.return_value = ''
        mock_ws_cls.return_value = ws_instance

        conv_instance = MagicMock()
        conv_instance.get_conversation_history.return_value = []
        mock_conv_cls.return_value = conv_instance

        mock_llm_factory.return_value = _make_llm_mock(json.dumps({'gists': [], 'scope': 'test'}))

        memory_chunker_worker(_make_job_data())

        mock_enqueue.assert_called_once_with({'topic': 'test-topic', 'thread_id': 'test:user1:chan1:1'})

    @patch('workers.memory_chunker_worker.ThreadConversationService')
    @patch('workers.memory_chunker_worker.create_llm_service')
    @patch('workers.memory_chunker_worker.WorldStateService')
    @patch('workers.memory_chunker_worker.ConfigService')
    def test_json_decode_error_handled(
        self, mock_config_cls, mock_ws_cls, mock_llm_factory, mock_conv_cls,
    ):
        """Bad LLM output → graceful error, no crash."""
        mock_config_cls.resolve_agent_config.return_value = {'model': 'test'}
        mock_config_cls.get_agent_prompt.return_value = 'prompt {{world_state}}'

        ws_instance = MagicMock()
        ws_instance.get_world_state.return_value = ''
        mock_ws_cls.return_value = ws_instance

        conv_instance = MagicMock()
        conv_instance.get_conversation_history.return_value = []
        mock_conv_cls.return_value = conv_instance

        mock_llm_factory.return_value = _make_llm_mock('not valid json {{{}')

        with pytest.raises(json.JSONDecodeError):
            memory_chunker_worker(_make_job_data())

    @patch('workers.memory_chunker_worker.ConfigService')
    def test_config_loaded_fresh(self, mock_config_cls):
        """Config should be read fresh each invocation."""
        mock_config_cls.resolve_agent_config.return_value = {'model': 'test'}
        mock_config_cls.get_agent_prompt.return_value = 'prompt'

        load_config()
        load_config()

        assert mock_config_cls.resolve_agent_config.call_count == 2
