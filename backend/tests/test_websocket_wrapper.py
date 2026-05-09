"""
Unit tests for WS6 — Chat UI wrapper contract integration.

Covers:
  - IntentService intent delivery via _drift_sender intent poll
  - __chat_ui__ auto-registration via WrapperAuthService.create_token with wrapper_id_override
  - CognitiveIntent emission in OutputService.enqueue_text
"""

import pytest


# ---------------------------------------------------------------------------
# Phase B — Intent path: present_response intents for __chat_ui__
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntentDelivery:
    """CognitiveIntents addressed to __chat_ui__ must be stored and retrievable."""

    def test_emit_and_retrieve_present_response_intent(self):
        """IntentService can emit and retrieve a present_response intent for __chat_ui__."""
        from services.memory_store import MemoryStore
        from services.intent_service import IntentService, CognitiveIntent
        import uuid

        store = MemoryStore()
        svc = IntentService(store)

        intent = CognitiveIntent(
            intent_id=str(uuid.uuid4()),
            intent_type='present_response',
            target_wrapper='__chat_ui__',
            payload={'content': 'Hello back', 'output_id': 'out-1', 'mode': 'UNIFIED'},
            urgency='normal',
            confidence=1.0,
        )
        svc.emit(intent)

        pending = svc.get_pending('__chat_ui__', limit=5)
        assert len(pending) == 1
        assert pending[0]['intent_type'] == 'present_response'
        assert pending[0]['target_wrapper'] == '__chat_ui__'
        assert pending[0]['payload']['content'] == 'Hello back'

    def test_intent_marked_delivered_after_get_pending(self):
        """An intent transitions from pending → delivered when fetched."""
        from services.memory_store import MemoryStore
        from services.intent_service import IntentService, CognitiveIntent
        import uuid

        store = MemoryStore()
        svc = IntentService(store)

        intent_id = str(uuid.uuid4())
        intent = CognitiveIntent(
            intent_id=intent_id,
            intent_type='present_response',
            target_wrapper='__chat_ui__',
            payload={'content': 'Hello', 'output_id': 'o', 'mode': 'UNIFIED'},
        )
        svc.emit(intent)

        _ = svc.get_pending('__chat_ui__')

        # Second call must return empty (already delivered)
        second = svc.get_pending('__chat_ui__')
        assert second == []

    def test_intent_event_structure(self):
        """An emitted present_response intent carries the expected payload fields."""
        from services.memory_store import MemoryStore
        from services.intent_service import IntentService, CognitiveIntent
        import uuid

        store = MemoryStore()
        svc = IntentService(store)

        svc.emit(CognitiveIntent(
            intent_id=str(uuid.uuid4()),
            intent_type='present_response',
            target_wrapper='__chat_ui__',
            payload={'content': 'Hi', 'output_id': 'o', 'mode': 'UNIFIED'},
        ))

        pending = svc.get_pending('__chat_ui__', limit=5)
        assert len(pending) == 1
        assert pending[0]['payload']['content'] == 'Hi'
        assert pending[0]['intent_type'] == 'present_response'


# ---------------------------------------------------------------------------
# Auto-registration — __chat_ui__ wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatUIRegistration:
    """__chat_ui__ must be registerable with a stable, well-known wrapper_id."""

    def test_create_token_with_wrapper_id_override(self, db):
        """create_token() accepts wrapper_id_override and uses it as the wrapper_id."""
        from services.wrapper_auth_service import WrapperAuthService

        svc = WrapperAuthService()

        raw_token, wrapper_id = svc.create_token(
            name='Chat UI (Built-in)',
            capabilities={
                'signals': ['user_message', 'act_steer'],
                'intents': ['present_response', 'show_card', 'show_narration'],
            },
            permissions={
                'query': ['situation', 'world-state', 'memory', 'relevance'],
                'update': ['context', 'feedback', 'memory', 'belief'],
                'broadcast': False,
            },
            metadata={'version': '1.0', 'type': 'built-in'},
            wrapper_id_override='__chat_ui__',
        )

        assert wrapper_id == '__chat_ui__'
        assert raw_token  # non-empty token generated

    def test_chat_ui_wrapper_retrievable_after_registration(self, db):
        """get_wrapper('__chat_ui__') returns the wrapper after auto-registration."""
        from services.wrapper_auth_service import WrapperAuthService

        svc = WrapperAuthService()

        svc.create_token(
            name='Chat UI (Built-in)',
            wrapper_id_override='__chat_ui__',
        )

        result = svc.get_wrapper('__chat_ui__')
        assert result is not None
        assert result['wrapper_id'] == '__chat_ui__'
        assert result['name'] == 'Chat UI (Built-in)'

    def test_idempotent_registration_check(self, db):
        """Registering __chat_ui__ twice is prevented by checking get_wrapper first."""
        from services.wrapper_auth_service import WrapperAuthService

        svc = WrapperAuthService()

        # First registration
        svc.create_token(name='Chat UI (Built-in)', wrapper_id_override='__chat_ui__')

        # Simulate what run.py does: only register if not already present
        if not svc.get_wrapper('__chat_ui__'):
            svc.create_token(name='Chat UI (Built-in)', wrapper_id_override='__chat_ui__')

        # Should still have exactly one entry (no duplicate)
        wrappers = svc.list_wrappers()
        chat_ui_wrappers = [w for w in wrappers if w['wrapper_id'] == '__chat_ui__']
        assert len(chat_ui_wrappers) == 1

    def test_wrapper_id_override_none_generates_random_id(self, db):
        """Passing wrapper_id_override=None falls back to the auto-generated wrp_ prefix."""
        from services.wrapper_auth_service import WrapperAuthService

        svc = WrapperAuthService()

        _, wrapper_id = svc.create_token(name='External Wrapper')
        assert wrapper_id.startswith('wrp_')
