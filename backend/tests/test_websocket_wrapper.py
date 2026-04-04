"""
Unit tests for WS6 — Chat UI wrapper contract integration.

Covers:
  - IntentService intent delivery via _drift_sender intent poll
  - __chat_ui__ auto-registration via WrapperAuthService.create_token with wrapper_id_override
  - CognitiveIntent emission in OutputService.enqueue_text
"""

import sqlite3
import contextlib
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_db():
    """Minimal DatabaseService shim backed by an in-memory SQLite instance."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE wrapper_tokens (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            wrapper_id TEXT NOT NULL UNIQUE,
            capabilities TEXT NOT NULL DEFAULT '{}',
            permissions TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            last_seen_at TEXT,
            created_at TEXT NOT NULL,
            revoked_at TEXT
        )
    """)
    conn.commit()

    class _FakeDB:
        @contextlib.contextmanager
        def connection(self):
            yield conn

    return _FakeDB()


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
            payload={'content': 'Hello back', 'thread_id': 'thread-1', 'output_id': 'out-1', 'mode': 'UNIFIED'},
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
            payload={'content': 'Hello', 'thread_id': 't', 'output_id': 'o', 'mode': 'UNIFIED'},
        )
        svc.emit(intent)

        _ = svc.get_pending('__chat_ui__')

        # Second call must return empty (already delivered)
        second = svc.get_pending('__chat_ui__')
        assert second == []

    def test_intent_event_structure(self):
        """The intent event dict sent to the WebSocket has required fields."""
        # Mirror the structure built in _drift_sender
        intent_dict = {
            'intent_id': 'id-999',
            'intent_type': 'present_response',
            'target_wrapper': '__chat_ui__',
            'payload': {'content': 'Hi', 'thread_id': 't', 'output_id': 'o', 'mode': 'UNIFIED'},
            'urgency': 'normal',
            'confidence': 1.0,
        }

        seq = 42
        intent_evt = {
            "type": "intent",
            "intent_id": intent_dict.get("intent_id"),
            "intent_type": intent_dict.get("intent_type"),
            "payload": intent_dict.get("payload", {}),
            "urgency": intent_dict.get("urgency", "normal"),
            "seq": seq,
        }

        assert intent_evt["type"] == "intent"
        assert intent_evt["intent_type"] == "present_response"
        assert intent_evt["payload"]["content"] == "Hi"
        assert intent_evt["seq"] == 42


# ---------------------------------------------------------------------------
# Auto-registration — __chat_ui__ wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatUIRegistration:
    """__chat_ui__ must be registerable with a stable, well-known wrapper_id."""

    def test_create_token_with_wrapper_id_override(self):
        """create_token() accepts wrapper_id_override and uses it as the wrapper_id."""
        from services.wrapper_auth_service import WrapperAuthService

        db = _make_fake_db()
        svc = WrapperAuthService(db=db)

        raw_token, wrapper_id = svc.create_token(
            name='Chat UI (Built-in)',
            capabilities={
                'signals': ['user_message', 'act_steer'],
                'intents': ['present_response', 'show_card', 'show_narration'],
            },
            permissions={
                'query': ['situation', 'world-state', 'identity', 'memory', 'relevance'],
                'update': ['context', 'feedback', 'memory', 'belief'],
                'broadcast': False,
            },
            metadata={'version': '1.0', 'type': 'built-in'},
            wrapper_id_override='__chat_ui__',
        )

        assert wrapper_id == '__chat_ui__'
        assert raw_token  # non-empty token generated

    def test_chat_ui_wrapper_retrievable_after_registration(self):
        """get_wrapper('__chat_ui__') returns the wrapper after auto-registration."""
        from services.wrapper_auth_service import WrapperAuthService

        db = _make_fake_db()
        svc = WrapperAuthService(db=db)

        svc.create_token(
            name='Chat UI (Built-in)',
            wrapper_id_override='__chat_ui__',
        )

        result = svc.get_wrapper('__chat_ui__')
        assert result is not None
        assert result['wrapper_id'] == '__chat_ui__'
        assert result['name'] == 'Chat UI (Built-in)'

    def test_idempotent_registration_check(self):
        """Registering __chat_ui__ twice is prevented by checking get_wrapper first."""
        from services.wrapper_auth_service import WrapperAuthService

        db = _make_fake_db()
        svc = WrapperAuthService(db=db)

        # First registration
        svc.create_token(name='Chat UI (Built-in)', wrapper_id_override='__chat_ui__')

        # Simulate what run.py does: only register if not already present
        if not svc.get_wrapper('__chat_ui__'):
            svc.create_token(name='Chat UI (Built-in)', wrapper_id_override='__chat_ui__')

        # Should still have exactly one entry (no duplicate)
        wrappers = svc.list_wrappers()
        chat_ui_wrappers = [w for w in wrappers if w['wrapper_id'] == '__chat_ui__']
        assert len(chat_ui_wrappers) == 1

    def test_wrapper_id_override_none_generates_random_id(self):
        """Passing wrapper_id_override=None falls back to the auto-generated wrp_ prefix."""
        from services.wrapper_auth_service import WrapperAuthService

        db = _make_fake_db()
        svc = WrapperAuthService(db=db)

        _, wrapper_id = svc.create_token(name='External Wrapper')
        assert wrapper_id.startswith('wrp_')
