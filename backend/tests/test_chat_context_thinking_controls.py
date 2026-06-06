"""Feature tests — chat composer context-size indicator + thinking-level override.

Real production hot path, real SQLite (the `db` fixture), zero mocks:
  * the context-usage REST endpoint reads the real ``llm_call_log`` written by
    the production ``log_call`` logger and the real selected-provider window;
  * the thinking-override is persisted through the real ``SettingsService`` and
    read back through the real resolver;
  * the deliberation gate is the REAL gate method on a REAL MessageProcessor;
  * the send-time precedence is the REAL function ``providers.send`` calls.
"""

import pytest

pytestmark = pytest.mark.unit


# ── Fixtures ────────────────────────────────────────────────────────────────
# Endpoint tests drive the real production app via the shared ``authed_client``
# fixture (conftest) — it builds the full ``create_app()`` with every blueprint
# registered and the suite-wide auth/session harness, so there is no per-test
# mock of any collaborator under test.


def _make_provider(db, *, name, max_tokens):
    cur = db.cursor()
    cur.execute(
        "INSERT INTO providers (name, platform, model, max_tokens) "
        "VALUES (?, 'ollama', 'llama3', ?)",
        (name, max_tokens),
    )
    db.commit()
    return cur.lastrowid


# ── Context-size indicator ──────────────────────────────────────────────────

class TestContextUsageEndpoint:

    def test_returns_last_user_turn_tokens_and_window(self, authed_client):
        """GET /system/context-usage = last user-turn tokens_input / selected window.

        The production logger keys the row by ``job_name`` = ``channel:role``; the
        UserConfig turn is 'user:user' (asserted in
        ``test_indicator_filter_matches_real_config_jobs``).
        """
        from services.llm_call_log_service import log_call
        from services.provider_db_service import ProviderDbService
        import services.database_service as _db_mod

        client, db, _ = authed_client
        pid = _make_provider(db, name='active', max_tokens=64400)
        ProviderDbService(_db_mod._shared_db_service).set_selected_provider(pid)

        # Production logger writes the real row the endpoint reads.
        log_call('user:user', 'ollama', 'llama3',
                 tokens_input=2412, tokens_output=88, latency_ms=10,
                 usage_class='chat')

        resp = client.get('/system/context-usage')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['last_request_tokens'] == 2412
        assert data['context_window'] == 64400

    def test_delegate_and_thinking_calls_do_not_shadow_user_turn(self, authed_client):
        """THE regression: thinking + web_search delegate share usage_class='chat'
        (they inherit the parent's CHAT policy_channel), so a usage_class filter
        made the indicator oscillate user-turn (~17k) → delegate (~2k). Keying on
        job_name='user:user' must keep the user turn's size pinned even though
        the delegate's tiny sub-request is the NEWEST chat-class row.
        """
        from services.llm_call_log_service import log_call
        from services.provider_db_service import ProviderDbService
        import services.database_service as _db_mod

        client, db, _ = authed_client
        pid = _make_provider(db, name='active', max_tokens=200000)
        ProviderDbService(_db_mod._shared_db_service).set_selected_provider(pid)

        # The real user turn — large request.
        log_call('user:user', 'ollama', 'llama3', tokens_input=17240,
                 tokens_output=120, latency_ms=30, usage_class='chat')
        # Newer rows fired by the SAME turn's sub-agents, all usage_class='chat':
        #   the thinking pre-pass and several web_search delegate iterations.
        log_call('thinking:thinking', 'ollama', 'llama3', tokens_input=16980,
                 tokens_output=40, latency_ms=20, usage_class='chat')
        log_call('delegate:web_search:web_search', 'ollama', 'llama3',
                 tokens_input=2010, tokens_output=15, latency_ms=8, usage_class='chat')
        log_call('delegate:web_search:web_search', 'ollama', 'llama3',
                 tokens_input=2240, tokens_output=15, latency_ms=8, usage_class='chat')

        data = client.get('/system/context-usage').get_json()
        # Pinned to the user turn — NOT the newest (delegate) chat-class row.
        assert data['last_request_tokens'] == 17240

    def test_indicator_filter_matches_real_config_jobs(self):
        """Guard the magic string: the endpoint filters job_name='user:user', so
        the REAL UserConfig must produce that job, and the sub-agent configs that
        share usage_class='chat' must NOT — else the filter silently breaks.
        """
        from configs.channels import UserConfig
        from services.processor_config import ProcessorConfig
        from abilities.web_search import WebSearchConfig
        from abilities.thinking import ThinkingConfig

        chat = ProcessorConfig.POLICY_CHANNEL.CHAT
        assert UserConfig().job == 'user:user'
        assert WebSearchConfig(chat).job != 'user:user'
        assert ThinkingConfig([], frozenset(), chat).job != 'user:user'
        # All three share the SAME usage_class — proving why usage_class can't
        # be the discriminator and job_name must be.
        assert UserConfig().usage_class == WebSearchConfig(chat).usage_class == 'chat'

    def test_nulls_when_no_calls_or_provider(self, authed_client):
        """Empty DB → both fields null; endpoint never raises."""
        client, _db, _ = authed_client
        data = client.get('/system/context-usage').get_json()
        assert data == {'last_request_tokens': None, 'context_window': None}


# ── Thinking-level override: persistence + resolver ─────────────────────────

class TestThinkingOverridePersistence:

    def test_set_then_read_medium(self, db):
        from services.settings_service import SettingsService
        from services.thinking_override_service import get_thinking_override
        import services.database_service as _db_mod

        SettingsService(_db_mod._shared_db_service).set('thinking_level_override', 'medium')
        assert get_thinking_override() == 'medium'

    def test_set_then_read_high(self, db):
        from services.settings_service import SettingsService
        from services.thinking_override_service import get_thinking_override
        import services.database_service as _db_mod

        SettingsService(_db_mod._shared_db_service).set('thinking_level_override', 'high')
        assert get_thinking_override() == 'high'

    def test_absent_is_auto_none(self, db):
        from services.thinking_override_service import get_thinking_override
        assert get_thinking_override() is None

    def test_invalid_value_treated_as_auto(self, db):
        from services.settings_service import SettingsService
        from services.thinking_override_service import get_thinking_override
        import services.database_service as _db_mod

        SettingsService(_db_mod._shared_db_service).set('thinking_level_override', 'banana')
        assert get_thinking_override() is None


# ── Thinking-level override: precedence (the exact fn send() uses) ───────────

class TestThinkingModePrecedence:

    def test_config_pin_wins_over_override(self):
        from services.providers import resolve_thinking_mode
        # A system ability hard-pins 'high' (compactor/thinking) — override must NOT downgrade it.
        assert resolve_thinking_mode('high', 'medium', 'low') == 'high'

    def test_override_wins_over_gate_level(self):
        from services.providers import resolve_thinking_mode
        assert resolve_thinking_mode(None, 'medium', 'low') == 'medium'

    def test_gate_level_when_no_override(self):
        from services.providers import resolve_thinking_mode
        assert resolve_thinking_mode(None, None, 'high') == 'high'


# ── Thinking-level override: gate bypass on the real MessageProcessor ───────

class TestThinkingGateBypass:

    def test_override_short_circuits_gate(self, db):
        """With an override set, the REAL gate sets thinking_level to it and
        never reaches the deliberation classifier — proven by a trivial input
        that the gate would otherwise classify low, yet the level is the override."""
        from services.message_processor import MessageProcessor
        from configs.channels import UserConfig

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = UserConfig()          # channel='user' → gate runs
        mp.uid = None
        mp._uid = None
        mp.thinking_override = 'high'
        mp.thinking_level = '__SENTINEL__'

        mp._run_thinking_gate()           # the REAL gate, zero mocks

        assert mp.thinking_level == 'high'

    def test_no_override_leaves_gate_in_control(self, db):
        """Sanity: a fresh MP defaults thinking_override to None so the gate path
        is unchanged when the user has not set an override."""
        from services.message_processor import MessageProcessor

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        assert mp.thinking_override is None
