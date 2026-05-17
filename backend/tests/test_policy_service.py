"""Unit tests for PolicyService — per-action permission control."""

import sqlite3
import pytest

from services.policy_service import (
    PolicyService,
    get_defaults,
    VALID_STATES,
    VALID_CONTEXTS,
    USAGE_CLASS_TO_CONTEXT,
)


@pytest.fixture()
def db():
    """In-memory SQLite with policy tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE policy_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id TEXT NOT NULL,
            context TEXT NOT NULL,
            state TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(action_id, context)
        );
        CREATE TABLE policy_blocked_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id TEXT NOT NULL,
            context TEXT NOT NULL,
            reason TEXT NOT NULL,
            params_json TEXT,
            created_at TEXT NOT NULL
        );
    """)
    yield conn
    conn.close()


class _FakeDB:
    """Minimal shim matching DatabaseService.connection() protocol."""
    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            yield self._conn
        return _ctx()


@pytest.fixture()
def svc(db):
    return PolicyService(_FakeDB(db))


# ── Defaults ─────────────────────────────────────────────────────────────────

class TestDefaults:

    def test_defaults_cover_all_contexts(self):
        defaults = get_defaults()
        for action_id, contexts in defaults.items():
            assert set(contexts.keys()) == VALID_CONTEXTS, f"{action_id} missing context"
            for ctx, state in contexts.items():
                assert state in VALID_STATES, f"{action_id}/{ctx} has invalid state {state}"

    def test_defaults_have_known_actions(self):
        defaults = get_defaults()
        assert "email.search" in defaults
        assert "email.manage" in defaults
        assert "code_eval" in defaults
        assert "find_tools" in defaults
        assert "subagent" in defaults

    def test_defaults_email_search_is_allow_in_chat(self):
        defaults = get_defaults()
        assert defaults["email.search"]["chat"] == "allow"

    def test_defaults_email_manage_is_ask_in_chat(self):
        defaults = get_defaults()
        assert defaults["email.manage"]["chat"] == "ask"

    def test_defaults_email_manage_is_deny_in_subconscious(self):
        defaults = get_defaults()
        assert defaults["email.manage"]["subconscious"] == "deny"

    def test_defaults_code_eval_is_ask_in_chat(self):
        defaults = get_defaults()
        assert defaults["code_eval"]["chat"] == "ask"

    def test_defaults_code_eval_is_deny_in_subconscious(self):
        defaults = get_defaults()
        assert defaults["code_eval"]["subconscious"] == "deny"

    def test_defaults_schedule_create_is_allow_in_chat(self):
        defaults = get_defaults()
        assert defaults["schedule.create"]["chat"] == "allow"

    def test_defaults_schedule_create_is_deny_in_subconscious(self):
        defaults = get_defaults()
        assert defaults["schedule.create"]["subconscious"] == "deny"

    def test_defaults_subagent_contexts_independent(self):
        defaults = get_defaults()
        assert defaults["email.manage"]["subagent"] == "ask"
        assert defaults["email.manage"]["subconscious"] == "deny"


# ── Seed ──────────────────────────────────────────────────────────────────────

class TestSeed:

    def test_seed_defaults_inserts_rows(self, svc, db):
        n = svc.seed_defaults()
        assert n > 0
        row_count = db.execute("SELECT COUNT(*) FROM policy_rules").fetchone()[0]
        assert row_count == n

    def test_seed_is_idempotent(self, svc):
        n1 = svc.seed_defaults()
        n2 = svc.seed_defaults()
        assert n1 > 0
        assert n2 == 0


# ── Check ─────────────────────────────────────────────────────────────────────

class TestCheck:

    def test_check_returns_db_value(self, svc, db):
        svc.seed_defaults()
        assert svc.check("email.search", "chat") == "allow"
        assert svc.check("email.manage", "chat") == "ask"

    def test_check_falls_back_to_defaults_when_no_row(self, svc):
        assert svc.check("email.search", "chat") == "allow"
        assert svc.check("email.manage", "subconscious") == "deny"

    def test_check_unknown_action_defaults_to_ask(self, svc):
        assert svc.check("nonexistent.action", "chat") == "ask"


# ── Upsert ────────────────────────────────────────────────────────────────────

class TestUpsert:

    def test_upsert_changes_state(self, svc):
        svc.seed_defaults()
        assert svc.check("email.manage", "chat") == "ask"

        svc.upsert([{"action_id": "email.manage", "context": "chat", "state": "allow"}])
        assert svc.check("email.manage", "chat") == "allow"

    def test_upsert_ignores_invalid_context(self, svc):
        affected = svc.upsert([{"action_id": "email.manage", "context": "invalid", "state": "allow"}])
        assert affected == 0

    def test_upsert_ignores_invalid_state(self, svc):
        affected = svc.upsert([{"action_id": "email.manage", "context": "chat", "state": "invalid"}])
        assert affected == 0

    def test_upsert_batch(self, svc):
        svc.seed_defaults()
        affected = svc.upsert([
            {"action_id": "email.manage", "context": "chat", "state": "allow"},
            {"action_id": "code_eval", "context": "chat", "state": "deny"},
        ])
        assert affected == 2
        assert svc.check("email.manage", "chat") == "allow"
        assert svc.check("code_eval", "chat") == "deny"


# ── Reset ──────────────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_restores_defaults(self, svc):
        svc.seed_defaults()
        svc.upsert([{"action_id": "email.manage", "context": "chat", "state": "allow"}])
        assert svc.check("email.manage", "chat") == "allow"

        svc.reset_to_defaults("chat")
        assert svc.check("email.manage", "chat") == "ask"

    def test_reset_scoped_to_context(self, svc):
        svc.seed_defaults()
        svc.upsert([
            {"action_id": "email.manage", "context": "chat", "state": "allow"},
            {"action_id": "email.manage", "context": "subagent", "state": "allow"},
        ])

        svc.reset_to_defaults("chat")
        assert svc.check("email.manage", "chat") == "ask"
        assert svc.check("email.manage", "subagent") == "allow"


# ── Get All ───────────────────────────────────────────────────────────────────

class TestGetAll:

    def test_get_all_returns_all_actions(self, svc):
        svc.seed_defaults()
        all_rules = svc.get_all()
        defaults = get_defaults()
        assert len(all_rules) >= len(defaults)

    def test_get_all_reflects_upserts(self, svc):
        svc.seed_defaults()
        svc.upsert([{"action_id": "email.manage", "context": "chat", "state": "allow"}])
        all_rules = svc.get_all()
        assert all_rules["email.manage"]["chat"] == "allow"


# ── Blocked Log ───────────────────────────────────────────────────────────────

class TestBlockedLog:

    def test_log_and_retrieve(self, svc):
        svc.log_blocked("email.manage", "subconscious", "user_unavailable", {"to": "test@x.com"})
        entries = svc.get_blocked_log()
        assert len(entries) == 1
        assert entries[0]["action_id"] == "email.manage"
        assert entries[0]["reason"] == "user_unavailable"
        assert entries[0]["params"]["to"] == "test@x.com"

    def test_clear_blocked_log(self, svc):
        svc.log_blocked("a", "chat", "user_denied")
        svc.log_blocked("b", "chat", "policy_deny")
        assert len(svc.get_blocked_log()) == 2

        cleared = svc.clear_blocked_log()
        assert cleared == 2
        assert len(svc.get_blocked_log()) == 0


# ── Resolve helpers ──────────────────────────────────────────────────────────

class TestResolve:

    def test_resolve_action_id_with_sub_action(self):
        assert PolicyService.resolve_action_id("email", {"action": "send"}) == "email.send"

    def test_resolve_action_id_without_sub_action(self):
        assert PolicyService.resolve_action_id("find_tools", {}) == "find_tools"

    def test_resolve_action_id_none_action(self):
        assert PolicyService.resolve_action_id("timer", {"action": None}) == "timer"

    def test_usage_class_mapping(self):
        assert USAGE_CLASS_TO_CONTEXT["chat"] == "chat"
        assert USAGE_CLASS_TO_CONTEXT["subagent"] == "subagent"
        assert USAGE_CLASS_TO_CONTEXT["subconscious"] == "subconscious"
