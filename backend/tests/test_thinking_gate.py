"""Feature tests for _run_thinking_gate() deliberation_score persistence.

Asserts that:
  - classifier returns None → 0.0 written to transcript.deliberation_score
  - classifier returns scalar → scalar written to transcript.deliberation_score
  - _uid is None → no DB write, no exception
  - DB exception in fallback path is swallowed

Uses real in-memory SQLite (db fixture from conftest). No mocks for the
database path — the real get_shared_db_service singleton is already patched
by the db fixture.
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_CHANNEL = 'user'
_ROLE = 'user'

# Patch targets — imports happen lazily inside _run_thinking_gate so we patch
# at the source module, not at services.message_processor.
_DSS_PATCH = 'services.deliberation_score_service.DeliberationScoreService'
_EMA_PATCH = 'services.deliberation_ema_service.DeliberationEmaService'
_DB_PATCH = 'services.database_service.get_shared_db_service'


def _make_processor(raw_input='hello'):
    """Return a concrete MessageProcessor subclass instance wired for CHANNEL='user'."""
    from services.message_processor import MessageProcessor
    from services.system_message_prompt import SystemMessagePrompt

    class _StubPrompt(SystemMessagePrompt):
        _SYSTEM_PROMPT = ''

    class _FakeUMP(MessageProcessor):
        CHANNEL = _CHANNEL
        ROLE = _ROLE
        SYSTEM_PROMPT_CLASS = _StubPrompt

        def getUserDefinition(self) -> str:
            return 'test user'

        def getUserPrompt(self) -> str:
            return raw_input

    return _FakeUMP(raw_input, {})


def _seed_transcript(conn):
    """Insert one transcript row and return its rowid."""
    conn.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES ('user', 'user', 'hello', '2026-04-29 10:00:00')"
    )
    uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    return uid


def _mock_classify(return_value):
    mock_svc = MagicMock()
    mock_svc.classify.return_value = return_value
    return mock_svc


def _mock_ema(ema_val=0.3, bucket='low'):
    mock_ema = MagicMock()
    mock_ema.peek.return_value = ema_val
    mock_ema.update_and_bucket.return_value = (ema_val, bucket)
    return mock_ema


class TestThinkingGateFallbackPersist:
    """classifier returns None → deliberation_score=0.0 written to transcript."""

    def test_none_scalar_persists_zero(self, db):
        uid = _seed_transcript(db)
        proc = _make_processor()
        proc._uid = uid

        with patch(_DSS_PATCH, return_value=_mock_classify(None)), \
             patch(_EMA_PATCH, return_value=_mock_ema()):
            proc._run_thinking_gate()

        row = db.execute(
            "SELECT deliberation_score FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.0, abs=1e-9)

    def test_none_scalar_sets_thinking_level_low(self, db):
        uid = _seed_transcript(db)
        proc = _make_processor()
        proc._uid = uid

        with patch(_DSS_PATCH, return_value=_mock_classify(None)), \
             patch(_EMA_PATCH, return_value=_mock_ema()):
            proc._run_thinking_gate()

        assert proc._thinking_level == 'low'


class TestThinkingGateScalarPersist:
    """classifier returns a real scalar → that scalar written (regression guard)."""

    def test_non_none_scalar_persisted(self, db):
        uid = _seed_transcript(db)
        proc = _make_processor()
        proc._uid = uid

        with patch(_DSS_PATCH, return_value=_mock_classify(0.42)), \
             patch(_EMA_PATCH, return_value=_mock_ema(ema_val=0.42, bucket='medium')):
            proc._run_thinking_gate()

        row = db.execute(
            "SELECT deliberation_score FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.42, abs=1e-6)


class TestThinkingGateNullUid:
    """_uid is None → no DB write, no exception raised."""

    def test_uid_none_no_exception(self, db):
        proc = _make_processor()
        proc._uid = None

        with patch(_DSS_PATCH, return_value=_mock_classify(None)), \
             patch(_EMA_PATCH, return_value=_mock_ema()):
            proc._run_thinking_gate()

        assert proc._thinking_level == 'low'

    def test_uid_none_no_db_write(self, db):
        uid = _seed_transcript(db)
        proc = _make_processor()
        proc._uid = None

        with patch(_DSS_PATCH, return_value=_mock_classify(None)), \
             patch(_EMA_PATCH, return_value=_mock_ema()):
            proc._run_thinking_gate()

        row = db.execute(
            "SELECT deliberation_score FROM transcript WHERE id = ?", (uid,)
        ).fetchone()
        assert row[0] is None


class TestThinkingGateDbExceptionSwallowed:
    """DB failure in the fallback persist path is swallowed — no exception raised."""

    def test_db_exception_does_not_propagate(self, db):
        uid = _seed_transcript(db)
        proc = _make_processor()
        proc._uid = uid

        boom = RuntimeError("disk full")

        with patch(_DSS_PATCH, return_value=_mock_classify(None)), \
             patch(_EMA_PATCH, return_value=_mock_ema()), \
             patch(_DB_PATCH, side_effect=boom):
            proc._run_thinking_gate()

        assert proc._thinking_level == 'low'
