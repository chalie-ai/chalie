"""
Tests for UserSummaryProcessor and UserSummarySystemPrompt.

Unit tests use in-memory SQLite via the ``db`` fixture from conftest (built from
schema.sql via SchemaConvergenceService.converge()).  Integration tests require
a live provider configured for ``frontal-cortex-unified``.

All tests: @pytest.mark.unit or @pytest.mark.integration
"""

import json
import logging

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _insert_trait(db, key, value, last_confirmed_at=None):
    """Insert a user_specific row into data_graph."""
    from services.time_utils import utc_now

    now = utc_now().isoformat()
    lc = last_confirmed_at or now
    db.execute(
        """
        INSERT INTO data_graph
            (kind, key, value, active, deleted_at, retrieval_weight,
             storage_strength, evidence_count, first_seen_at, last_confirmed_at,
             source)
        VALUES ('user_specific', ?, ?, 1, NULL, 1.0, 0.5, 1, ?, ?, 'test')
        """,
        (key, value, now, lc),
    )
    db.commit()


def _insert_summary(db, value, key='user_summary', last_confirmed_at=None):
    """Insert a kind='system' user_summary row into data_graph."""
    from services.time_utils import utc_now

    now = utc_now().isoformat()
    lc = last_confirmed_at or now
    db.execute(
        """
        INSERT INTO data_graph
            (kind, key, value, active, deleted_at, retrieval_weight,
             storage_strength, evidence_count, first_seen_at, last_confirmed_at,
             source)
        VALUES ('system', ?, ?, 1, NULL, 0.7, 0.5, 1, ?, ?, 'user_summary_processor')
        """,
        (key, value, now, lc),
    )
    db.commit()


def _make_fake_response(content: str):
    """Return a minimal LLMResponse-like object with a text attribute."""
    from services.llm_service import LLMResponse

    return LLMResponse(text=content, model='test', provider='test')


# ── TestUserSummarySystemPrompt ────────────────────────────────────────────────

@pytest.mark.unit
class TestUserSummarySystemPrompt:

    def test_prompt_instantiates_and_returns_string(self):
        from services.system_message_prompt import UserSummarySystemPrompt

        inst = UserSummarySystemPrompt()
        body = inst.getPrompt()
        assert isinstance(body, str)
        assert len(body) > 0

    def test_prompt_contains_json_schema_keys(self):
        from services.system_message_prompt import UserSummarySystemPrompt

        body = UserSummarySystemPrompt().getPrompt()
        assert '"short"' in body
        assert '"long"' in body

    def test_prompt_instructs_third_person(self):
        from services.system_message_prompt import UserSummarySystemPrompt

        body = UserSummarySystemPrompt().getPrompt()
        assert 'third person' in body.lower()


# ── TestGetUserPrompt ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetUserPrompt:

    def test_user_prompt_contains_only_key_value(self, db):
        """Seeded rows appear as ``key: value`` with NO telemetry tokens."""
        from services.user_summary_processor import UserSummaryProcessor

        _insert_trait(db, 'name', 'Alice')
        _insert_trait(db, 'location', 'Malta')
        _insert_trait(db, 'role', 'Engineer')

        proc = UserSummaryProcessor()
        prompt = proc.getUserPrompt()

        # Each key: value pair must appear
        assert 'name: Alice' in prompt
        assert 'location: Malta' in prompt
        assert 'role: Engineer' in prompt

        # No telemetry tokens
        for forbidden in ('weight', 'source=', 'created_at', 'last_confirmed_at', 'retrieval_weight'):
            assert forbidden not in prompt, f"Telemetry token '{forbidden}' found in prompt"

    def test_user_prompt_starts_with_facts_prefix(self, db):
        _insert_trait(db, 'hobby', 'cycling')

        from services.user_summary_processor import UserSummaryProcessor

        prompt = UserSummaryProcessor().getUserPrompt()
        assert prompt.startswith('Facts:')

    def test_user_prompt_no_traits_returns_empty_placeholder(self, db):
        from services.user_summary_processor import UserSummaryProcessor

        prompt = UserSummaryProcessor().getUserPrompt()
        assert 'no facts available' in prompt


# ── TestGetUserDefinition ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetUserDefinition:

    def test_returns_static_synthesiser_string(self):
        from services.user_summary_processor import UserSummaryProcessor

        proc = UserSummaryProcessor()
        defn = proc.getUserDefinition()
        assert 'synthesiser' in defn.lower()
        assert 'real human' in defn.lower()


# ── TestPostTurn ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPostTurn:

    def test_post_turn_writes_both_rows(self, db):
        """Valid JSON response writes user_summary and user_summary_long rows."""
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        payload = json.dumps({'short': 'foo', 'long': 'bar baz'})

        proc = UserSummaryProcessor()
        proc._last_response = payload
        proc.postTurn()

        dgs = get_data_graph_service()

        short_rows = [
            r for r in dgs.fetch(kinds=['system'], include_inactive=True)
            if r.get('key') == 'user_summary' and r.get('source') == 'user_summary_processor'
        ]
        long_rows = [
            r for r in dgs.fetch(kinds=['system'], include_inactive=True)
            if r.get('key') == 'user_summary_long' and r.get('source') == 'user_summary_processor'
        ]

        assert short_rows, "user_summary row not written"
        assert long_rows, "user_summary_long row not written"
        assert short_rows[0]['value'] == 'foo'
        assert long_rows[0]['value'] == 'bar baz'

    def test_post_turn_malformed_json_writes_nothing_and_logs_warning(self, db, caplog):
        """Garbage content: no rows written, warning logged, existing row untouched."""
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        # Seed a pre-existing summary row
        _insert_summary(db, 'existing summary')

        proc = UserSummaryProcessor()
        proc._last_response = 'this is definitely not json }{{'

        with caplog.at_level(logging.WARNING, logger='services.user_summary_processor'):
            proc.postTurn()

        dgs = get_data_graph_service()

        # The pre-existing row must still be there and unchanged
        rows = [
            r for r in dgs.fetch(kinds=['system'])
            if r.get('key') == 'user_summary'
        ]
        assert rows, "Pre-existing user_summary row was removed"
        assert rows[0]['value'] == 'existing summary'

        # No long row written
        long_rows = [
            r for r in dgs.fetch(kinds=['system'])
            if r.get('key') == 'user_summary_long'
        ]
        assert not long_rows, "user_summary_long written despite malformed JSON"

        assert any('parse' in rec.message.lower() or 'json' in rec.message.lower()
                   for rec in caplog.records), "No warning logged for malformed JSON"

    def test_post_turn_missing_short_key_writes_nothing(self, db):
        """JSON with missing 'short' key writes nothing."""
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        proc = UserSummaryProcessor()
        proc._last_response = json.dumps({'long': 'some long text'})
        proc.postTurn()

        dgs = get_data_graph_service()
        rows = [r for r in dgs.fetch(kinds=['system']) if r.get('key') == 'user_summary']
        assert not rows, "user_summary written despite missing 'short' key"

    def test_post_turn_empty_response_writes_nothing(self, db):
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        proc = UserSummaryProcessor()
        proc._last_response = ''
        proc.postTurn()

        dgs = get_data_graph_service()
        rows = [r for r in dgs.fetch(kinds=['system']) if r.get('key') in ('user_summary', 'user_summary_long')]
        assert not rows

    def test_post_turn_strips_markdown_code_fences(self, db):
        """LLM wrapping JSON in ```json ... ``` fences is handled gracefully."""
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        fenced = '```json\n{"short": "hello", "long": "world"}\n```'

        proc = UserSummaryProcessor()
        proc._last_response = fenced
        proc.postTurn()

        dgs = get_data_graph_service()
        rows = [
            r for r in dgs.fetch(kinds=['system'])
            if r.get('key') == 'user_summary' and r.get('source') == 'user_summary_processor'
        ]
        assert rows, "user_summary not written despite valid fenced JSON"
        assert rows[0]['value'] == 'hello'

    def test_post_turn_empty_short_value_writes_nothing(self, db, caplog):
        """JSON with 'short' present but empty string must be treated as missing.

        The LLM may return {"short": "", "long": "..."} when it has no short
        synthesis to offer.  The postTurn() guard (``if not short or not long_``)
        must catch this and skip the write — an empty user_summary row would
        silently become the user definition in all subsequent turns.
        """
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        proc = UserSummaryProcessor()
        proc._last_response = '{"short": "", "long": "a thorough description"}'

        with caplog.at_level(logging.WARNING, logger='services.user_summary_processor'):
            proc.postTurn()

        dgs = get_data_graph_service()
        rows = [
            r for r in dgs.fetch(kinds=['system'])
            if r.get('key') in ('user_summary', 'user_summary_long')
        ]
        assert not rows, (
            "postTurn() wrote rows despite empty 'short' value — "
            "empty string would become the user definition"
        )
        assert any(
            'short' in rec.message.lower() or 'long' in rec.message.lower() or 'missing' in rec.message.lower() or 'empty' in rec.message.lower()
            for rec in caplog.records
        ), "No warning logged when 'short' value is empty string"

    def test_post_turn_whitespace_only_long_value_writes_nothing(self, db, caplog):
        """JSON with 'long' as whitespace-only string must be treated as empty.

        After ``.strip()`` the value collapses to '' — the guard triggers and
        nothing is written.  Verifies the strip() is applied before the emptiness
        check, not after.
        """
        from services.data_graph_service import get_data_graph_service
        from services.user_summary_processor import UserSummaryProcessor

        proc = UserSummaryProcessor()
        proc._last_response = '{"short": "Alice is an engineer", "long": "   "}'

        with caplog.at_level(logging.WARNING, logger='services.user_summary_processor'):
            proc.postTurn()

        dgs = get_data_graph_service()
        rows = [
            r for r in dgs.fetch(kinds=['system'])
            if r.get('key') in ('user_summary', 'user_summary_long')
        ]
        assert not rows, (
            "postTurn() wrote rows despite whitespace-only 'long' value"
        )


# ── TestPostTurnStorageFailure ─────────────────────────────────────────────────

@pytest.mark.unit
class TestPostTurnStorageFailure:
    """Verify that postTurn() is resilient to a DataGraphService write failure.

    The DB table is deliberately dropped after the fixture sets up the schema
    so that ``dgs.store()`` raises ``OperationalError: no such table: data_graph``.
    The production ``except`` block must catch it, log a warning, and return
    without propagating — protecting any caller (worker or lazy daemon).
    """

    def test_post_turn_store_raises_does_not_propagate_and_warns(self, db, caplog):
        """Valid JSON + broken DB write: postTurn() must not raise; must log a warning.

        Pre-existing user_summary content must survive — the broken write for the
        new synthesis must not corrupt what was already there.
        """
        from services.user_summary_processor import UserSummaryProcessor

        # Drop the table so dgs.store() will raise OperationalError on the
        # INSERT — the real production failure mode when the DB is closed or
        # the schema diverges mid-run.
        db.execute("DROP TABLE IF EXISTS data_graph")
        db.commit()

        proc = UserSummaryProcessor()
        proc._last_response = '{"short": "Alice is a tester", "long": "Alice works in QA"}'

        # Must not raise — caller must never see the storage error.
        with caplog.at_level(logging.WARNING, logger='services.user_summary_processor'):
            try:
                proc.postTurn()
            except Exception as exc:
                raise AssertionError(
                    f"postTurn() must not propagate storage exceptions to callers, "
                    f"but raised: {type(exc).__name__}: {exc}"
                ) from exc

        assert any(
            'write failed' in rec.message.lower() or 'failed' in rec.message.lower()
            for rec in caplog.records
        ), "No warning logged when data_graph write fails"


# ── TestClassAttributes ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestClassAttributes:

    def test_skip_transcript_write_is_true(self):
        from services.user_summary_processor import UserSummaryProcessor

        assert UserSummaryProcessor.SKIP_TRANSCRIPT_WRITE is True

    def test_max_iterations_is_one(self):
        from services.user_summary_processor import UserSummaryProcessor

        assert UserSummaryProcessor.MAX_ITERATIONS == 1

    def test_native_tools_is_empty(self):
        from services.user_summary_processor import UserSummaryProcessor

        assert UserSummaryProcessor.NATIVE_TOOLS == []

    def test_job_is_frontal_cortex_unified(self):
        from services.user_summary_processor import UserSummaryProcessor

        assert UserSummaryProcessor.JOB == 'frontal-cortex-unified'


# ── TestShouldSynthesiseBehavioralPattern ─────────────────────────────────────

@pytest.mark.unit
class TestShouldSynthesiseBehavioralPattern:

    def test_should_synthesise_returns_true_for_pattern_only_update(self, db):
        """Broadened MAX: a new behavioral_pattern row (no new user_specific) fires synthesis.

        Regression guard: before §5.3 the MAX query only included 'user_specific'.
        A fresh pattern written after the last synthesis would be invisible to the
        sentinel and synthesis would never fire — the pattern would never reach the
        synopsis.  This test proves the broadened IN clause is effective.
        """
        from datetime import timedelta

        from services.time_utils import utc_now
        from services.user_summary_processor import UserSummaryProcessor

        now = utc_now()
        two_days_ago = (now - timedelta(days=2)).isoformat()
        one_hour_ago = (now - timedelta(hours=1)).isoformat()

        # Seed the existing user_summary row (kind='system') at T-2 days.
        db.execute(
            """
            INSERT INTO data_graph
                (kind, key, value, active, deleted_at, retrieval_weight,
                 storage_strength, evidence_count, first_seen_at, last_confirmed_at,
                 source)
            VALUES ('system', 'user_summary', 'old synopsis', 1, NULL, 0.7, 0.5, 1, ?, ?, 'user_summary_processor')
            """,
            (two_days_ago, two_days_ago),
        )

        # Seed a behavioral_pattern row at T-1 hour (newer than the synopsis).
        # No user_specific rows at all — the trigger must come purely from this pattern.
        pattern_content = json.dumps({
            'vertical': 'meal_times',
            'class': 'time_routine',
            'slots': {'event_label': 'dinner', 'day_bucket': 'fri', 'hour_window': '18:30-20:00'},
            'recurrence': 7,
            'sigma_confidence': 4.8,
            'first_seen': two_days_ago,
            'last_seen': one_hour_ago,
            'status': 'active',
            'source': 'llm',
        })
        db.execute(
            """
            INSERT INTO data_graph
                (kind, key, value, active, deleted_at, retrieval_weight,
                 storage_strength, evidence_count, first_seen_at, last_confirmed_at,
                 source)
            VALUES ('behavioral_pattern', 'meal_times', ?, 1, NULL, 0.5, 0.5, 1, ?, ?, 'pattern_extractor')
            """,
            (pattern_content, two_days_ago, one_hour_ago),
        )
        db.commit()

        assert UserSummaryProcessor()._should_synthesise() is True, (
            "_should_synthesise() returned False despite a behavioral_pattern row "
            "newer than the user_summary — broadened MAX clause not effective"
        )


# ── TestFormatPatternLine ──────────────────────────────────────────────────────
# The old extractor had 4-class × 6-vertical logic. PatternMatchProcessor
# (v0.5.0 §6 rewrite) uses a flat content shape: name, frequency, time_anchor,
# summary, confidence, last_seen_at. The four old tests tested extinct concepts
# and are replaced with two that verify the new flat format.

@pytest.mark.unit
class TestFormatPatternLine:

    def test_format_pattern_line_with_time_anchor(self):
        """content with time_anchor includes '@ <anchor>' in the formatted line."""
        from services.user_summary_processor import _format_pattern_line

        content = {
            'name': 'morning_run',
            'frequency': 'weekday',
            'time_anchor': '07:00',
            'summary': 'goes for a run in the morning',
            'confidence': 7.0,
            'last_seen_at': '2026-04-22T07:15:00+00:00',
        }
        line = _format_pattern_line(content)

        assert 'morning_run' in line
        assert 'weekday' in line
        assert '@ 07:00' in line
        assert 'goes for a run in the morning' in line

    def test_format_pattern_line_without_time_anchor(self):
        """content with empty time_anchor must not include '@' in the formatted line."""
        from services.user_summary_processor import _format_pattern_line

        content = {
            'name': 'weekend_reading',
            'frequency': 'weekend',
            'time_anchor': '',
            'summary': 'reads books on weekends',
            'confidence': 5.0,
            'last_seen_at': '2026-04-20T14:00:00+00:00',
        }
        line = _format_pattern_line(content)

        assert 'weekend_reading' in line
        assert '@' not in line, (
            "No '@' expected when time_anchor is empty, but '@' found in: " + line
        )


# ── Integration tests (require live provider) ──────────────────────────────────

@pytest.mark.integration
class TestUserSummaryProcessorIntegration:

    def test_synthesise_then_should_not_refire(self, db):
        """Seed traits, run synthesis, then _should_synthesise() must return False.

        Proves no infinite re-synthesis loop.
        """
        from services.user_summary_processor import UserSummaryProcessor

        _insert_trait(db, 'name', 'Bob')
        _insert_trait(db, 'city', 'Valletta')

        UserSummaryProcessor().send()

        assert not UserSummaryProcessor()._should_synthesise(), (
            "_should_synthesise() returned True immediately after synthesis — "
            "infinite loop risk"
        )
