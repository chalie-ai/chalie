"""
Tests for backend/schema.sql structural integrity.

Each test absorbs a specific nightly scenario, validating that the production
schema contains the expected tables, columns, indexes, and seed data.
The fixture loads the full schema into an in-memory SQLite database so these
tests run with zero external dependencies.
"""

import sqlite3
import pytest
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_columns(conn, table_name):
    """Return set of column names for a table."""
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _get_tables(conn):
    """Return set of table names."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row[0] for row in rows}


def _get_indexes(conn, table_name):
    """Return set of index names for a table."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table_name,)
    ).fetchall()
    return {row[0] for row in rows}


# ── Test class ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSchemaValidation:

    @pytest.fixture
    def schema_db(self):
        """Load full production schema into in-memory SQLite.

        sqlite-vec (vec0) and FTS5 virtual tables are filtered out before
        execution because the bare in-memory SQLite used in unit tests does
        not have those extensions loaded.  All regular tables, indexes, and
        seed INSERT statements are preserved exactly as they appear in
        schema.sql.
        """
        import re
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        raw = schema_path.read_text()

        # Strip statements that create virtual tables requiring extensions
        # (vec0 for sqlite-vec, fts5 for full-text search).
        # A statement runs from the keyword CREATE to the closing semicolon.
        cleaned = re.sub(
            r'CREATE\s+VIRTUAL\s+TABLE\s+.*?;',
            '',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        conn = sqlite3.connect(':memory:')
        conn.executescript(cleaned)
        yield conn
        conn.close()

    # ── Scenario 102 ─────────────────────────────────────────────────────────

    def test_temporal_tables_exist(self, schema_db):
        """Absorbs scenario 102: temporal observation tables are present in schema.

        Verifies that both the raw-append table and the bounded aggregate table
        required by the ambient inference system are created by schema.sql.
        """
        tables = _get_tables(schema_db)
        assert 'temporal_observations' in tables, (
            "temporal_observations table missing from schema"
        )
        assert 'temporal_aggregate' in tables, (
            "temporal_aggregate table missing from schema"
        )

    # ── Scenario 103 ─────────────────────────────────────────────────────────

    def test_temporal_aggregate_columns(self, schema_db):
        """Absorbs scenario 103: temporal_aggregate has all required columns.

        The mining layer reads observation_type, observed_value, day_of_week,
        hour_bucket, device_class, count, and last_seen from this table.
        """
        columns = _get_columns(schema_db, 'temporal_aggregate')
        required = {
            'observation_type',
            'observed_value',
            'day_of_week',
            'hour_bucket',
            'device_class',
            'count',
            'last_seen',
        }
        missing = required - columns
        assert not missing, (
            f"temporal_aggregate is missing columns: {missing}"
        )

    # ── Scenario 107 ─────────────────────────────────────────────────────────

    def test_temporal_observation_types_enum(self, schema_db):
        """Absorbs scenario 107: temporal_observations.observation_type column exists.

        Valid types are: attention, energy, place, tempo, mobility.
        This test confirms the column exists and that the table accepts inserts
        for each valid observation type without error.
        """
        columns = _get_columns(schema_db, 'temporal_observations')
        assert 'observation_type' in columns, (
            "observation_type column missing from temporal_observations"
        )

        # Verify the table accepts all documented valid types
        valid_types = ('attention', 'energy', 'place', 'tempo', 'mobility')
        for obs_type in valid_types:
            schema_db.execute(
                "INSERT INTO temporal_observations "
                "(observation_type, observed_value, day_of_week, hour_bucket) "
                "VALUES (?, 'test_value', 0, 0)",
                (obs_type,)
            )
        schema_db.rollback()

    # ── Scenario 108 ─────────────────────────────────────────────────────────

    def test_temporal_day_hour_constraints(self, schema_db):
        """Absorbs scenario 108: day_of_week and hour_bucket are INTEGER columns.

        Both temporal_observations and temporal_aggregate store these as
        INTEGER so range comparisons and GROUP BY work correctly.
        """
        obs_rows = schema_db.execute(
            "PRAGMA table_info(temporal_observations)"
        ).fetchall()
        obs_col_types = {row[1]: row[2].upper() for row in obs_rows}

        assert 'day_of_week' in obs_col_types, (
            "day_of_week missing from temporal_observations"
        )
        assert 'hour_bucket' in obs_col_types, (
            "hour_bucket missing from temporal_observations"
        )
        assert obs_col_types['day_of_week'] == 'INTEGER', (
            f"day_of_week type is {obs_col_types['day_of_week']!r}, expected INTEGER"
        )
        assert obs_col_types['hour_bucket'] == 'INTEGER', (
            f"hour_bucket type is {obs_col_types['hour_bucket']!r}, expected INTEGER"
        )

        agg_rows = schema_db.execute(
            "PRAGMA table_info(temporal_aggregate)"
        ).fetchall()
        agg_col_types = {row[1]: row[2].upper() for row in agg_rows}

        assert 'day_of_week' in agg_col_types, (
            "day_of_week missing from temporal_aggregate"
        )
        assert 'hour_bucket' in agg_col_types, (
            "hour_bucket missing from temporal_aggregate"
        )
        assert agg_col_types['day_of_week'] == 'INTEGER', (
            f"temporal_aggregate.day_of_week type is {agg_col_types['day_of_week']!r}, expected INTEGER"
        )
        assert agg_col_types['hour_bucket'] == 'INTEGER', (
            f"temporal_aggregate.hour_bucket type is {agg_col_types['hour_bucket']!r}, expected INTEGER"
        )

    # ── Scenario 145 ─────────────────────────────────────────────────────────

    def test_effort_column_in_tool_profiles(self, schema_db):
        """Absorbs scenario 145: tool_capability_profiles has an effort column.

        The effort field distinguishes low/moderate/high-cost tools so the
        ACT loop can factor in execution cost during tool selection.
        """
        columns = _get_columns(schema_db, 'tool_capability_profiles')
        assert 'effort' in columns, (
            "effort column missing from tool_capability_profiles"
        )

    # ── Scenario 200 ─────────────────────────────────────────────────────────

    def test_uncertainty_tables_and_columns(self, schema_db):
        """Absorbs scenario 200: uncertainties table and reliability columns exist.

        Verifies:
        - uncertainties table has all required columns
        - reliability column is present on user_traits, episodes, semantic_concepts
        - required indexes on uncertainties are created
        """
        tables = _get_tables(schema_db)
        assert 'uncertainties' in tables, "uncertainties table missing from schema"

        uncertainty_columns = _get_columns(schema_db, 'uncertainties')
        required_cols = {
            'id',
            'memory_a_type',
            'memory_a_id',
            'memory_b_type',
            'memory_b_id',
            'uncertainty_type',
            'severity',
            'state',
            'resolution_strategy',
            'created_at',
        }
        missing_cols = required_cols - uncertainty_columns
        assert not missing_cols, (
            f"uncertainties table missing columns: {missing_cols}"
        )

        # reliability must be on all three memory tables
        for table in ('user_traits', 'episodes', 'semantic_concepts'):
            cols = _get_columns(schema_db, table)
            assert 'reliability' in cols, (
                f"reliability column missing from {table}"
            )

        # Required indexes
        uncertainty_indexes = _get_indexes(schema_db, 'uncertainties')
        required_indexes = {
            'idx_uncertainties_state',
            'idx_uncertainties_memory_a',
            'idx_uncertainties_memory_b',
            'idx_uncertainties_severity',
        }
        missing_indexes = required_indexes - uncertainty_indexes
        assert not missing_indexes, (
            f"uncertainties table missing indexes: {missing_indexes}"
        )

    # ── Scenario 201 ─────────────────────────────────────────────────────────

    def test_reliability_default_values(self, schema_db):
        """Absorbs scenario 201: reliability columns default to 'reliable'.

        Inserts a minimal row into user_traits, episodes, and semantic_concepts
        without specifying reliability and confirms the column defaults correctly.
        """
        import uuid

        # user_traits
        schema_db.execute(
            "INSERT INTO user_traits (id, trait_key, trait_value) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), 'test_key', 'test_value')
        )
        row = schema_db.execute(
            "SELECT reliability FROM user_traits WHERE trait_key = 'test_key'"
        ).fetchone()
        assert row is not None, "user_traits INSERT failed"
        assert row[0] == 'reliable', (
            f"user_traits.reliability default is {row[0]!r}, expected 'reliable'"
        )

        # episodes
        ep_id = str(uuid.uuid4())
        schema_db.execute(
            "INSERT INTO episodes "
            "(id, intent, context, action, emotion, outcome, gist, salience, freshness, topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ep_id, '{}', '{}', 'test', '{}', 'test', 'test gist', 5, 5, 'test-topic')
        )
        row = schema_db.execute(
            "SELECT reliability FROM episodes WHERE id = ?", (ep_id,)
        ).fetchone()
        assert row is not None, "episodes INSERT failed"
        assert row[0] == 'reliable', (
            f"episodes.reliability default is {row[0]!r}, expected 'reliable'"
        )

        # semantic_concepts
        sc_id = str(uuid.uuid4())
        schema_db.execute(
            "INSERT INTO semantic_concepts "
            "(id, concept_name, concept_type, definition) "
            "VALUES (?, ?, ?, ?)",
            (sc_id, 'test concept', 'fact', 'a test definition')
        )
        row = schema_db.execute(
            "SELECT reliability FROM semantic_concepts WHERE id = ?", (sc_id,)
        ).fetchone()
        assert row is not None, "semantic_concepts INSERT failed"
        assert row[0] == 'reliable', (
            f"semantic_concepts.reliability default is {row[0]!r}, expected 'reliable'"
        )

        schema_db.rollback()

    # ── Scenario 215 ─────────────────────────────────────────────────────────

    def test_tolerance_vector_baseline(self, schema_db):
        """Absorbs scenario 215: identity_vectors table exists and uncertainty_tolerance is seeded.

        schema.sql seeds all identity vectors including uncertainty_tolerance via
        INSERT OR IGNORE. This test verifies the table structure and confirms the
        seed row is present after loading the schema.
        """
        tables = _get_tables(schema_db)
        assert 'identity_vectors' in tables, (
            "identity_vectors table missing from schema"
        )

        columns = _get_columns(schema_db, 'identity_vectors')
        required_cols = {'vector_name', 'baseline_weight', 'min_cap', 'max_cap'}
        missing = required_cols - columns
        assert not missing, (
            f"identity_vectors missing columns: {missing}"
        )

        row = schema_db.execute(
            "SELECT vector_name, baseline_weight, min_cap, max_cap "
            "FROM identity_vectors WHERE vector_name = 'uncertainty_tolerance'"
        ).fetchone()
        assert row is not None, (
            "uncertainty_tolerance seed row missing from identity_vectors"
        )
        vector_name, baseline_weight, min_cap, max_cap = row
        assert vector_name == 'uncertainty_tolerance'
        assert baseline_weight == pytest.approx(0.5)
        assert min_cap == pytest.approx(0.2)
        assert max_cap == pytest.approx(0.8)

    # ── Scenario 230 ─────────────────────────────────────────────────────────

    def test_interaction_log_event_type_column(self, schema_db):
        """Absorbs scenario 230: interaction_log has event_type column.

        event_type is NOT NULL and indexed; it is the primary discriminator
        for filtering audit trail entries (e.g. 'action_gate_rejected').
        """
        tables = _get_tables(schema_db)
        assert 'interaction_log' in tables, (
            "interaction_log table missing from schema"
        )

        columns = _get_columns(schema_db, 'interaction_log')
        assert 'event_type' in columns, (
            "event_type column missing from interaction_log"
        )

        # Verify the index that makes filtering efficient also exists
        indexes = _get_indexes(schema_db, 'interaction_log')
        assert 'idx_interaction_log_event_type_created' in indexes, (
            "idx_interaction_log_event_type_created index missing"
        )

    # ── Scenario 231 ─────────────────────────────────────────────────────────

    def test_procedural_memory_reward_columns(self, schema_db):
        """Absorbs scenario 231: procedural_memory has reward_history, weight, success_rate.

        These three columns form the reinforcement-learning feedback loop:
        reward_history (JSONB) stores per-invocation rewards, weight drives
        action selection probability, and success_rate is a rolling average.
        """
        tables = _get_tables(schema_db)
        assert 'procedural_memory' in tables, (
            "procedural_memory table missing from schema"
        )

        columns = _get_columns(schema_db, 'procedural_memory')
        required = {'reward_history', 'weight', 'success_rate'}
        missing = required - columns
        assert not missing, (
            f"procedural_memory missing columns: {missing}"
        )
