"""
Unit tests for UncertaintyService — epistemic confidence tracking.

Uses a real SQLite database (tmp file) with the full schema loaded so that
actual SQL runs against the uncertainties, user_traits, episodes, and
semantic_concepts tables. No mocks of the DB layer.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from services.uncertainty_service import UncertaintyService
from services.database_service import DatabaseService


SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"

pytestmark = pytest.mark.unit


# ─── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """DatabaseService backed by a real SQLite file with the full schema."""
    db_path = str(tmp_path / "test_uncertainty.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    conn.close()

    svc = DatabaseService.__new__(DatabaseService)
    svc.db_path = db_path

    @contextmanager
    def _conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    svc.connection = _conn
    return svc


@pytest.fixture
def svc(db_service):
    return UncertaintyService(db_service)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _insert_trait(db_service, trait_id=None, reliability='reliable'):
    """Insert a minimal user_trait row for testing."""
    tid = trait_id or str(uuid.uuid4())
    with db_service.connection() as conn:
        conn.execute("""
            INSERT INTO user_traits (id, trait_key, trait_value, reliability)
            VALUES (?, ?, ?, ?)
        """, (tid, f'key_{tid[:8]}', 'value', reliability))
    return tid


def _insert_episode(db_service, episode_id=None, reliability='reliable'):
    """Insert a minimal episode row for testing."""
    eid = episode_id or str(uuid.uuid4())
    with db_service.connection() as conn:
        conn.execute("""
            INSERT INTO episodes
                (id, intent, context, action, emotion, outcome, gist,
                 salience, freshness, topic, reliability)
            VALUES (?, '{}', '{}', 'act', '{}', 'out', 'gist', 5, 5, 'test', ?)
        """, (eid, reliability))
    return eid


def _insert_concept(db_service, concept_id=None, reliability='reliable'):
    """Insert a minimal semantic_concept row for testing."""
    cid = concept_id or str(uuid.uuid4())
    with db_service.connection() as conn:
        conn.execute("""
            INSERT INTO semantic_concepts
                (id, concept_name, concept_type, definition, reliability)
            VALUES (?, ?, 'entity', 'test definition', ?)
        """, (cid, f'concept_{cid[:8]}', reliability))
    return cid


def _fetch_uncertainty(db_service, uncertainty_id):
    with db_service.connection() as conn:
        row = conn.execute(
            "SELECT * FROM uncertainties WHERE id = ?", (uncertainty_id,)
        ).fetchone()
    return dict(row) if row else None


def _get_reliability(db_service, table, record_id):
    with db_service.connection() as conn:
        row = conn.execute(
            f"SELECT reliability FROM {table} WHERE id = ?", (record_id,)
        ).fetchone()
    return row['reliability'] if row else None


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestCreateUncertainty:

    def test_basic_contradiction_creates_record(self, svc, db_service):
        """create_uncertainty inserts a row in the uncertainties table."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)

        uid = svc.create_uncertainty(
            memory_a_type='trait', memory_a_id=t1,
            memory_b_type='trait', memory_b_id=t2,
            uncertainty_type='contradiction',
            detection_context='test',
        )

        record = _fetch_uncertainty(db_service, uid)
        assert record is not None
        assert record['memory_a_id'] == t1
        assert record['memory_b_id'] == t2
        assert record['state'] == 'open'
        assert record['severity'] == 'critical'  # trait+trait

    def test_contradiction_tags_both_memories_contradicted(self, svc, db_service):
        """A contradiction sets reliability='contradicted' on both sides."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)

        svc.create_uncertainty(
            memory_a_type='trait', memory_a_id=t1,
            memory_b_type='trait', memory_b_id=t2,
            uncertainty_type='contradiction', detection_context='test',
        )

        assert _get_reliability(db_service, 'user_traits', t1) == 'contradicted'
        assert _get_reliability(db_service, 'user_traits', t2) == 'contradicted'

    def test_non_contradiction_tags_uncertain(self, svc, db_service):
        """A non-contradiction uncertainty type tags memories as 'uncertain'."""
        e1 = _insert_episode(db_service)

        svc.create_uncertainty(
            memory_a_type='episode', memory_a_id=e1,
            uncertainty_type='stale', detection_context='test',
        )

        assert _get_reliability(db_service, 'episodes', e1) == 'uncertain'

    def test_solo_uncertainty_no_b(self, svc, db_service):
        """Uncertainty with no B side is allowed (solo, severity=low)."""
        e1 = _insert_episode(db_service)

        uid = svc.create_uncertainty(
            memory_a_type='episode', memory_a_id=e1,
            uncertainty_type='unverified', detection_context='test',
        )

        record = _fetch_uncertainty(db_service, uid)
        assert record['severity'] == 'low'
        assert record['memory_b_id'] is None

    def test_returns_uuid_string(self, svc, db_service):
        """Return value is a valid UUID string."""
        e1 = _insert_episode(db_service)
        uid = svc.create_uncertainty('episode', e1, detection_context='test')
        assert uuid.UUID(uid)  # raises ValueError if not valid UUID


class TestSeverityClassification:

    @pytest.mark.parametrize("a_type,b_type,expected", [
        ('trait',   'trait',   'critical'),
        ('concept', 'trait',   'high'),
        ('trait',   'concept', 'high'),      # order-independent
        ('concept', 'concept', 'high'),
        ('episode', 'trait',   'medium'),
        ('episode', 'concept', 'medium'),
        ('episode', 'episode', 'low'),
        ('episode', None,      'low'),       # solo
    ])
    def test_severity_mapping(self, svc, a_type, b_type, expected):
        result = svc._classify_severity(a_type, b_type)
        assert result == expected


class TestResolveUncertainty:

    def test_resolve_changes_state(self, svc, db_service):
        """Resolving an uncertainty moves it to state='resolved'."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')

        result = svc.resolve_uncertainty(uid, strategy='accepted')

        assert result is True
        record = _fetch_uncertainty(db_service, uid)
        assert record['state'] == 'resolved'
        assert record['resolution_strategy'] == 'accepted'

    def test_resolve_with_winner_loser(self, svc, db_service):
        """Winner gets 'reliable', loser gets 'superseded'."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)
        uid = svc.create_uncertainty(
            'trait', t1, 'trait', t2,
            uncertainty_type='contradiction', detection_context='test',
        )

        svc.resolve_uncertainty(
            uid, strategy='superseded',
            winner_type='trait', winner_id=t1,
            loser_type='trait', loser_id=t2,
        )

        # Winner rank guard: 'reliable' (rank 4) > 'contradicted' (rank 2), write succeeds
        assert _get_reliability(db_service, 'user_traits', t1) == 'reliable'
        assert _get_reliability(db_service, 'user_traits', t2) == 'superseded'

    def test_double_resolve_returns_false(self, svc, db_service):
        """Resolving an already-resolved uncertainty returns False (idempotent)."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')

        assert svc.resolve_uncertainty(uid, strategy='accepted') is True
        assert svc.resolve_uncertainty(uid, strategy='accepted') is False


class TestGetActiveUncertainties:

    def test_returns_open_and_surfaced_excludes_resolved(self, svc, db_service):
        """get_active_uncertainties returns open+surfaced, not resolved."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)
        t3 = _insert_trait(db_service)

        uid_open = svc.create_uncertainty('trait', t1, detection_context='test')
        uid_surfaced = svc.create_uncertainty('trait', t2, detection_context='test')
        svc.mark_surfaced(uid_surfaced)
        uid_resolved = svc.create_uncertainty('trait', t3, detection_context='test')
        svc.resolve_uncertainty(uid_resolved, strategy='accepted')

        results = svc.get_active_uncertainties()
        ids = [r['id'] for r in results]

        assert uid_open in ids
        assert uid_surfaced in ids
        assert uid_resolved not in ids

    def test_severity_filter(self, svc, db_service):
        """severity_filter narrows results to matching severity only."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)
        e1 = _insert_episode(db_service)

        # critical: trait+trait
        svc.create_uncertainty('trait', t1, 'trait', t2, detection_context='test')
        # medium: episode+trait
        svc.create_uncertainty('episode', e1, 'trait', t1, detection_context='test')

        critical = svc.get_active_uncertainties(severity_filter='critical')
        assert all(r['severity'] == 'critical' for r in critical)

        medium = svc.get_active_uncertainties(severity_filter='medium')
        assert all(r['severity'] == 'medium' for r in medium)

    def test_ordered_by_severity_then_created(self, svc, db_service):
        """Results are ordered: critical before high before medium before low."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)
        c1 = _insert_concept(db_service)
        e1 = _insert_episode(db_service)

        svc.create_uncertainty('episode', e1, detection_context='test')           # low
        svc.create_uncertainty('concept', c1, 'trait', t1, detection_context='test')  # high
        svc.create_uncertainty('trait', t1, 'trait', t2, detection_context='test')    # critical

        results = svc.get_active_uncertainties()
        severities = [r['severity'] for r in results]
        rank = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        # Verify non-decreasing rank order
        assert all(rank[severities[i]] <= rank[severities[i+1]] for i in range(len(severities)-1))


class TestGetUncertaintiesForMemory:

    def test_finds_by_a_side(self, svc, db_service):
        """Finds uncertainties where the memory is on the A side."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')

        results = svc.get_uncertainties_for_memory('trait', t1)
        assert any(r['id'] == uid for r in results)

    def test_finds_by_b_side(self, svc, db_service):
        """Finds uncertainties where the memory is on the B side."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, 'trait', t2, detection_context='test')

        results = svc.get_uncertainties_for_memory('trait', t2)
        assert any(r['id'] == uid for r in results)

    def test_excludes_resolved(self, svc, db_service):
        """Resolved uncertainties are excluded."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')
        svc.resolve_uncertainty(uid, strategy='accepted')

        results = svc.get_uncertainties_for_memory('trait', t1)
        assert not any(r['id'] == uid for r in results)


class TestCheckMemoryReliability:

    def test_returns_stored_reliability(self, svc, db_service):
        """Returns the actual reliability value from the source table."""
        t1 = _insert_trait(db_service, reliability='uncertain')
        assert svc.check_memory_reliability('trait', t1) == 'uncertain'

    def test_returns_reliable_for_missing(self, svc, db_service):
        """Returns 'reliable' when the record does not exist."""
        assert svc.check_memory_reliability('trait', 'nonexistent-id') == 'reliable'

    def test_returns_reliable_for_unknown_type(self, svc, db_service):
        """Returns 'reliable' for unrecognised memory types."""
        assert svc.check_memory_reliability('unknown_type', 'any-id') == 'reliable'


class TestMarkSurfaced:

    def test_state_changes_to_surfaced(self, svc, db_service):
        """mark_surfaced transitions state from open → surfaced."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')

        svc.mark_surfaced(uid)

        record = _fetch_uncertainty(db_service, uid)
        assert record['state'] == 'surfaced'
        assert record['surfaced_count'] == 1

    def test_surfaced_count_increments(self, svc, db_service):
        """surfaced_count only increments while state == 'open' (idempotent thereafter)."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')

        svc.mark_surfaced(uid)  # open → surfaced, count=1
        svc.mark_surfaced(uid)  # already surfaced, WHERE state='open' won't match

        record = _fetch_uncertainty(db_service, uid)
        assert record['surfaced_count'] == 1


class TestResolveDecayed:

    def test_bulk_resolves_all_linked_uncertainties(self, svc, db_service):
        """resolve_decayed closes all open/surfaced uncertainties for a memory."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)

        uid1 = svc.create_uncertainty('trait', t1, detection_context='test')
        uid2 = svc.create_uncertainty('trait', t2, 'trait', t1, detection_context='test')

        count = svc.resolve_decayed('trait', t1)

        assert count == 2
        for uid in (uid1, uid2):
            record = _fetch_uncertainty(db_service, uid)
            assert record['state'] == 'resolved'
            assert record['resolution_strategy'] == 'decayed'

    def test_does_not_touch_resolved(self, svc, db_service):
        """Already-resolved uncertainties are not counted or re-touched."""
        t1 = _insert_trait(db_service)
        uid = svc.create_uncertainty('trait', t1, detection_context='test')
        svc.resolve_uncertainty(uid, strategy='accepted')

        count = svc.resolve_decayed('trait', t1)
        assert count == 0

    def test_returns_zero_for_no_matches(self, svc, db_service):
        """resolve_decayed with no linked records returns 0, not an error."""
        count = svc.resolve_decayed('trait', 'ghost-id')
        assert count == 0


class TestReliabilityRankGuard:

    def test_rank_guard_prevents_uncertain_overwriting_contradicted(self, svc, db_service):
        """'uncertain' must not overwrite an existing 'contradicted' label (rank guard)."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)

        # First call: contradiction tags both as 'contradicted'
        svc.create_uncertainty(
            'trait', t1, 'trait', t2,
            uncertainty_type='contradiction', detection_context='test',
        )
        assert _get_reliability(db_service, 'user_traits', t1) == 'contradicted'

        # Second call: 'stale' would tag as 'uncertain' — must be blocked by rank guard
        svc.create_uncertainty(
            'trait', t1,
            uncertainty_type='stale', detection_context='test',
        )
        # 'contradicted' (rank 2) is lower/worse than 'uncertain' (rank 3)
        # so guard must preserve 'contradicted'
        assert _get_reliability(db_service, 'user_traits', t1) == 'contradicted'

    def test_resolve_winner_upgrades_contradicted_to_reliable(self, svc, db_service):
        """Resolving with a winner allows upgrading 'contradicted' → 'reliable'."""
        t1 = _insert_trait(db_service)
        t2 = _insert_trait(db_service)
        uid = svc.create_uncertainty(
            'trait', t1, 'trait', t2,
            uncertainty_type='contradiction', detection_context='test',
        )
        # Both are 'contradicted' now
        svc.resolve_uncertainty(uid, strategy='winner',
                                winner_type='trait', winner_id=t1,
                                loser_type='trait', loser_id=t2)
        # resolve_uncertainty calls _set_reliability with 'reliable' — rank guard
        # allows this because 'reliable' (rank 4) > 'contradicted' (rank 2)
        assert _get_reliability(db_service, 'user_traits', t1) == 'reliable'
        assert _get_reliability(db_service, 'user_traits', t2) == 'superseded'
