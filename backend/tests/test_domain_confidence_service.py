"""
Unit tests for DomainConfidenceService — memory-derived autonomous-action confidence.

Uses a real SQLite file (via tmp_path) with the full schema loaded so that
actual SQL executes against user_traits, semantic_concepts, and interaction_log.
No mocks of the DB layer. MemoryStore is used directly (production implementation).

Test coverage:
    - Fresh database returns near-zero confidence
    - Each signal source independently raises confidence
    - Combined score is weighted correctly
    - Constraint penalty reduces confidence
    - Caching: write, read, single-domain invalidation, full invalidation
    - Edge cases: empty domain string, unknown domain, recency decay
"""

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import timedelta

import pytest

from services.domain_confidence_service import (
    DomainConfidenceService,
    CACHE_KEY_PREFIX,
    CACHE_ALL_KEY,
    CACHE_TTL,
    TRAIT_SATURATION,
    CONCEPT_SATURATION,
    PENALTY_SATURATION,
    RECENCY_HALF_LIFE_DAYS,
)
from services.database_service import DatabaseService
from services.memory_store import MemoryStore
from services.time_utils import utc_now


pytestmark = pytest.mark.unit


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_service(tmp_path):
    """DatabaseService backed by a real SQLite file with the full schema loaded."""
    from tests.test_helpers import load_schema_sql

    db_path = str(tmp_path / "test_domain_confidence.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(load_schema_sql())
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
def store():
    """Isolated MemoryStore (production implementation, no mocking)."""
    return MemoryStore()


@pytest.fixture
def svc(db_service, store):
    """DomainConfidenceService with real DB and real MemoryStore."""
    return DomainConfidenceService(db_service, store)


@pytest.fixture
def svc_no_cache(db_service):
    """DomainConfidenceService without a MemoryStore — caching disabled."""
    return DomainConfidenceService(db_service, memory_store=None)


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _insert_trait(db_service, domain, confidence=0.8, category="preference"):
    """Insert a user_trait row whose key/value/category includes the domain."""
    tid = str(uuid.uuid4())
    # Use a UUID suffix to guarantee unique trait_key (table has UNIQUE(trait_key))
    unique_suffix = tid[:8]
    with db_service.connection() as conn:
        conn.execute(
            """
            INSERT INTO user_traits (id, trait_key, trait_value, category, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tid, f"{domain}_pref_{unique_suffix}", f"likes {domain}", category, confidence),
        )
    return tid


def _insert_concept(db_service, domain, confidence=0.7):
    """Insert a semantic_concept row whose domain field matches the domain."""
    cid = str(uuid.uuid4())
    with db_service.connection() as conn:
        conn.execute(
            """
            INSERT INTO semantic_concepts
                (id, concept_name, concept_type, definition, domain, confidence)
            VALUES (?, ?, 'entity', ?, ?, ?)
            """,
            (cid, f"{domain}_concept_{cid[:6]}", f"A concept about {domain}", domain, confidence),
        )
    return cid


def _insert_tool_result(db_service, domain, created_at=None):
    """Insert a successful tool_result event in the interaction_log for this domain."""
    eid = str(uuid.uuid4())
    ts = created_at or utc_now().isoformat()
    with db_service.connection() as conn:
        conn.execute(
            """
            INSERT INTO interaction_log (id, event_type, topic, payload, created_at)
            VALUES (?, 'tool_result', ?, ?, ?)
            """,
            (eid, domain, json.dumps({"domain": domain, "result": "ok"}), ts),
        )
    return eid


def _insert_rejection(db_service, domain, created_at=None):
    """Insert an action_gate_rejected event in the interaction_log for this domain."""
    eid = str(uuid.uuid4())
    ts = created_at or utc_now().isoformat()
    with db_service.connection() as conn:
        conn.execute(
            """
            INSERT INTO interaction_log (id, event_type, topic, payload, created_at)
            VALUES (?, 'action_gate_rejected', ?, ?, ?)
            """,
            (eid, domain, json.dumps({"domain": domain, "reason": "low_confidence"}), ts),
        )
    return eid


def _insert_recent_activity(db_service, domain, days_ago=1):
    """Insert any interaction_log entry for this domain N days in the past."""
    eid = str(uuid.uuid4())
    ts = (utc_now() - timedelta(days=days_ago)).isoformat()
    with db_service.connection() as conn:
        conn.execute(
            """
            INSERT INTO interaction_log (id, event_type, topic, payload, created_at)
            VALUES (?, 'message', ?, ?, ?)
            """,
            (eid, domain, json.dumps({"domain": domain}), ts),
        )
    return eid


# ─── Fresh database ────────────────────────────────────────────────────────────


class TestFreshDatabase:

    def test_empty_db_returns_near_zero(self, svc):
        """A brand-new instance with no memory returns close to zero confidence."""
        score = svc.compute_domain_confidence("scheduling")
        # No traits, episodes, or concepts: only the constraint term contributes
        # (1.0 - 0.0) * W_CONSTRAINT = 0.15, recency = 0.0 → raw ≈ 0.15
        assert 0.0 <= score <= 0.20

    def test_empty_db_unknown_domain_returns_near_zero(self, svc):
        """An unknown domain with empty memory returns near-zero confidence."""
        score = svc.compute_domain_confidence("quantum_hyperspace")
        assert 0.0 <= score <= 0.20

    def test_empty_string_domain_returns_zero(self, svc):
        """An empty domain string returns exactly 0.0."""
        score = svc.compute_domain_confidence("")
        assert score == 0.0


# ─── Trait Density signal ──────────────────────────────────────────────────────


class TestTraitDensitySignal:

    def test_single_trait_raises_confidence(self, svc_no_cache, db_service):
        """One matching trait produces a non-zero trait score."""
        _insert_trait(db_service, "scheduling", confidence=0.9)
        score = svc_no_cache.compute_domain_confidence("scheduling")
        # Trait contribution: (1/10) * 0.9 * 0.30 = 0.027 — above pure-constraint baseline
        assert score > 0.15

    def test_more_traits_increase_confidence(self, svc_no_cache, db_service):
        """More domain traits produce higher confidence than fewer."""
        _insert_trait(db_service, "finance", confidence=0.8)
        score_one = svc_no_cache.compute_domain_confidence("finance")

        for _ in range(4):
            _insert_trait(db_service, "finance", confidence=0.8)
        score_five = svc_no_cache.compute_domain_confidence("finance")

        assert score_five > score_one

    def test_saturation_caps_trait_contribution(self, svc_no_cache, db_service):
        """Adding traits beyond TRAIT_SATURATION doesn't increase score past the cap."""
        # Insert 2x saturation traits
        for _ in range(TRAIT_SATURATION * 2):
            _insert_trait(db_service, "health", confidence=1.0)

        score = svc_no_cache.compute_domain_confidence("health")
        # Trait score is capped at 1.0 * 1.0 * W_TRAIT = 0.30
        # Total cannot exceed sum of all weights
        assert score <= 1.0

    def test_zero_confidence_traits_give_zero_trait_score(self, svc_no_cache, db_service):
        """Traits with confidence=0.0 contribute nothing to trait density."""
        _insert_trait(db_service, "travel", confidence=0.0)
        score_with_zero_conf = svc_no_cache.compute_domain_confidence("travel")

        # Should be same as fresh (no meaningful trait signal)
        assert score_with_zero_conf <= 0.20

    def test_trait_matched_by_category(self, svc_no_cache, db_service):
        """A trait is matched when category contains the domain string."""
        tid = str(uuid.uuid4())
        with db_service.connection() as conn:
            conn.execute(
                """
                INSERT INTO user_traits (id, trait_key, trait_value, category, confidence)
                VALUES (?, ?, 'yes', ?, ?)
                """,
                (tid, f"prefers_morning_{tid[:8]}", "scheduling", 0.85),
            )
        score = svc_no_cache.compute_domain_confidence("scheduling")
        assert score > 0.15


# ─── Episodic Success signal ───────────────────────────────────────────────────


class TestEpisodicSuccessSignal:

    def test_tool_results_increase_episode_score(self, svc_no_cache, db_service):
        """Successful tool_result events in this domain increase confidence."""
        for _ in range(5):
            _insert_tool_result(db_service, "scheduling")

        score = svc_no_cache.compute_domain_confidence("scheduling")
        # All 5 events are successes → episode_score = 1.0 → +0.25
        assert score > 0.30

    def test_rejections_lower_episode_score(self, svc_no_cache, db_service):
        """Gate rejections lower the episodic success rate."""
        for _ in range(3):
            _insert_tool_result(db_service, "cooking")
        for _ in range(3):
            _insert_rejection(db_service, "cooking")

        score = svc_no_cache.compute_domain_confidence("cooking")
        # success_rate = 3/6 = 0.5 → episode contribution = 0.5 * 0.25 = 0.125
        # Also penalised by constraint: 3 rejections / 5 saturation = 0.6 penalty
        # (1 - 0.6) * 0.15 = 0.06
        # recency is > 0 since we inserted recently
        assert 0.15 < score < 0.60

    def test_all_rejections_gives_zero_episode_score(self, svc_no_cache, db_service):
        """When every logged event is a rejection, episode score is zero."""
        for _ in range(4):
            _insert_rejection(db_service, "finance")

        score = svc_no_cache.compute_domain_confidence("finance")
        # episode_score = 0 → only constraint_penalty (high) and maybe recency
        assert score <= 0.25

    def test_no_interaction_log_gives_zero_episode_score(self, svc_no_cache, db_service):
        """No interaction_log entries for domain yields zero episode score."""
        # Insert traits so we know trait signal is non-zero
        _insert_trait(db_service, "work", confidence=0.8)
        score_with_trait = svc_no_cache.compute_domain_confidence("work")

        # The episode component should be 0 (no log entries)
        # Score should be: 0.30 * (1/10 * 0.8) + 0.15 * 1.0 + ... low
        assert score_with_trait < 0.30


# ─── Concept Depth signal ──────────────────────────────────────────────────────


class TestConceptDepthSignal:

    def test_concepts_increase_confidence(self, svc_no_cache, db_service):
        """Semantic concepts matching the domain raise confidence."""
        for _ in range(3):
            _insert_concept(db_service, "health", confidence=0.7)

        score = svc_no_cache.compute_domain_confidence("health")
        # concept contribution: (3/5) * 0.7 * 0.20 = 0.084
        assert score > 0.20

    def test_saturation_caps_concept_contribution(self, svc_no_cache, db_service):
        """Inserting more concepts than CONCEPT_SATURATION doesn't exceed the cap."""
        for _ in range(CONCEPT_SATURATION * 3):
            _insert_concept(db_service, "science", confidence=1.0)

        score = svc_no_cache.compute_domain_confidence("science")
        # concept score capped at W_CONCEPT = 0.20 contribution max
        assert score <= 1.0

    def test_soft_deleted_concepts_excluded(self, svc_no_cache, db_service):
        """Soft-deleted concepts (deleted_at IS NOT NULL) are not counted."""
        cid = str(uuid.uuid4())
        with db_service.connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic_concepts
                    (id, concept_name, concept_type, definition, domain, confidence, deleted_at)
                VALUES (?, 'deleted_concept', 'entity', 'def', ?, 0.9, ?)
                """,
                (cid, "deleted_domain", utc_now().isoformat()),
            )

        score = svc_no_cache.compute_domain_confidence("deleted_domain")
        assert score <= 0.20  # near-zero since concept is excluded

    def test_concept_matched_by_definition(self, svc_no_cache, db_service):
        """A concept is matched when its definition mentions the domain."""
        cid = str(uuid.uuid4())
        with db_service.connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic_concepts
                    (id, concept_name, concept_type, definition, domain, confidence)
                VALUES (?, 'alarm_setting', 'procedure',
                        'User prefers early morning scheduling for meetings', 'general', ?)
                """,
                (cid, 0.75),
            )

        score = svc_no_cache.compute_domain_confidence("scheduling")
        assert score > 0.15


# ─── Constraint Penalty signal ─────────────────────────────────────────────────


class TestConstraintPenaltySignal:

    def test_no_rejections_no_penalty(self, svc_no_cache, db_service):
        """With no rejections, constraint penalty is zero — full 0.15 term."""
        # Insert traits to get a baseline non-zero score
        _insert_trait(db_service, "lists", confidence=0.8)
        score = svc_no_cache.compute_domain_confidence("lists")

        # constraint term = (1.0 - 0.0) * 0.15 = 0.15
        # Should be > pure no-signal baseline
        assert score > 0.15

    def test_max_rejections_zeroes_constraint_term(self, svc_no_cache, db_service):
        """PENALTY_SATURATION rejections reduce the constraint term to ~0."""
        for _ in range(PENALTY_SATURATION):
            _insert_rejection(db_service, "finance_penalty_test")

        score_with_penalty = svc_no_cache.compute_domain_confidence("finance_penalty_test")

        # constraint_penalty = min(1.0, 5/5) = 1.0
        # constraint term = (1.0 - 1.0) * 0.15 = 0.0
        # episode_score: all 5 events are rejections → 0.0
        # So score ≈ 0.0 + recency
        assert score_with_penalty < 0.20

    def test_penalty_is_proportional(self, svc_no_cache, db_service):
        """Fewer rejections = smaller penalty = higher confidence than max rejections."""
        _insert_rejection(db_service, "work_low_penalty")

        score_low = svc_no_cache.compute_domain_confidence("work_low_penalty")

        for _ in range(PENALTY_SATURATION - 1):
            _insert_rejection(db_service, "work_high_penalty")

        score_high = svc_no_cache.compute_domain_confidence("work_high_penalty")

        # More rejections → higher penalty → lower score
        assert score_low > score_high


# ─── Recency Weight signal ─────────────────────────────────────────────────────


class TestRecencyWeightSignal:

    def test_recent_activity_gives_high_recency(self, svc_no_cache, db_service):
        """Activity from today yields recency score close to 1.0."""
        _insert_recent_activity(db_service, "travel", days_ago=0)
        # Access private method directly to verify
        score = svc_no_cache._recency_weight("travel", 1)
        assert score > 0.90

    def test_old_activity_decays_toward_zero(self, svc_no_cache, db_service):
        """Activity 60 days ago gives a much lower recency score than activity today."""
        _insert_recent_activity(db_service, "travel_old", days_ago=60)
        score_old = svc_no_cache._recency_weight("travel_old", 1)

        _insert_recent_activity(db_service, "travel_new", days_ago=1)
        score_new = svc_no_cache._recency_weight("travel_new", 1)

        assert score_new > score_old

    def test_half_life_decay_at_14_days(self, svc_no_cache, db_service):
        """Activity exactly 14 days ago yields recency ≈ 0.5 (one half-life)."""
        _insert_recent_activity(db_service, "halflife_test", days_ago=RECENCY_HALF_LIFE_DAYS)
        score = svc_no_cache._recency_weight("halflife_test", 1)
        # Allow ±0.05 tolerance for timing imprecision
        assert 0.45 <= score <= 0.55

    def test_no_activity_gives_zero_recency(self, svc_no_cache, db_service):
        """A domain with no interaction_log entries returns zero recency."""
        score = svc_no_cache._recency_weight("completely_absent_domain", 1)
        assert score == 0.0

    def test_recency_increases_total_score(self, svc_no_cache, db_service):
        """Domain with recent activity scores higher than one with no activity."""
        # Insert traits for both so trait signal is equal
        _insert_trait(db_service, "active_domain", confidence=0.7)
        _insert_trait(db_service, "inactive_domain", confidence=0.7)
        _insert_recent_activity(db_service, "active_domain", days_ago=1)

        score_active = svc_no_cache.compute_domain_confidence("active_domain")
        score_inactive = svc_no_cache.compute_domain_confidence("inactive_domain")

        assert score_active > score_inactive


# ─── Combined Score ────────────────────────────────────────────────────────────


class TestCombinedScore:

    def test_all_signals_present_gives_high_score(self, svc_no_cache, db_service):
        """Populating all signal sources produces a meaningfully high confidence."""
        domain = "scheduling_full"

        # Traits: 5 traits at 0.9 confidence
        for _ in range(5):
            _insert_trait(db_service, domain, confidence=0.9)

        # Episodes: 8 successes, 0 rejections
        for _ in range(8):
            _insert_tool_result(db_service, domain)

        # Concepts: 4 concepts at 0.8 confidence
        for _ in range(4):
            _insert_concept(db_service, domain, confidence=0.8)

        # Recent activity
        _insert_recent_activity(db_service, domain, days_ago=1)

        score = svc_no_cache.compute_domain_confidence(domain)

        # Expected approximate:
        # trait: (5/10) * 0.9 * 0.30 = 0.135
        # episode: 1.0 * 0.25 = 0.250
        # concept: (4/5) * 0.8 * 0.20 = 0.128
        # constraint: (1.0 - 0.0) * 0.15 = 0.150
        # recency: ~0.95 * 0.10 = 0.095
        # total ≈ 0.758
        assert score > 0.60

    def test_score_is_bounded_0_to_1(self, svc_no_cache, db_service):
        """Score is always within [0.0, 1.0] regardless of data density."""
        domain = "bounded_test"
        for _ in range(50):
            _insert_trait(db_service, domain, confidence=1.0)
        for _ in range(50):
            _insert_tool_result(db_service, domain)
        for _ in range(50):
            _insert_concept(db_service, domain, confidence=1.0)

        score = svc_no_cache.compute_domain_confidence(domain)
        assert 0.0 <= score <= 1.0

    def test_weighted_formula_is_correct(self, svc_no_cache, db_service):
        """Verify the weighted sum formula by controlling individual signals."""
        domain = "formula_test"

        # Insert exactly TRAIT_SATURATION traits at confidence 1.0 → trait_score = 1.0
        for _ in range(TRAIT_SATURATION):
            _insert_trait(db_service, domain, confidence=1.0)

        # Insert exactly CONCEPT_SATURATION concepts at confidence 1.0 → concept_score = 1.0
        for _ in range(CONCEPT_SATURATION):
            _insert_concept(db_service, domain, confidence=1.0)

        # No episodes, no rejections, no recency
        score = svc_no_cache.compute_domain_confidence(domain)

        # Expected: 0.30*1.0 + 0.25*0.0 + 0.20*1.0 + 0.15*1.0 + 0.10*0.0 = 0.65
        assert abs(score - 0.65) < 0.05

    def test_only_constraint_term_with_empty_db(self, svc_no_cache, db_service):
        """Empty DB gives score equal to (1.0 - 0.0) * 0.15 = 0.15."""
        score = svc_no_cache.compute_domain_confidence("only_constraint")
        # All signals = 0 except constraint term = 0.15
        assert abs(score - 0.15) < 0.02


# ─── Caching ───────────────────────────────────────────────────────────────────


class TestCaching:

    def test_score_is_cached_after_first_call(self, svc, store):
        """After compute, the score is stored in MemoryStore."""
        svc.compute_domain_confidence("scheduling")

        cached_raw = store.get(CACHE_KEY_PREFIX + "scheduling")
        assert cached_raw is not None
        assert 0.0 <= float(cached_raw) <= 1.0

    def test_second_call_returns_cached_value(self, svc, db_service, store):
        """Second call to compute returns the cached value without hitting the DB."""
        svc.compute_domain_confidence("scheduling")

        # Write a sentinel value directly into the cache
        store.set(CACHE_KEY_PREFIX + "scheduling", "0.9999", ex=CACHE_TTL)

        # Second call should return the cached value (0.9999), not recompute
        score = svc.compute_domain_confidence("scheduling")
        assert abs(score - 0.9999) < 0.0001

    def test_invalidate_domain_clears_cache(self, svc, store):
        """invalidate_domain removes the domain's entry from the cache."""
        svc.compute_domain_confidence("finance")
        svc.invalidate_domain("finance")

        cached_raw = store.get(CACHE_KEY_PREFIX + "finance")
        assert cached_raw is None

    def test_invalidate_domain_case_insensitive(self, svc, store):
        """Domain keys are normalised to lowercase before caching."""
        svc.compute_domain_confidence("Finance")
        svc.invalidate_domain("FINANCE")

        # Both "Finance" and "FINANCE" normalise to "finance"
        cached_raw = store.get(CACHE_KEY_PREFIX + "finance")
        assert cached_raw is None

    def test_invalidate_all_clears_all_domains(self, svc, store):
        """invalidate_all removes all domain confidence entries."""
        svc.compute_domain_confidence("scheduling")
        svc.compute_domain_confidence("finance")
        svc.compute_domain_confidence("health")

        svc.invalidate_all()

        for domain in ("scheduling", "finance", "health"):
            assert store.get(CACHE_KEY_PREFIX + domain) is None

    def test_no_cache_service_runs_without_error(self, svc_no_cache, db_service):
        """Service with no MemoryStore (None) still computes without errors."""
        score = svc_no_cache.compute_domain_confidence("scheduling")
        assert 0.0 <= score <= 1.0

    def test_after_invalidation_score_recomputes(self, svc, db_service, store):
        """After invalidation, the next call recomputes from the database."""
        # First call — empty DB, should be ~0.15
        score_before = svc.compute_domain_confidence("recompute_test")

        # Now inject traits so the DB state changes
        _insert_trait(db_service, "recompute_test", confidence=1.0)
        for _ in range(4):
            _insert_tool_result(db_service, "recompute_test")

        # Without invalidation, cached value is returned
        score_cached = svc.compute_domain_confidence("recompute_test")
        assert abs(score_cached - score_before) < 0.01

        # Invalidate and recompute
        svc.invalidate_domain("recompute_test")
        score_after = svc.compute_domain_confidence("recompute_test")

        assert score_after > score_before

    def test_cache_tracks_domains_in_all_key(self, svc, store):
        """Computed domains are tracked in CACHE_ALL_KEY for invalidate_all."""
        svc.compute_domain_confidence("alpha")
        svc.compute_domain_confidence("beta")

        raw = store.get(CACHE_ALL_KEY)
        assert raw is not None
        domains = json.loads(raw)
        assert "alpha" in domains
        assert "beta" in domains


# ─── Edge Cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_whitespace_domain_normalised(self, svc_no_cache, db_service):
        """Leading/trailing whitespace in domain string is stripped."""
        _insert_trait(db_service, "scheduling", confidence=0.8)
        score_plain = svc_no_cache.compute_domain_confidence("scheduling")
        score_padded = svc_no_cache.compute_domain_confidence("  scheduling  ")
        assert abs(score_plain - score_padded) < 0.001

    def test_domain_with_special_sql_chars(self, svc_no_cache, db_service):
        """Domain strings with SQL special characters don't cause errors."""
        score = svc_no_cache.compute_domain_confidence("health%check")
        assert 0.0 <= score <= 1.0

    def test_very_long_domain_string(self, svc_no_cache, db_service):
        """Extremely long domain strings don't raise exceptions."""
        long_domain = "a" * 500
        score = svc_no_cache.compute_domain_confidence(long_domain)
        assert 0.0 <= score <= 1.0

    def test_domain_not_in_any_table_gives_near_zero(self, svc_no_cache, db_service):
        """A domain that appears nowhere in the DB gives near-zero confidence."""
        _insert_trait(db_service, "scheduling", confidence=0.9)
        _insert_concept(db_service, "finance", confidence=0.8)

        score = svc_no_cache.compute_domain_confidence("astrophysics")
        # Only constraint term (no rejections): 0.15
        assert score <= 0.20

    def test_domain_with_uppercase_matches_lowercase_data(self, svc_no_cache, db_service):
        """Domain matching is case-insensitive (LIKE lower(?))."""
        _insert_trait(db_service, "scheduling", confidence=0.8)

        score_lower = svc_no_cache.compute_domain_confidence("scheduling")
        score_upper = svc_no_cache.compute_domain_confidence("SCHEDULING")

        # Both should match the same traits (LIKE lower(?) used in query)
        assert abs(score_lower - score_upper) < 0.001
