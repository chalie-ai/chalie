"""
Tests for CognitiveReflexService — learned fast path via semantic abstraction.

Covers: heuristic pre-screen, cold start, clustering, generalization,
activation thresholds, correction feedback, auto-disable, warmth gate,
pipeline utility, cluster isolation, shadow validation.
"""

import json
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from services.cognitive_reflex_service import (
    CognitiveReflexService,
    ReflexResult,
    CLUSTER_DISTANCE_THRESHOLD,
    MIN_CONFIDENCE,
    MIN_OBSERVATIONS,
    MIN_SUCCESSES,
    MAX_FAILURE_RATE,
    MAX_STALE_DAYS,
    ROLLING_AVG_CAP,
    SHADOW_VALIDATION_RATE,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_embedding(seed=42, dim=768):
    """Generate a deterministic L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_similar_embedding(base, noise_scale=0.05, seed=99):
    """Generate an embedding similar to base (cosine distance < 0.35)."""
    rng = np.random.RandomState(seed)
    base_arr = np.array(base, dtype=np.float32)
    noise = rng.randn(len(base)).astype(np.float32) * noise_scale
    similar = base_arr + noise
    similar /= np.linalg.norm(similar)
    return similar.tolist()


def _make_distant_embedding(base, seed=77):
    """Generate an embedding distant from base (cosine distance > 0.35)."""
    rng = np.random.RandomState(seed)
    # Orthogonal-ish vector
    vec = rng.randn(len(base)).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


from datetime import datetime, timezone, timedelta


def _make_cluster_row(
    cluster_id=1, embedding=None, times_seen=10, times_unnecessary=9,
    times_activated=5, times_succeeded=4, times_failed=0,
    sample_queries=None, last_seen=None, last_activated=None,
    distance=0.1,
):
    """Build a tuple matching the _find_matching_reflex query result."""
    if embedding is None:
        embedding = _make_embedding()
    if sample_queries is None:
        sample_queries = ['2+2', '3*5']
    if last_seen is None:
        last_seen = datetime.now(timezone.utc)
    return (
        cluster_id,
        embedding,
        times_seen,
        times_unnecessary,
        times_activated,
        times_succeeded,
        times_failed,
        sample_queries,
        last_seen,
        last_activated,
        distance,
    )


# ─── Test Class ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCognitiveReflexService:

    @pytest.fixture
    def service(self, mock_db_rows):
        """Create service with mocked DB and Redis."""
        db, cursor = mock_db_rows
        redis = MagicMock()
        redis.get.return_value = None
        return CognitiveReflexService(db=db, redis=redis)

    @pytest.fixture
    def cursor(self, mock_db_rows):
        """Expose cursor for setting return values."""
        _, cursor = mock_db_rows
        return cursor

    @pytest.fixture
    def redis(self, service):
        return service.redis

    # ── Heuristic pre-screen ──────────────────────────────────────────────

    def test_candidate_simple_question(self, service):
        """Simple factual questions should pass pre-screen."""
        assert service._is_candidate("What is 2+2?", 0.0) is True
        assert service._is_candidate("Define entropy", 0.0) is True
        assert service._is_candidate("How many legs does a spider have?", 0.0) is True
        assert service._is_candidate("Capital of France?", 0.0) is True

    def test_reject_anaphoric(self, service):
        """Context-dependent pronouns should be rejected."""
        assert service._is_candidate("What does it mean?", 0.0) is False
        assert service._is_candidate("Tell me more about that", 0.0) is False

    def test_reject_personal_memory(self, service):
        """Personal/memory queries should be rejected."""
        assert service._is_candidate("My password is...", 0.0) is False
        assert service._is_candidate("What did I say?", 0.0) is False

    def test_reject_conversational_ref(self, service):
        """References to prior conversation should be rejected."""
        assert service._is_candidate("What did you say earlier?", 0.0) is False
        assert service._is_candidate("Remember when we talked?", 0.0) is False

    def test_reject_tool_action(self, service):
        """Tool/action requests should be rejected."""
        assert service._is_candidate("Search for restaurants", 0.0) is False
        assert service._is_candidate("Can you remind me later?", 0.0) is False
        assert service._is_candidate("Find a good recipe", 0.0) is False

    def test_reject_freshness(self, service):
        """Real-time data requests should be rejected."""
        assert service._is_candidate("What's the latest news?", 0.0) is False
        assert service._is_candidate("What happened today?", 0.0) is False

    def test_reject_reasoning(self, service):
        """Reasoning/deliberation requests should be rejected."""
        assert service._is_candidate("Why is the sky blue?", 0.0) is False
        assert service._is_candidate("Explain quantum mechanics", 0.0) is False

    def test_reject_comparative(self, service):
        """Comparative queries should be rejected."""
        assert service._is_candidate("Difference between Python and Java?", 0.0) is False

    def test_reject_uncertainty(self, service):
        """Uncertainty/estimation should be rejected."""
        assert service._is_candidate("About how many stars?", 0.0) is False

    def test_reject_too_long(self, service):
        """Multi-word queries exceeding 15 words should be rejected."""
        long_query = "This is a very long query that has way too many words to be a simple question for reflex"
        assert service._is_candidate(long_query, 0.0) is False

    def test_reject_multi_clause(self, service):
        """Multi-clause compound requests should be rejected."""
        assert service._is_candidate("Do X and also Y", 0.0) is False

    def test_reject_high_warmth(self, service):
        """Active conversation (high warmth) should be rejected."""
        assert service._is_candidate("What is 2+2?", 0.6) is False
        assert service._is_candidate("What is 2+2?", 0.51) is False

    def test_reject_url(self, service):
        """URLs should be rejected (need ACT mode)."""
        assert service._is_candidate("Check https://example.com", 0.0) is False

    def test_reject_empty(self, service):
        """Empty/whitespace should be rejected."""
        assert service._is_candidate("", 0.0) is False
        assert service._is_candidate("   ", 0.0) is False

    def test_warmth_boundary(self, service):
        """Warmth at exactly 0.5 should pass (boundary)."""
        assert service._is_candidate("What is 2+2?", 0.5) is True
        assert service._is_candidate("What is 2+2?", 0.50) is True

    # ── Cold start ────────────────────────────────────────────────────────

    @patch('services.embedding_service.get_embedding_service')
    def test_cold_start_no_clusters(self, mock_emb, service, cursor):
        """With zero clusters in DB, check returns can_activate=False."""
        mock_emb.return_value.generate_embedding.return_value = _make_embedding()
        cursor.fetchone.return_value = None  # No matching cluster

        result = service.check("What is 2+2?", 0.0)

        assert result.is_candidate is True
        assert result.can_activate is False
        assert result.cluster_id is None
        assert result.embedding is not None

    # ── Semantic reflex lookup ────────────────────────────────────────────

    @patch('services.embedding_service.get_embedding_service')
    def test_matching_cluster_below_threshold(self, mock_emb, service, cursor):
        """Cluster with insufficient observations → can_activate=False."""
        emb = _make_embedding()
        mock_emb.return_value.generate_embedding.return_value = emb

        # Cluster exists but only seen 3 times (< MIN_OBSERVATIONS)
        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=3, times_unnecessary=3,
            times_activated=0, times_succeeded=0, times_failed=0,
        )

        result = service.check("What is 2+2?", 0.0)

        assert result.is_candidate is True
        assert result.can_activate is False
        assert result.cluster_id == 1

    @patch('services.embedding_service.get_embedding_service')
    def test_matching_cluster_activates(self, mock_emb, service, cursor):
        """Mature cluster meets all criteria → can_activate=True."""
        emb = _make_embedding()
        mock_emb.return_value.generate_embedding.return_value = emb

        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=10, times_unnecessary=9,
            times_activated=5, times_succeeded=4, times_failed=0,
        )

        result = service.check("What is 7+8?", 0.0)

        assert result.is_candidate is True
        assert result.can_activate is True
        assert result.confidence == 0.9  # 9/10
        assert result.cluster_id == 1

    # ── Activation threshold edge cases ───────────────────────────────────

    @patch('services.embedding_service.get_embedding_service')
    def test_low_confidence_blocks_activation(self, mock_emb, service, cursor):
        """Cluster with < 85% unnecessary rate → can_activate=False."""
        mock_emb.return_value.generate_embedding.return_value = _make_embedding()

        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=10, times_unnecessary=7,  # 70% < 85%
            times_activated=5, times_succeeded=4, times_failed=0,
        )

        result = service.check("What is 2+2?", 0.0)
        assert result.can_activate is False

    @patch('services.embedding_service.get_embedding_service')
    def test_insufficient_successes_blocks(self, mock_emb, service, cursor):
        """Cluster with < 3 successes → can_activate=False."""
        mock_emb.return_value.generate_embedding.return_value = _make_embedding()

        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=10, times_unnecessary=9,
            times_activated=3, times_succeeded=2, times_failed=0,
        )

        result = service.check("What is 2+2?", 0.0)
        assert result.can_activate is False

    @patch('services.embedding_service.get_embedding_service')
    def test_high_failure_rate_blocks(self, mock_emb, service, cursor):
        """Cluster with > 20% failure rate → can_activate=False (auto-disable)."""
        mock_emb.return_value.generate_embedding.return_value = _make_embedding()

        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=10, times_unnecessary=9,
            times_activated=10, times_succeeded=5, times_failed=3,  # 30% > 20%
        )

        result = service.check("What is 2+2?", 0.0)
        assert result.can_activate is False

    @patch('services.embedding_service.get_embedding_service')
    def test_stale_cluster_blocks(self, mock_emb, service, cursor):
        """Cluster not seen in > 30 days → can_activate=False."""
        mock_emb.return_value.generate_embedding.return_value = _make_embedding()

        old_date = datetime.now(timezone.utc) - timedelta(days=45)
        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=10, times_unnecessary=9,
            times_activated=5, times_succeeded=4, times_failed=0,
            last_seen=old_date,
        )

        result = service.check("What is 2+2?", 0.0)
        assert result.can_activate is False

    # ── Observation recording (clustering) ────────────────────────────────

    @patch('services.embedding_service.get_embedding_service')
    def test_record_creates_new_cluster(self, mock_emb, service, cursor):
        """When no matching cluster, record_observation creates a new one."""
        emb = _make_embedding()
        cursor.fetchone.side_effect = [None, (42,)]  # No match, then INSERT RETURNING

        service.record_observation("What is 2+2?", emb, was_useful=False)

        # Verify INSERT was called
        calls = cursor.execute.call_args_list
        insert_call = [c for c in calls if 'INSERT INTO cognitive_reflexes' in str(c)]
        assert len(insert_call) == 1

    @patch('services.embedding_service.get_embedding_service')
    def test_record_merges_into_existing(self, mock_emb, service, cursor):
        """When matching cluster exists, record_observation updates it."""
        emb = _make_embedding(seed=42)

        # First call: _find_matching_reflex returns a match
        cursor.fetchone.return_value = _make_cluster_row(
            times_seen=5, times_unnecessary=4,
        )

        service.record_observation("What is 3*5?", emb, was_useful=False)

        # Verify UPDATE was called
        calls = cursor.execute.call_args_list
        update_call = [c for c in calls if 'UPDATE cognitive_reflexes' in str(c)]
        assert len(update_call) >= 1

    # ── Correction feedback ───────────────────────────────────────────────

    def test_correction_detected(self, service):
        """Correction patterns should be detected."""
        assert service._is_correction("No, that's wrong") is True
        assert service._is_correction("Try again please") is True
        assert service._is_correction("That's not what I asked") is True
        assert service._is_correction("You misunderstood me") is True

    def test_rephrase_detected(self, service):
        """Rephrase patterns should be detected as corrections."""
        assert service._is_correction("What I meant was something else") is True
        assert service._is_correction("Let me rephrase that") is True

    def test_no_correction(self, service):
        """Normal follow-up messages should not be detected as corrections."""
        assert service._is_correction("Thanks!") is False
        assert service._is_correction("What about addition?") is False
        assert service._is_correction("Cool") is False

    def test_pending_validation_correction(self, service, cursor, redis):
        """Pending validation with correction → times_failed incremented."""
        redis.get.return_value = "42"

        service.check_pending_validation("thread-1", "No, that's wrong")

        # Should have deleted the pending key
        redis.delete.assert_called_once_with("reflex:pending:thread-1")

        # Should have incremented times_failed
        calls = cursor.execute.call_args_list
        failed_update = [c for c in calls if 'times_failed' in str(c)]
        assert len(failed_update) >= 1

    def test_pending_validation_no_correction(self, service, cursor, redis):
        """Pending validation without correction → times_succeeded incremented."""
        redis.get.return_value = "42"

        service.check_pending_validation("thread-1", "Thanks, got it!")

        # Should have incremented times_succeeded
        calls = cursor.execute.call_args_list
        succeeded_update = [c for c in calls if 'times_succeeded' in str(c)]
        assert len(succeeded_update) >= 1

    def test_pending_validation_no_pending(self, service, redis):
        """No pending validation → no-op."""
        redis.get.return_value = None

        # Should not raise
        service.check_pending_validation("thread-1", "Hello")
        redis.delete.assert_not_called()

    # ── Pipeline utility evaluation ───────────────────────────────────────

    def test_pipeline_useful_for_act_mode(self, service):
        """ACT mode always means pipeline was useful."""
        triage = MagicMock()
        triage.mode = 'ACT'
        triage.tools = ['web_search']
        triage.skills = []
        triage.confidence_internal = 0.3
        assert service.evaluate_pipeline_utility(triage, None) is True

    def test_pipeline_useful_for_clarify(self, service):
        """CLARIFY mode always means pipeline was useful."""
        triage = MagicMock()
        triage.mode = 'CLARIFY'
        triage.tools = []
        triage.skills = []
        triage.confidence_internal = 0.9
        assert service.evaluate_pipeline_utility(triage, None) is True

    def test_pipeline_useful_with_tools(self, service):
        """RESPOND with tools → pipeline was useful."""
        triage = MagicMock()
        triage.mode = 'RESPOND'
        triage.tools = ['calculator']
        triage.skills = []
        triage.confidence_internal = 0.9
        assert service.evaluate_pipeline_utility(triage, None) is True

    def test_pipeline_useful_with_skills(self, service):
        """RESPOND with skills → pipeline was useful."""
        triage = MagicMock()
        triage.mode = 'RESPOND'
        triage.tools = []
        triage.skills = ['recall']
        triage.confidence_internal = 0.9
        assert service.evaluate_pipeline_utility(triage, None) is True

    def test_pipeline_useful_low_confidence(self, service):
        """RESPOND with low confidence → pipeline was useful."""
        triage = MagicMock()
        triage.mode = 'RESPOND'
        triage.tools = []
        triage.skills = []
        triage.confidence_internal = 0.5
        assert service.evaluate_pipeline_utility(triage, None) is True

    def test_pipeline_useful_rich_context(self, service):
        """RESPOND with substantial context → pipeline was useful."""
        triage = MagicMock()
        triage.mode = 'RESPOND'
        triage.tools = []
        triage.skills = []
        triage.confidence_internal = 0.9
        context = {'total_tokens_est': 500}
        assert service.evaluate_pipeline_utility(triage, context) is True

    def test_pipeline_unnecessary(self, service):
        """RESPOND + no tools + high confidence + sparse context → unnecessary."""
        triage = MagicMock()
        triage.mode = 'RESPOND'
        triage.tools = []
        triage.skills = []
        triage.confidence_internal = 0.9
        context = {'total_tokens_est': 50}
        assert service.evaluate_pipeline_utility(triage, context) is False

    def test_pipeline_unnecessary_no_context(self, service):
        """RESPOND + no tools + high confidence + no context → unnecessary."""
        triage = MagicMock()
        triage.mode = 'RESPOND'
        triage.tools = []
        triage.skills = []
        triage.confidence_internal = 0.9
        assert service.evaluate_pipeline_utility(triage, None) is False

    def test_pipeline_utility_none_triage(self, service):
        """None triage → assume useful (can't evaluate)."""
        assert service.evaluate_pipeline_utility(None, None) is True

    # ── Cluster isolation ─────────────────────────────────────────────────

    @patch('services.embedding_service.get_embedding_service')
    def test_cluster_isolation(self, mock_emb, service, cursor):
        """Cluster A's failures don't affect cluster B's confidence."""
        emb_b = _make_embedding(seed=100)
        mock_emb.return_value.generate_embedding.return_value = emb_b

        # Cluster B is healthy — A's failures are irrelevant
        cursor.fetchone.return_value = _make_cluster_row(
            cluster_id=2,
            times_seen=10, times_unnecessary=9,
            times_activated=5, times_succeeded=4, times_failed=0,
        )

        result = service.check("Capital of France?", 0.0)
        assert result.can_activate is True
        assert result.cluster_id == 2

    # ── Shadow validation ─────────────────────────────────────────────────

    def test_shadow_validation_queued(self, service, redis):
        """Shadow validation should store data in Redis."""
        with patch('services.cognitive_reflex_service.random') as mock_random:
            mock_random.random.return_value = 0.05  # < 0.10 threshold

            service.maybe_queue_shadow_validation(
                text="What is 2+2?",
                metadata={'uuid': 'test'},
                thread_id="thread-1",
                reflex_response="4",
                cluster_id=1,
            )

            redis.setex.assert_called_once()
            call_args = redis.setex.call_args
            assert 'reflex:shadow:thread-1' in str(call_args)

    def test_shadow_validation_skipped(self, service, redis):
        """Shadow validation should be skipped ~90% of the time."""
        with patch('services.cognitive_reflex_service.random') as mock_random:
            mock_random.random.return_value = 0.5  # > 0.10 threshold

            service.maybe_queue_shadow_validation(
                text="What is 2+2?",
                metadata={'uuid': 'test'},
                thread_id="thread-1",
                reflex_response="4",
                cluster_id=1,
            )

            redis.setex.assert_not_called()

    @patch('services.embedding_service.get_embedding_service')
    def test_shadow_result_agreement(self, mock_emb, service, cursor, redis):
        """When shadow and reflex responses agree → times_succeeded++."""
        # Same embedding for both (distance ≈ 0)
        same_emb = _make_embedding()
        mock_emb.return_value.generate_embedding.return_value = same_emb

        redis.get.return_value = json.dumps({
            'text': 'What is 2+2?',
            'reflex_response': '4',
            'cluster_id': 1,
            'queued_at': time.time(),
        })

        service.process_shadow_result("thread-1", "4")

        redis.delete.assert_called_with("reflex:shadow:thread-1")
        calls = cursor.execute.call_args_list
        succeeded = [c for c in calls if 'times_succeeded' in str(c)]
        assert len(succeeded) >= 1

    @patch('services.embedding_service.get_embedding_service')
    def test_shadow_result_divergence(self, mock_emb, service, cursor, redis):
        """When shadow and reflex responses diverge → times_failed++."""
        emb1 = _make_embedding(seed=1)
        emb2 = _make_distant_embedding(emb1, seed=2)

        # Return different embeddings for each call
        mock_emb.return_value.generate_embedding.side_effect = [
            emb1,  # reflex response embedding
            emb2,  # pipeline response embedding
        ]

        redis.get.return_value = json.dumps({
            'text': 'What is 2+2?',
            'reflex_response': '4',
            'cluster_id': 1,
            'queued_at': time.time(),
        })

        service.process_shadow_result("thread-1", "The answer is four, which is the sum of two and two.")

        calls = cursor.execute.call_args_list
        failed = [c for c in calls if 'times_failed' in str(c)]
        assert len(failed) >= 1

    # ── Record activation ─────────────────────────────────────────────────

    def test_record_activation(self, service, cursor):
        """record_activation should UPDATE times_activated + last_activated."""
        service.record_activation(42)

        calls = cursor.execute.call_args_list
        activation_update = [c for c in calls if 'times_activated' in str(c)]
        assert len(activation_update) >= 1

    # ── Set pending validation ────────────────────────────────────────────

    def test_set_pending_validation(self, service, redis):
        """set_pending_validation should store cluster_id with TTL."""
        service.set_pending_validation("thread-1", 42)

        redis.setex.assert_called_once_with(
            "reflex:pending:thread-1",
            300,  # PENDING_VALIDATION_TTL
            "42",
        )

    # ── Observability stats ───────────────────────────────────────────────

    def test_get_stats(self, service, cursor):
        """get_stats should return aggregate statistics."""
        # Mock multiple cursor.fetchone calls
        cursor.fetchone.side_effect = [
            (5,),     # total_clusters
            (2,),     # active_clusters
            (50, 10, 8, 1, 40),  # aggregate counters
            (1,),     # new_24h
        ]
        cursor.fetchall.return_value = []  # recent clusters

        stats = service.get_stats()

        assert stats['total_clusters'] == 5
        assert stats['active_clusters'] == 2
        assert stats['total_observations'] == 50
        assert stats['total_activations'] == 10
        assert stats['activation_rate'] == 0.2  # 10/50
        assert stats['success_rate'] == 0.8  # 8/10

    # ── ReflexResult dataclass ────────────────────────────────────────────

    def test_reflex_result_fields(self):
        """ReflexResult should hold all expected fields."""
        result = ReflexResult(
            is_candidate=True,
            can_activate=True,
            confidence=0.9,
            cluster_id=42,
            observations=10,
            embedding=[0.1] * 768,
            reasoning="Test",
        )
        assert result.is_candidate is True
        assert result.can_activate is True
        assert result.confidence == 0.9
        assert result.cluster_id == 42
        assert result.observations == 10
        assert len(result.embedding) == 768
        assert result.reasoning == "Test"
