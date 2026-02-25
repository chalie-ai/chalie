"""
CommunicateAction — Decides whether a drift thought is worth sharing with the user.

Four gates must pass:
  1. Quality gate: thought type, activation energy, topic relevance, novelty
  2. Timing gate: min/max idle, quiet hours, session requirement, activity histogram
  3. Engagement gate: one-at-a-time, rolling score, backoff, auto-pause, recovery
  4. Cognitive load gate: holds back when user is showing high-load / disengagement signals

When all gates pass, the thought is enqueued as a candidate. The best candidate
is delivered through the full digest pipeline (mode router as final judge).
"""

import json
import time
import uuid
import logging
import math
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from services.redis_client import RedisClientService
from services.embedding_service import EmbeddingService
from services.working_memory_service import WorkingMemoryService
from services.gist_storage_service import GistStorageService

from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[COMMUNICATE]"

# Redis key namespace — all keys are per-user
# Current system is single-user; user_id defaults to 'default'
_NS = "proactive"


def _key(user_id: str, suffix: str) -> str:
    return f"{_NS}:{user_id}:{suffix}"


class CommunicateAction(AutonomousAction):
    """
    Evaluates whether a drift thought should be proactively sent to the user.

    Quality + Timing + Engagement gates must all pass for eligibility.
    Score = activation_energy * type_bonus * relevance, competing with other actions.
    """

    def __init__(self, config: dict = None):
        super().__init__(name='COMMUNICATE', enabled=True, priority=10)

        config = config or {}
        self.redis = RedisClientService.create_connection()

        # User ID (single-user for now, extensible via config)
        self.user_id = config.get('user_id', 'default')

        # Quality gate config
        self.type_bonuses = config.get('type_bonuses', {
            'question': 1.3,
            'hypothesis': 1.2,
            'reflection': 0.8,
        })
        self.bootstrap_threshold = config.get('bootstrap_threshold', 0.6)
        self.bootstrap_cycles = config.get('bootstrap_cycles', 20)
        self.relevance_threshold = config.get('relevance_threshold', 0.4)
        self.novelty_threshold = config.get('novelty_threshold', 0.7)  # Jaccard above this = duplicate
        self.lookback_hours_active = config.get('lookback_hours_active', 24)
        self.lookback_hours_infrequent = config.get('lookback_hours_infrequent', 72)

        # Timing gate config
        self.min_idle_seconds = config.get('min_idle_seconds', 1800)   # 30 min
        self.max_idle_seconds = config.get('max_idle_seconds', 86400)  # 24h
        self.quiet_hours_start = config.get('quiet_hours_start', 23)   # 23:00
        self.quiet_hours_end = config.get('quiet_hours_end', 8)        # 08:00

        # Engagement gate config
        self.pending_timeout_seconds = config.get('pending_timeout_seconds', 14400)  # 4h
        self.auto_pause_threshold = config.get('auto_pause_threshold', 0.3)
        self.max_backoff_multiplier = config.get('max_backoff_multiplier', 16)
        self.suppression_recovery_days = config.get('suppression_recovery_days', 7)

        # Candidate queue config
        self.max_candidates = config.get('max_candidates', 3)
        self.max_deferred = config.get('max_deferred', 3)
        self.deferred_ttl = config.get('deferred_ttl', 172800)  # 48h

        # Circuit breaker
        self.circuit_breaker_window = config.get('circuit_breaker_window', 14400)  # 4h
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 2)  # 2 of 3
        self.circuit_breaker_pause = config.get('circuit_breaker_pause', 28800)     # 8h

        # Lazy-loaded services
        self._embedding_service = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    # ── Gate 1: Quality ───────────────────────────────────────────

    def _quality_score(self, thought: ThoughtContext) -> Tuple[float, bool, Dict]:
        """
        Evaluate thought quality.

        Returns:
            (score, passes, gate_details)
        """
        details = {}

        # 1. Type bonus
        type_bonus = self.type_bonuses.get(thought.thought_type, 1.0)
        details['type_bonus'] = type_bonus

        # 2. Activation energy threshold (self-calibrating or bootstrap)
        drift_count = int(self.redis.get(_key(self.user_id, 'drift_count')) or 0)
        details['drift_count'] = drift_count

        if drift_count < self.bootstrap_cycles:
            threshold = self.bootstrap_threshold
            details['threshold_mode'] = 'bootstrap'
        else:
            threshold = self._get_median_activation_energy()
            details['threshold_mode'] = 'self_calibrating'
        details['activation_threshold'] = threshold

        # Reflections need top-25% activation (higher bar)
        if thought.thought_type == 'reflection':
            threshold = threshold * 1.25
            details['reflection_raised_threshold'] = threshold

        if thought.activation_energy < threshold:
            details['rejected'] = 'activation_energy_below_threshold'
            return (0.0, False, details)

        # 3. Topic relevance via embedding similarity
        relevance = self._compute_topic_relevance(thought)
        details['topic_relevance'] = relevance

        if relevance < self.relevance_threshold:
            details['rejected'] = 'topic_relevance_below_threshold'
            return (0.0, False, details)

        # 4. Novelty check (not already discussed)
        is_novel, novelty_details = self._check_novelty(thought)
        details['novelty'] = novelty_details

        if not is_novel:
            details['rejected'] = 'not_novel'
            return (0.0, False, details)

        # Composite quality score
        score = thought.activation_energy * type_bonus * relevance
        details['composite_score'] = score

        return (score, True, details)

    def _get_median_activation_energy(self) -> float:
        """Get median activation energy from recent drift history."""
        history_key = _key(self.user_id, 'activation_history')
        values = self.redis.lrange(history_key, 0, -1)

        if not values or len(values) < 5:
            return self.bootstrap_threshold

        float_values = sorted(float(v) for v in values)
        mid = len(float_values) // 2
        return float_values[mid]

    def record_activation_energy(self, energy: float):
        """Record activation energy for self-calibration. Called by drift engine."""
        history_key = _key(self.user_id, 'activation_history')
        self.redis.rpush(history_key, str(round(energy, 4)))
        self.redis.ltrim(history_key, -100, -1)  # Keep last 100

        # Increment drift count
        self.redis.incr(_key(self.user_id, 'drift_count'))

    def _compute_topic_relevance(self, thought: ThoughtContext) -> float:
        """Compute cosine similarity between thought and recent user messages."""
        if thought.thought_embedding is None:
            return 0.0

        # Get recent user message embeddings from Redis
        lookback = self._get_lookback_hours()
        embeddings_key = _key(self.user_id, 'recent_msg_embeddings')
        stored = self.redis.lrange(embeddings_key, 0, -1)

        if not stored:
            return 0.0

        max_similarity = 0.0
        thought_emb = thought.thought_embedding

        for raw in stored:
            try:
                entry = json.loads(raw)
                # Skip entries older than lookback
                if time.time() - entry.get('ts', 0) > lookback * 3600:
                    continue
                msg_emb = entry.get('embedding', [])
                sim = self._cosine_similarity(thought_emb, msg_emb)
                if sim > max_similarity:
                    max_similarity = sim
            except (json.JSONDecodeError, TypeError):
                continue

        return max_similarity

    def _get_lookback_hours(self) -> int:
        """Adaptive lookback: 24h for active users, 72h for infrequent."""
        last_ts = self.redis.get(_key(self.user_id, 'last_interaction_ts'))
        if not last_ts:
            return self.lookback_hours_active

        gap = time.time() - float(last_ts)
        # If last interaction was > 48h ago, treat as infrequent
        if gap > 172800:
            return self.lookback_hours_infrequent
        return self.lookback_hours_active

    def _check_novelty(self, thought: ThoughtContext) -> Tuple[bool, Dict]:
        """Check the thought doesn't substantially overlap with recent working memory."""
        details = {}
        wm = WorkingMemoryService(max_turns=4)
        turns = wm.get_recent_turns(thought.seed_topic)

        if not turns:
            return (True, {'reason': 'no_working_memory'})

        # Jaccard similarity against recent content
        max_sim = 0.0
        for turn in turns:
            content = turn.get('content', '')
            sim = GistStorageService._calculate_jaccard_similarity(
                thought.thought_content, content
            )
            if sim > max_sim:
                max_sim = sim

        details['max_jaccard'] = max_sim
        details['threshold'] = self.novelty_threshold

        if max_sim >= self.novelty_threshold:
            return (False, details)
        return (True, details)

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Gate 2: Timing ────────────────────────────────────────────

    def _timing_passes(self, thought: ThoughtContext) -> Tuple[bool, Dict]:
        """
        Check timing constraints.

        Returns:
            (passes, gate_details)
        """
        details = {}
        now = time.time()

        # 1. Session requirement: at least one conversation in lookback window
        last_interaction = self.redis.get(_key(self.user_id, 'last_interaction_ts'))
        if not last_interaction:
            details['rejected'] = 'no_prior_interaction'
            return (False, details)

        last_ts = float(last_interaction)
        idle_seconds = now - last_ts
        details['idle_seconds'] = idle_seconds

        # 2. Minimum idle time (with backoff multiplier)
        backoff = int(self.redis.get(_key(self.user_id, 'backoff_multiplier')) or 1)
        effective_min_idle = self.min_idle_seconds * backoff
        details['effective_min_idle'] = effective_min_idle
        details['backoff_multiplier'] = backoff

        if idle_seconds < effective_min_idle:
            details['rejected'] = 'too_soon'
            return (False, details)

        # 3. Maximum idle (context staleness)
        max_idle = self.max_idle_seconds
        lookback = self._get_lookback_hours()
        if lookback > 24:
            max_idle = lookback * 3600
        details['max_idle'] = max_idle

        if idle_seconds > max_idle:
            details['rejected'] = 'context_stale'
            return (False, details)

        # 4. Quiet hours check (use client timezone, not server time)
        current_hour = self._get_user_hour()
        details['current_hour'] = current_hour

        if self._is_quiet_hours(current_hour):
            details['rejected'] = 'quiet_hours'
            return (False, details)

        return (True, details)

    def _is_quiet_hours(self, hour: int) -> bool:
        """Check if the given hour falls in quiet hours."""
        start = self.quiet_hours_start
        end = self.quiet_hours_end

        if start > end:
            # Wraps midnight (e.g., 23:00 - 08:00)
            return hour >= start or hour < end
        else:
            return start <= hour < end

    def _get_user_hour(self) -> int:
        """Get current hour in the user's timezone (falls back to server time)."""
        try:
            from services.client_context_service import ClientContextService
            from zoneinfo import ZoneInfo
            ctx = ClientContextService().get()
            tz = ctx.get("timezone")
            if tz:
                return datetime.now(ZoneInfo(tz)).hour
        except Exception:
            pass
        return datetime.now().hour

    # ── Gate 3: Engagement ────────────────────────────────────────

    def _engagement_passes(self, thought: ThoughtContext) -> Tuple[bool, Dict]:
        """
        Check engagement constraints.

        Returns:
            (passes, gate_details)
        """
        details = {}
        now = time.time()

        # 1. Auto-pause check
        paused = self.redis.get(_key(self.user_id, 'paused'))
        if paused == '1':
            # Check for suppression recovery
            paused_since = float(self.redis.get(_key(self.user_id, 'paused_since')) or 0)
            days_paused = (now - paused_since) / 86400 if paused_since else 0

            if days_paused >= self.suppression_recovery_days:
                # Check if user has been active during pause
                last_ts = self.redis.get(_key(self.user_id, 'last_interaction_ts'))
                if last_ts and float(last_ts) > paused_since:
                    # Recovery probe: reset backoff to 2x and unpause
                    self.redis.set(_key(self.user_id, 'backoff_multiplier'), 2)
                    self.redis.delete(_key(self.user_id, 'paused'))
                    self.redis.delete(_key(self.user_id, 'paused_since'))
                    details['suppression_recovery'] = True
                    logger.info(f"{LOG_PREFIX} Suppression recovery: resuming with 2x backoff")
                else:
                    details['rejected'] = 'paused_no_user_activity'
                    return (False, details)
            else:
                details['rejected'] = 'auto_paused'
                details['days_paused'] = days_paused
                return (False, details)

        # 2. One-at-a-time rule
        pending = self.redis.get(_key(self.user_id, 'pending_response'))
        if pending:
            pending_ts = float(self.redis.get(_key(self.user_id, 'last_sent_ts')) or 0)
            if now - pending_ts < self.pending_timeout_seconds:
                details['rejected'] = 'pending_response'
                return (False, details)
            else:
                # Timeout expired — treat as ignored
                self._record_outcome(pending, -0.5)
                self.redis.delete(_key(self.user_id, 'pending_response'))
                details['timeout_expired'] = pending

        # 3. Circuit breaker check
        if self._circuit_breaker_tripped():
            details['rejected'] = 'circuit_breaker'
            return (False, details)

        # 4. Check engagement score
        engagement = float(self.redis.get(_key(self.user_id, 'engagement_score')) or 1.0)
        details['engagement_score'] = engagement

        if engagement < self.auto_pause_threshold:
            # Auto-pause
            self.redis.set(_key(self.user_id, 'paused'), '1')
            self.redis.set(_key(self.user_id, 'paused_since'), str(now))
            details['rejected'] = 'engagement_too_low'
            logger.info(f"{LOG_PREFIX} Auto-paused: engagement {engagement:.2f} < {self.auto_pause_threshold}")
            return (False, details)

        return (True, details)

    def _circuit_breaker_tripped(self) -> bool:
        """Check if recent outcomes trigger the circuit breaker."""
        outcomes_key = _key(self.user_id, 'recent_outcomes')
        raw = self.redis.lrange(outcomes_key, 0, 2)  # Last 3

        if len(raw) < 2:
            return False

        now = time.time()
        failures = 0
        for entry_raw in raw:
            try:
                entry = json.loads(entry_raw)
                if now - entry.get('ts', 0) > self.circuit_breaker_window:
                    continue
                if entry.get('outcome', '') in ('ignored', 'dismissed', 'router_ignored'):
                    failures += 1
            except (json.JSONDecodeError, TypeError):
                continue

        if failures >= self.circuit_breaker_threshold:
            # Pause for circuit_breaker_pause seconds
            self.redis.set(_key(self.user_id, 'paused'), '1')
            self.redis.set(_key(self.user_id, 'paused_since'), str(now))
            # Auto-expire the pause after circuit_breaker_pause
            self.redis.expire(_key(self.user_id, 'paused'), self.circuit_breaker_pause)
            self.redis.expire(_key(self.user_id, 'paused_since'), self.circuit_breaker_pause)
            logger.info(
                f"{LOG_PREFIX} Circuit breaker tripped: {failures} failures in window, "
                f"pausing for {self.circuit_breaker_pause}s"
            )
            return True

        return False

    # ── Candidate queue ───────────────────────────────────────────

    def _add_candidate(self, thought: ThoughtContext, score: float):
        """Add thought to candidate queue (sorted set, max 3)."""
        candidates_key = _key(self.user_id, 'candidates')

        candidate = {
            'id': str(uuid.uuid4()),
            'type': thought.thought_type,
            'content': thought.thought_content,
            'topic': thought.seed_topic,
            'seed_concept': thought.seed_concept,
            'activation_energy': thought.activation_energy,
            'score': score,
            'created_at': time.time(),
            'gist_ttl': thought.drift_gist_ttl,
            'embedding': thought.thought_embedding,
        }

        # Score in sorted set = quality score (will be age-decayed on read)
        self.redis.zadd(candidates_key, {json.dumps(candidate): score})

        # Trim to max candidates (keep highest scoring)
        count = self.redis.zcard(candidates_key)
        if count > self.max_candidates:
            self.redis.zremrangebyrank(candidates_key, 0, count - self.max_candidates - 1)

        # TTL on the set = max gist TTL
        self.redis.expire(candidates_key, thought.drift_gist_ttl)

        logger.info(
            f"{LOG_PREFIX} Added candidate: [{thought.thought_type}] "
            f"score={score:.3f} ({self.redis.zcard(candidates_key)} in queue)"
        )

    def _get_best_candidate(self) -> Optional[Dict]:
        """Get the best candidate with age-decayed scoring."""
        candidates_key = _key(self.user_id, 'candidates')
        raw_members = self.redis.zrange(candidates_key, 0, -1, withscores=True)

        if not raw_members:
            return None

        now = time.time()
        best = None
        best_effective_score = -1

        for raw, _stored_score in raw_members:
            try:
                candidate = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            created = candidate.get('created_at', now)
            ttl = candidate.get('gist_ttl', 1800)
            elapsed = now - created

            # Expired candidate
            if elapsed >= ttl:
                self.redis.zrem(candidates_key, raw)
                continue

            # Age decay: effective_score = score * (remaining_ttl / original_ttl)
            remaining = ttl - elapsed
            decay_factor = remaining / ttl
            effective_score = candidate.get('score', 0) * decay_factor

            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best = candidate
                best['_raw'] = raw  # For removal after selection

        return best

    def _pop_best_candidate(self) -> Optional[Dict]:
        """Get and remove the best candidate."""
        best = self._get_best_candidate()
        if best:
            raw = best.pop('_raw', None)
            if raw:
                self.redis.zrem(_key(self.user_id, 'candidates'), raw)
        return best

    # ── Deferred queue (quiet hours) ──────────────────────────────

    def _add_deferred(self, thought: ThoughtContext, score: float):
        """Add thought to deferred queue for post-quiet-hours delivery."""
        deferred_key = _key(self.user_id, 'deferred')

        deferred = {
            'id': str(uuid.uuid4()),
            'type': thought.thought_type,
            'content': thought.thought_content,
            'topic': thought.seed_topic,
            'seed_concept': thought.seed_concept,
            'activation_energy': thought.activation_energy,
            'score': score,
            'created_at': time.time(),
            'embedding': thought.thought_embedding,
        }

        self.redis.zadd(deferred_key, {json.dumps(deferred): score})

        # Trim to max
        count = self.redis.zcard(deferred_key)
        if count > self.max_deferred:
            self.redis.zremrangebyrank(deferred_key, 0, count - self.max_deferred - 1)

        self.redis.expire(deferred_key, self.deferred_ttl)

        logger.debug(f"{LOG_PREFIX} Deferred thought for post-quiet delivery")

    def process_deferred_queue(self) -> Optional[Dict]:
        """
        Check if quiet hours just ended and deliver top deferred thought.

        Called by the drift engine at the start of each cycle.
        Returns the best deferred thought if ready for delivery, else None.
        """
        current_hour = self._get_user_hour()

        # Only process if we just exited quiet hours
        if self._is_quiet_hours(current_hour):
            return None

        deferred_key = _key(self.user_id, 'deferred')
        if not self.redis.exists(deferred_key):
            return None

        # Check if we already processed deferred this quiet-hours cycle
        processed_key = _key(self.user_id, 'deferred_processed')
        if self.redis.get(processed_key):
            return None

        # Pop the highest-scored deferred thought
        raw_members = self.redis.zrange(deferred_key, 0, -1, withscores=True)
        if not raw_members:
            return None

        # Top thought gets delivery attempt
        best_raw, _score = raw_members[-1]  # Highest score
        self.redis.zrem(deferred_key, best_raw)

        try:
            best = json.loads(best_raw)
        except (json.JSONDecodeError, TypeError):
            return None

        # 2nd thought enters candidate queue (if exists)
        if len(raw_members) > 1:
            second_raw, second_score = raw_members[-2]
            self.redis.zrem(deferred_key, second_raw)
            try:
                second = json.loads(second_raw)
                candidates_key = _key(self.user_id, 'candidates')
                self.redis.zadd(candidates_key, {second_raw: second_score})
                self.redis.expire(candidates_key, 1800)
            except (json.JSONDecodeError, TypeError):
                pass

        # 3rd+ discarded (already removed if we only keep max 3)
        self.redis.delete(deferred_key)

        # Mark as processed for this cycle (expire after quiet hours window)
        self.redis.setex(processed_key, 43200, '1')  # 12h TTL

        logger.info(f"{LOG_PREFIX} Delivering deferred thought: [{best.get('type')}]")
        return best

    # ── Gate 4: Cognitive load ────────────────────────────────────

    def _cognitive_load_gate(self, thought: ThoughtContext) -> Tuple[bool, Dict]:
        """
        Block proactive delivery when user shows high cognitive load signals.

        Estimates load from reply length trend in recent working memory.
        If the last user reply is very short AND dropped sharply vs the prior
        reply, the user is likely disengaging — hold off on proactive messages.
        """
        try:
            wm = WorkingMemoryService(max_turns=4)
            turns = wm.get_recent_turns(thought.seed_topic)
            user_lengths = [len(t.get('content', '')) for t in (turns or [])
                            if t.get('role', 'user') != 'assistant']
            if len(user_lengths) >= 2:
                last, prev = user_lengths[-1], user_lengths[-2]
                if last < 25 and prev > 0 and (last / max(prev, 1)) < 0.3:
                    return (False, {'load_state': 'HIGH', 'reason': 'declining_short_replies'})
        except Exception:
            pass
        return (True, {'load_state': 'NORMAL'})

    # ── Main interface ────────────────────────────────────────────

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """
        Evaluate all four gates. Returns (score, eligible).

        Always records activation energy for self-calibration.
        If quality passes but timing/engagement don't, adds to candidate queue.
        """
        # Always record for calibration
        self.record_activation_energy(thought.activation_energy)

        # Gate 1: Quality
        quality_score, quality_passes, quality_details = self._quality_score(thought)
        if not quality_passes:
            return (0.0, False)

        # Gate 2: Timing
        timing_passes, timing_details = self._timing_passes(thought)

        # If timing fails due to quiet hours, defer the thought
        if not timing_passes and timing_details.get('rejected') == 'quiet_hours':
            self._add_deferred(thought, quality_score)
            return (0.0, False)

        if not timing_passes:
            # Quality passed but timing didn't — add to candidate queue
            self._add_candidate(thought, quality_score)
            return (0.0, False)

        # Gate 3: Engagement
        engagement_passes, engagement_details = self._engagement_passes(thought)
        if not engagement_passes:
            # Quality + timing passed but engagement blocked — add to candidate queue
            self._add_candidate(thought, quality_score)
            return (0.0, False)

        # Gate 4: Cognitive load — don't interrupt a disengaging user
        load_passes, load_details = self._cognitive_load_gate(thought)
        if not load_passes:
            logger.debug(f"{LOG_PREFIX} Cognitive load gate blocked: {load_details}")
            self._add_candidate(thought, quality_score)
            return (0.0, False)

        # All gates pass — check if there's a better candidate waiting
        self._add_candidate(thought, quality_score)
        best = self._get_best_candidate()

        if best:
            return (best.get('score', quality_score), True)
        return (quality_score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """
        Enqueue the best candidate on prompt-queue for delivery via digest pipeline.
        """
        from services.prompt_queue import PromptQueue
        from workers.digest_worker import digest_worker

        # Pop best candidate
        candidate = self._pop_best_candidate()
        if not candidate:
            return ActionResult(action_name='COMMUNICATE', success=False,
                              details={'reason': 'no_candidate'})

        proactive_id = str(uuid.uuid4())

        # Build metadata for the digest worker
        metadata = {
            'source': 'proactive_drift',
            'type': 'proactive_drift',
            'drift_gist': candidate['content'],
            'drift_type': candidate.get('type', 'reflection'),
            'related_topic': candidate.get('topic', 'general'),
            'proactive_id': proactive_id,
            'destination': 'web',
        }

        # Enqueue on prompt-queue
        try:
            prompt_queue = PromptQueue(
                queue_name="prompt-queue",
                worker_func=digest_worker
            )
            prompt_queue.enqueue(candidate['content'], metadata)

            # Track pending response
            now = time.time()
            self.redis.set(_key(self.user_id, 'pending_response'), proactive_id)
            self.redis.set(_key(self.user_id, 'last_sent_ts'), str(now))

            # Store content for engagement scoring
            from .engagement_tracker import EngagementTracker
            tracker = EngagementTracker(config={'user_id': self.user_id})
            tracker.store_pending_content(
                proactive_id,
                candidate['content'],
                embedding=candidate.get('embedding'),
            )

            logger.info(
                f"{LOG_PREFIX} Enqueued proactive message: [{candidate.get('type')}] "
                f"id={proactive_id[:8]}, topic={candidate.get('topic')}"
            )

            return ActionResult(
                action_name='COMMUNICATE',
                success=True,
                details={
                    'proactive_id': proactive_id,
                    'thought_type': candidate.get('type'),
                    'topic': candidate.get('topic'),
                    'score': candidate.get('score'),
                }
            )

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to enqueue: {e}")
            return ActionResult(action_name='COMMUNICATE', success=False,
                              details={'error': str(e)})

    def on_outcome(self, result: ActionResult, user_feedback: Optional[Dict] = None) -> None:
        """Update engagement tracking based on outcome."""
        if not user_feedback:
            return

        proactive_id = result.details.get('proactive_id')
        if proactive_id:
            outcome = user_feedback.get('outcome', 'unknown')
            score = user_feedback.get('score', 0.0)
            self._record_outcome(proactive_id, score, outcome)

    # ── Engagement tracking helpers ───────────────────────────────

    def _record_outcome(self, proactive_id: str, score: float, outcome: str = None):
        """Record the outcome of a proactive message."""
        outcomes_key = _key(self.user_id, 'recent_outcomes')
        history_key = _key(self.user_id, 'engagement_history')

        # Determine outcome label
        if outcome is None:
            if score >= 0.8:
                outcome = 'engaged'
            elif score >= 0.3:
                outcome = 'acknowledged'
            elif score >= 0.0:
                outcome = 'dismissed'
            else:
                outcome = 'ignored'

        entry = json.dumps({
            'id': proactive_id,
            'outcome': outcome,
            'score': score,
            'ts': time.time(),
        })

        # Recent outcomes (circuit breaker, max 3)
        self.redis.lpush(outcomes_key, entry)
        self.redis.ltrim(outcomes_key, 0, 2)

        # Engagement history (rolling window of 10)
        self.redis.lpush(history_key, entry)
        self.redis.ltrim(history_key, 0, 9)

        # Recompute engagement score
        self._recompute_engagement_score()

        # Adjust backoff
        if outcome in ('engaged', 'acknowledged'):
            # Reset backoff on positive engagement
            self.redis.set(_key(self.user_id, 'backoff_multiplier'), 1)
        elif outcome in ('ignored', 'dismissed'):
            # Increase backoff
            current = int(self.redis.get(_key(self.user_id, 'backoff_multiplier')) or 1)
            new_backoff = min(current * 2, self.max_backoff_multiplier)
            self.redis.set(_key(self.user_id, 'backoff_multiplier'), new_backoff)
            logger.info(f"{LOG_PREFIX} Backoff increased to {new_backoff}x")

        # Clear pending
        self.redis.delete(_key(self.user_id, 'pending_response'))

    def _recompute_engagement_score(self):
        """Recompute rolling engagement score from history."""
        history_key = _key(self.user_id, 'engagement_history')
        raw = self.redis.lrange(history_key, 0, 9)

        if not raw:
            self.redis.set(_key(self.user_id, 'engagement_score'), '1.0')
            return

        total_score = 0.0
        count = 0
        for entry_raw in raw:
            try:
                entry = json.loads(entry_raw)
                # Normalize: engaged=1, acknowledged=0.5, dismissed=0, ignored=-0.5
                outcome = entry.get('outcome', 'unknown')
                if outcome == 'engaged':
                    total_score += 1.0
                elif outcome == 'acknowledged':
                    total_score += 0.5
                elif outcome == 'dismissed':
                    total_score += 0.0
                elif outcome == 'ignored':
                    total_score += -0.5
                count += 1
            except (json.JSONDecodeError, TypeError):
                continue

        if count > 0:
            engagement = max(0.0, (total_score / count + 0.5) / 1.5)  # Normalize to 0-1
        else:
            engagement = 1.0

        self.redis.set(_key(self.user_id, 'engagement_score'), str(round(engagement, 3)))

    # ── User interaction tracking ─────────────────────────────────

    def record_user_interaction(self, message_embedding: list = None):
        """
        Record a user interaction for timing and relevance calculations.

        Called by the digest worker on each user message.
        """
        now = time.time()
        self.redis.set(_key(self.user_id, 'last_interaction_ts'), str(now))

        # Update activity histogram (use client timezone)
        current_hour = self._get_user_hour()
        hist_key = _key(self.user_id, 'activity_histogram')
        self.redis.hincrby(hist_key, f'hour_{current_hour}', 1)

        # Store message embedding for topic relevance
        if message_embedding:
            embeddings_key = _key(self.user_id, 'recent_msg_embeddings')
            entry = json.dumps({
                'embedding': message_embedding,
                'ts': now,
            })
            self.redis.rpush(embeddings_key, entry)
            self.redis.ltrim(embeddings_key, -20, -1)  # Keep last 20
            self.redis.expire(embeddings_key, self.max_idle_seconds)  # TTL = max idle

    # ── State queries (for governance) ────────────────────────────

    def get_proactive_stats(self) -> Dict[str, Any]:
        """Get current proactive messaging state for governance/logging."""
        return {
            'engagement_score': float(self.redis.get(_key(self.user_id, 'engagement_score')) or 1.0),
            'backoff_multiplier': int(self.redis.get(_key(self.user_id, 'backoff_multiplier')) or 1),
            'paused': self.redis.get(_key(self.user_id, 'paused')) == '1',
            'pending_response': self.redis.get(_key(self.user_id, 'pending_response')) or None,
            'drift_count': int(self.redis.get(_key(self.user_id, 'drift_count')) or 0),
            'candidate_count': self.redis.zcard(_key(self.user_id, 'candidates')),
            'deferred_count': self.redis.zcard(_key(self.user_id, 'deferred')),
        }

    def get_weekly_engagement(self) -> List[float]:
        """Get weekly engagement trend (list of up to 8 weekly averages)."""
        weekly_key = _key(self.user_id, 'weekly_engagement')
        raw = self.redis.lrange(weekly_key, 0, 7)
        return [float(v) for v in raw] if raw else []

    def record_weekly_engagement(self):
        """Snapshot current engagement to weekly history. Called by governance."""
        engagement = float(self.redis.get(_key(self.user_id, 'engagement_score')) or 1.0)
        weekly_key = _key(self.user_id, 'weekly_engagement')
        self.redis.rpush(weekly_key, str(round(engagement, 3)))
        self.redis.ltrim(weekly_key, -8, -1)  # Keep last 8 weeks
