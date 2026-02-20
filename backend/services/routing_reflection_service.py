"""
Routing Reflection Service — Idle-time peer review of routing decisions.

Follows CognitiveDriftEngine pattern:
- Checks all queues empty before processing
- Processes batch of up to 5 decisions per idle window
- Min 30min between batches
- Uses strong LLM (qwen3:14b) as consultant, not authority
- Anti-authority safeguards: confidence gate, user override, sustained pattern
- Stratified sampling: low_confidence (50%), high_confidence (20%), tiebreaker (30%)
"""

import json
import re
import time
import random
import logging
from typing import Dict, Any, Optional, List

from services.redis_client import RedisClientService
from services.config_service import ConfigService
from services.llm_service import create_llm_service
from services.database_service import DatabaseService, get_merged_db_config
from services.routing_decision_service import RoutingDecisionService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ROUTING REFLECTION]"

# Redis keys
LAST_BATCH_KEY = "routing_reflection_last_batch"


class RoutingReflectionService:
    """
    Idle-time peer review of routing decisions.

    A strong LLM reviews past decisions to identify systematic blind spots
    in the mathematical router. Acts as consultant, not authority.
    """

    def __init__(self, check_interval: int = 300):
        self.redis = RedisClientService.create_connection()
        self.config = ConfigService.resolve_agent_config("mode-reflection")
        self.check_interval = check_interval

        # Load queue names for idle check
        conn_config = ConfigService.connections()
        topics = conn_config.get("redis", {}).get("topics", {})
        self.prompt_queue = topics.get("prompt_queue", "prompt-queue")
        self.memory_queue = topics.get("memory_chunker", "memory-chunker-queue")
        self.episodic_queue = topics.get("episodic_memory", "episodic-memory-queue")
        self.semantic_queue = conn_config.get("redis", {}).get("queues", {}).get(
            "semantic_consolidation_queue", {}
        ).get("name", "semantic_consolidation_queue")

        # Database
        db_config = get_merged_db_config()
        self.db_service = DatabaseService(db_config)
        self.decision_service = RoutingDecisionService(self.db_service)

        # LLM for reflection (provider resolved from mode-reflection config)
        self.ollama = create_llm_service(self.config)

        # Load reflection prompt
        self.prompt_template = ConfigService.get_agent_prompt("mode-reflection")

        # Config values
        self.batch_size = self.config.get('batch_size', 5)
        self.min_interval_minutes = self.config.get('min_interval_minutes', 30)
        self.confidence_gate = self.config.get('confidence_gate', 0.70)
        self.user_override_threshold = self.config.get('user_override_threshold', 0.2)

        # Sampling buckets
        self.sampling_buckets = self.config.get('sampling_buckets', {
            'low_confidence': 0.50,
            'high_confidence': 0.20,
            'tiebreaker': 0.30,
        })

        logger.info(
            f"{LOG_PREFIX} Service initialized "
            f"(batch_size={self.batch_size}, "
            f"interval={self.min_interval_minutes}min)"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop."""
        logger.info(f"{LOG_PREFIX} Service started")

        while True:
            try:
                time.sleep(self.check_interval)

                if not self._are_workers_idle():
                    continue

                if not self._can_run_batch():
                    continue

                self._run_reflection_batch()

            except KeyboardInterrupt:
                logger.info(f"{LOG_PREFIX} Service shutting down...")
                break
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
                time.sleep(60)

    def _are_workers_idle(self) -> bool:
        """Check if all worker queues are empty."""
        for queue_name in [self.prompt_queue, self.memory_queue,
                           self.episodic_queue, self.semantic_queue]:
            if self.redis.llen(queue_name) > 0:
                return False
        return True

    def _can_run_batch(self) -> bool:
        """Check if enough time has passed since last batch."""
        last_batch = self.redis.get(LAST_BATCH_KEY)
        if last_batch:
            elapsed = time.time() - float(last_batch)
            if elapsed < self.min_interval_minutes * 60:
                return False
        return True

    def _run_reflection_batch(self):
        """Execute one reflection batch."""
        logger.info(f"{LOG_PREFIX} Starting reflection batch...")

        # Select decisions using stratified sampling
        decisions = self._select_decisions()

        if not decisions:
            logger.info(f"{LOG_PREFIX} No unreflected decisions available")
            return

        reflected = 0
        for decision in decisions:
            try:
                reflection = self._reflect_on_decision(decision)
                if reflection:
                    # Apply anti-authority safeguards
                    filtered = self._apply_safeguards(decision, reflection)
                    self.decision_service.update_reflection(decision['id'], filtered)
                    reflected += 1
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to reflect on {decision['id']}: {e}")

        # Record batch time
        self.redis.set(LAST_BATCH_KEY, str(time.time()))

        logger.info(f"{LOG_PREFIX} Batch complete: {reflected}/{len(decisions)} reflected")

    def _extract_json(self, text: str) -> dict:
        """
        Robustly extract JSON from text that may contain markdown fences or preamble.

        Strips markdown code fences and finds the first { ... last } to parse JSON.
        """
        # Strip markdown fences
        text = re.sub(r'```(?:json)?\s*', '', text).strip()

        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")

        json_str = text[start:end+1]
        return json.loads(json_str)

    def _select_decisions(self) -> List[Dict]:
        """
        Stratified sampling of unreflected decisions.

        Buckets: low_confidence (50%), high_confidence (20%), tiebreaker (30%)
        """
        all_unreflected = self.decision_service.get_unreflected_decisions(limit=50)

        if not all_unreflected:
            return []

        # Categorize
        low_conf = [d for d in all_unreflected if (d.get('router_confidence') or 0) < 0.20]
        high_conf = [d for d in all_unreflected if (d.get('router_confidence') or 0) > 0.60]
        tiebreaker = [d for d in all_unreflected if d.get('tiebreaker_used')]

        # Sample per bucket
        batch = []
        bucket_sizes = {
            'low_confidence': max(1, round(self.batch_size * self.sampling_buckets.get('low_confidence', 0.5))),
            'high_confidence': max(1, round(self.batch_size * self.sampling_buckets.get('high_confidence', 0.2))),
            'tiebreaker': max(1, round(self.batch_size * self.sampling_buckets.get('tiebreaker', 0.3))),
        }

        selected_ids = set()

        for pool, count in [(low_conf, bucket_sizes['low_confidence']),
                            (high_conf, bucket_sizes['high_confidence']),
                            (tiebreaker, bucket_sizes['tiebreaker'])]:
            available = [d for d in pool if d['id'] not in selected_ids]
            sample = random.sample(available, min(count, len(available)))
            for d in sample:
                batch.append(d)
                selected_ids.add(d['id'])

        # Fill remaining slots from any bucket
        remaining = self.batch_size - len(batch)
        if remaining > 0:
            extras = [d for d in all_unreflected if d['id'] not in selected_ids]
            batch.extend(random.sample(extras, min(remaining, len(extras))))

        return batch[:self.batch_size]

    def _reflect_on_decision(self, decision: Dict) -> Optional[Dict]:
        """
        Send a decision to the strong LLM for review.

        Returns parsed reflection dict or None on failure.
        """
        signals = decision.get('signal_snapshot', {})
        scores = decision.get('scores', {})

        # Find runner-up
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        runner_up_mode = sorted_scores[1][0] if len(sorted_scores) > 1 else 'RESPOND'
        margin = decision.get('margin', 0)

        # Build prompt
        prompt = self.prompt_template
        prompt = prompt.replace('{{prompt_text}}', signals.get('_prompt_text', '(not available)'))
        prompt = prompt.replace('{{context_warmth}}', f"{signals.get('context_warmth', 0):.2f}")
        prompt = prompt.replace('{{fact_count}}', str(signals.get('fact_count', 0)))
        prompt = prompt.replace('{{fact_keys}}', ', '.join(signals.get('fact_keys', [])))
        prompt = prompt.replace('{{working_memory_turns}}', str(signals.get('working_memory_turns', 0)))
        prompt = prompt.replace('{{topic}}', decision.get('topic', 'unknown'))
        prompt = prompt.replace('{{new_or_existing}}', 'new' if signals.get('is_new_topic') else 'existing')
        prompt = prompt.replace('{{gist_count}}', str(signals.get('gist_count', 0)))
        prompt = prompt.replace('{{selected_mode}}', decision['selected_mode'])
        prompt = prompt.replace('{{router_confidence}}', f"{decision.get('router_confidence', 0):.3f}")
        prompt = prompt.replace('{{runner_up_mode}}', runner_up_mode)
        prompt = prompt.replace('{{margin}}', f"{margin:.3f}")
        prompt = prompt.replace('{{response_text}}', '(not available)')

        # User feedback
        feedback = decision.get('feedback', {})
        if feedback:
            prompt = prompt.replace('{{user_feedback}}', json.dumps(feedback))
        else:
            prompt = prompt.replace('{{user_feedback}}', 'Not yet available')

        try:
            response = self.ollama.send_message("Analyze this routing decision.", prompt).text
            result = self._extract_json(response)

            # Validate required fields
            if 'agree_with_decision' not in result:
                logger.warning(f"{LOG_PREFIX} Missing agree_with_decision in reflection")
                return None
            if 'confidence' not in result:
                result['confidence'] = 0.5

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"{LOG_PREFIX} Failed to parse reflection: {e}")
            return None
        except Exception as e:
            logger.error(f"{LOG_PREFIX} LLM call failed: {e}")
            return None

    def _apply_safeguards(self, decision: Dict, reflection: Dict) -> Dict:
        """
        Apply anti-authority safeguards to reflection.

        1. Confidence gate: discard if LLM confidence < threshold
        2. User override: trust user feedback over LLM
        3. Value coherence: reject incoherent suggestions
        """
        counted = True
        filter_reason = None

        # 1. Confidence gate
        llm_confidence = reflection.get('confidence', 0.0)
        if llm_confidence < self.confidence_gate:
            counted = False
            filter_reason = f'low_confidence ({llm_confidence:.2f} < {self.confidence_gate})'
            logger.debug(f"{LOG_PREFIX} Reflection filtered: {filter_reason}")

        # 2. User override: positive user feedback overrides LLM disagreement
        feedback = decision.get('feedback', {})
        if feedback and not reflection.get('agree_with_decision', True):
            reward = feedback.get('reward', 0.0)
            if reward > self.user_override_threshold:
                counted = False
                filter_reason = f'user_override (reward={reward:.2f} > {self.user_override_threshold})'
                logger.debug(f"{LOG_PREFIX} Reflection filtered: {filter_reason}")

        # 3. Value coherence: reject suggestions that ignore genuine user input
        if not reflection.get('agree_with_decision', True):
            signals = decision.get('signal_snapshot', {})
            # Don't suggest IGNORE for non-empty input
            dims = reflection.get('uncertainty_dimensions', [])
            for dim in dims:
                if 'IGNORE' in dim.get('affected_modes', []) and signals.get('prompt_token_count', 0) > 0:
                    counted = False
                    filter_reason = 'value_coherence (IGNORE for non-empty input)'
                    break

        reflection['counted'] = counted
        reflection['filter_reason'] = filter_reason

        return reflection


def routing_reflection_worker(shared_state=None):
    """Module-level wrapper for multiprocessing."""
    try:
        config = ConfigService.get_agent_config("mode-reflection")
        check_interval = config.get('check_interval', 300)
    except Exception:
        check_interval = 300

    service = RoutingReflectionService(check_interval=check_interval)
    service.run(shared_state)
