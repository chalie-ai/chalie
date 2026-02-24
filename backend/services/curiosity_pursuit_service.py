"""
CuriosityPursuitService — Background worker that explores curiosity threads.

Runs on a 6-hour cycle with ±30% jitter. Each cycle picks 1 eligible thread,
generates a self-prompt, routes it through the ACT loop, and stores learning notes.

When enough learning accumulates and surfacing conditions are met, enqueues a
proactive candidate for delivery to the user.
"""

import json
import os
import time
import random
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

from services.redis_client import RedisClientService
from services.config_service import ConfigService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CURIOSITY PURSUIT]"


class CuriosityPursuitService:
    """Background service that explores curiosity threads via the ACT loop."""

    def __init__(self, cycle_interval: int = 21600):
        """
        Args:
            cycle_interval: Base seconds between exploration cycles (default: 6h)
        """
        self.cycle_interval = cycle_interval
        self.jitter = 0.3  # ±30%

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop."""
        logger.info(f"{LOG_PREFIX} Service started (cycle={self.cycle_interval}s)")

        while True:
            try:
                # Sleep with jitter
                jitter_factor = 1.0 + random.uniform(-self.jitter, self.jitter)
                sleep_time = self.cycle_interval * jitter_factor
                time.sleep(sleep_time)

                self.run_once()

            except KeyboardInterrupt:
                logger.info(f"{LOG_PREFIX} Service shutting down...")
                break
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Cycle error: {e}", exc_info=True)
                time.sleep(300)  # 5min backoff on error

    def run_once(self) -> Optional[Dict]:
        """
        Run one exploration cycle.

        Returns:
            Dict with exploration results, or None if nothing to explore.
        """
        from services.curiosity_thread_service import CuriosityThreadService

        thread_service = CuriosityThreadService()
        threads = thread_service.get_threads_for_exploration(limit=1)

        if not threads:
            logger.debug(f"{LOG_PREFIX} No threads eligible for exploration")
            return None

        thread = threads[0]
        thread_id = thread['id']

        logger.info(
            f"{LOG_PREFIX} Exploring thread {thread_id}: "
            f"'{thread['title']}' (type={thread['thread_type']}, "
            f"engagement={thread['engagement_score']:.2f})"
        )

        # Mark explored immediately to prevent double-pickup
        thread_service.mark_explored(thread_id)

        # Compute fatigue budget from engagement
        fatigue_budget = thread_service.get_fatigue_budget(thread)

        # Generate self-prompt based on thread type
        self_prompt = self._build_self_prompt(thread)

        # Run through ACT loop
        learning_summary = self._run_act_loop(self_prompt, thread, fatigue_budget)

        if learning_summary:
            # Store learning note
            thread_service.add_learning_note(thread_id, learning_summary, source='pursuit')

            # Check surfacing conditions
            self._check_surfacing(thread, thread_service)

            logger.info(
                f"{LOG_PREFIX} Exploration complete for thread {thread_id}: "
                f"'{learning_summary[:80]}...'"
            )

            return {
                'thread_id': thread_id,
                'summary': learning_summary,
                'fatigue_budget': fatigue_budget,
            }
        else:
            logger.info(f"{LOG_PREFIX} No learning produced for thread {thread_id}")
            return None

    def _build_self_prompt(self, thread: Dict) -> str:
        """Generate a self-prompt based on thread type."""
        if thread['thread_type'] == 'learning':
            return (
                f"I've been curious about {thread['seed_topic']}. "
                f"{thread['rationale']}. "
                f"Let me see what I can find or recall."
            )
        else:  # behavioral
            return (
                f"I'm curious about {thread['title']}. "
                f"{thread['rationale']}. "
                f"Let me think about this."
            )

    def _run_act_loop(self, self_prompt: str, thread: Dict, fatigue_budget: float) -> Optional[str]:
        """
        Route self-prompt through ACT loop machinery.

        Returns:
            1-3 sentence learning note summary, or None on failure.
        """
        try:
            from services.act_loop_service import ActLoopService
            from services.frontal_cortex_service import FrontalCortexService
            from services.act_dispatcher_service import ActDispatcherService
            from services.innate_skills import register_innate_skills

            cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

            # Override fatigue budget for pursuit (lower than user ACT)
            cortex_config = dict(cortex_config)
            cortex_config['fatigue_budget'] = fatigue_budget

            act_config = ConfigService.resolve_agent_config("frontal-cortex")
            act_prompt_template = ConfigService.get_agent_prompt("frontal-cortex-act")

            cortex = FrontalCortexService(cortex_config)
            dispatcher = ActDispatcherService()
            register_innate_skills(dispatcher)

            act_loop = ActLoopService(
                config=cortex_config,
                cumulative_timeout=cortex_config.get('act_cumulative_timeout', 30.0),
                per_action_timeout=cortex_config.get('act_per_action_timeout', 10.0),
                max_iterations=cortex_config.get('max_act_iterations', 5),
            )

            topic = thread.get('seed_topic', 'general')
            classification = {'topic': topic, 'confidence': 0.5}

            # Determine skills for pursuit
            if thread['thread_type'] == 'learning':
                selected_skills = ['recall', 'memorize', 'introspect', 'associate']
            else:
                selected_skills = ['recall', 'introspect', 'associate']

            # Run ACT loop iterations
            for _ in range(act_loop.max_iterations):
                can_continue, reason = act_loop.can_continue()
                if not can_continue:
                    break

                response = cortex.generate_response(
                    system_prompt_template=act_prompt_template,
                    original_prompt=self_prompt,
                    classification=classification,
                    chat_history=[],
                    act_history=act_loop.get_history_context(),
                    selected_skills=selected_skills,
                )

                actions = response.get('actions')
                if not actions:
                    break

                results = act_loop.execute_actions(
                    topic=topic,
                    actions=actions,
                )
                act_loop.append_results(results)
                act_loop.accumulate_fatigue(results, act_loop.iteration_number)
                act_loop.iteration_number += 1

            # Summarize act_history into a learning note
            history = act_loop.get_history_context()
            if not history:
                return None

            return self._summarize_learning(history, thread)

        except Exception as e:
            logger.error(f"{LOG_PREFIX} ACT loop failed: {e}", exc_info=True)
            return None

    def _summarize_learning(self, act_history: str, thread: Dict) -> Optional[str]:
        """
        Summarize ACT loop results into a 1-3 sentence learning note.

        Uses the same LLM as drift for lightweight summarization.
        """
        if not act_history or len(str(act_history)) < 20:
            return None

        try:
            from services.llm_service import create_llm_service

            config = ConfigService.resolve_agent_config("cognitive-drift")
            llm = create_llm_service(config)

            prompt = (
                f"Summarize these internal exploration results about '{thread['seed_topic']}' "
                f"in 1-3 sentences. Focus on what was learned or discovered. "
                f"Be concise and factual.\n\n"
                f"Results:\n{act_history}"
            )

            response = llm.send_message("", prompt)
            summary = response.text.strip()

            # Strip any JSON wrapper if LLM wrapped it
            if summary.startswith('{'):
                try:
                    data = json.loads(summary)
                    summary = data.get('response', data.get('summary', summary))
                except json.JSONDecodeError:
                    pass

            return summary if len(summary) > 10 else None

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Learning summarization failed: {e}")
            return None

    def _check_surfacing(self, thread: Dict, thread_service) -> None:
        """
        Check if this thread should be surfaced to the user.

        Conditions:
          - exploration_count >= 2
          - last_surfaced_at is None OR now - last_surfaced_at > effective_surface_interval
          - get_surfacing_candidate() returns non-empty content
        """
        thread_id = thread['id']

        # Need at least 2 explorations
        if thread.get('exploration_count', 0) + 1 < 2:  # +1 for this cycle
            return

        # Check surfacing interval
        now = datetime.now(timezone.utc)
        effective_interval = thread_service.get_effective_surface_interval(thread)

        if thread.get('last_surfaced_at') is not None:
            elapsed = (now - thread['last_surfaced_at']).total_seconds()
            if elapsed < effective_interval:
                return

        # Get surfacing candidate
        candidate = thread_service.get_surfacing_candidate(thread_id)
        if not candidate:
            return

        # Generate surprise message and enqueue
        surprise_message = self._generate_surprise(thread, candidate)
        if not surprise_message:
            return

        self._enqueue_proactive(thread, surprise_message)
        thread_service.mark_surfaced(thread_id)

        logger.info(f"{LOG_PREFIX} Surfacing thread {thread_id} to user")

    def _generate_surprise(self, thread: Dict, learning_summary: str) -> Optional[str]:
        """
        Run learning summary through curiosity-thread-surprise.md prompt.

        Returns the surprise message or None if too thin/boring.
        """
        try:
            from services.llm_service import create_llm_service
            from services.user_trait_service import UserTraitService
            from services.database_service import get_shared_db_service

            config = ConfigService.resolve_agent_config("cognitive-drift")
            llm = create_llm_service(config)

            # Load surprise prompt
            prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'prompts',
                'curiosity-thread-surprise.md'
            )
            with open(prompt_path, 'r') as f:
                system_prompt = f.read()

            # Get user traits for context
            user_traits = ""
            try:
                db = get_shared_db_service()
                trait_service = UserTraitService(db)
                user_traits = trait_service.get_traits_for_prompt("")
            except Exception:
                pass

            system_prompt = (
                system_prompt
                .replace('{{thread_title}}', thread.get('title', ''))
                .replace('{{learning_summary}}', learning_summary)
                .replace('{{user_traits}}', user_traits)
            )

            response = llm.send_message(system_prompt, "").text.strip()

            # Parse JSON response
            if response.startswith('```'):
                response = response.split('\n', 1)[-1].rsplit('```', 1)[0]

            try:
                data = json.loads(response)
                message = data.get('response', '')
                if message and len(message) > 10:
                    return message
                return None
            except json.JSONDecodeError:
                # If not JSON, use raw text if it looks like a message
                if len(response) > 10 and len(response) < 500:
                    return response
                return None

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Surprise generation failed: {e}")
            return None

    def _enqueue_proactive(self, thread: Dict, message: str) -> None:
        """Enqueue the surprise message as a proactive candidate."""
        try:
            from services.prompt_queue import PromptQueue
            from workers.digest_worker import digest_worker
            from services.autonomous_actions.engagement_tracker import EngagementTracker
            from services.embedding_service import EmbeddingService
            import uuid

            proactive_id = str(uuid.uuid4())

            metadata = {
                'source': 'curiosity_thread',
                'type': 'curiosity_thread',
                'drift_gist': message,
                'drift_type': 'curiosity',
                'related_topic': thread.get('seed_topic', 'general'),
                'proactive_id': proactive_id,
                'thread_id': thread['id'],
                'destination': 'web',
            }

            prompt_queue = PromptQueue(
                queue_name="prompt-queue",
                worker_func=digest_worker,
            )
            prompt_queue.enqueue(message, metadata)

            # Track pending response for engagement scoring
            redis = RedisClientService.create_connection()
            user_id = 'default'
            redis.set(f"proactive:{user_id}:pending_response", proactive_id)
            redis.set(f"proactive:{user_id}:last_sent_ts", str(time.time()))

            # Store content for engagement scoring
            try:
                embedding = EmbeddingService().generate_embedding(message)
                tracker = EngagementTracker(config={'user_id': user_id})
                tracker.store_pending_content(proactive_id, message, embedding=embedding)
            except Exception:
                pass

            # Store thread_id in Redis for feedback routing
            redis.setex(
                f"proactive:{user_id}:pending_thread_id",
                14400,  # 4h TTL
                thread['id'],
            )

            logger.info(
                f"{LOG_PREFIX} Enqueued proactive: id={proactive_id[:8]}, "
                f"thread={thread['id']}"
            )

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to enqueue proactive: {e}")

    def on_new_episode(self, episode_data: Dict) -> None:
        """
        Hook called when a new episode is stored.

        Checks if the episode's topic matches any active curiosity thread
        and reinforces engagement if so. Filters out self-generated episodes.

        Args:
            episode_data: Dict with at least 'embedding' and 'salience_factors'
        """
        # Ignore self-generated episodes
        source = (episode_data.get('salience_factors') or {}).get('source', '')
        if source in ('tool_reflection', 'pursuit', 'drift', 'curiosity_thread'):
            return

        embedding = episode_data.get('embedding')
        if not embedding:
            return

        try:
            from services.curiosity_thread_service import CuriosityThreadService
            from services.embedding_service import EmbeddingService
            import math

            thread_service = CuriosityThreadService()
            active_threads = thread_service.get_active_threads()

            if not active_threads:
                return

            embedding_service = EmbeddingService()

            for thread in active_threads:
                seed_topic = thread.get('seed_topic', '')
                if not seed_topic:
                    continue

                # Generate embedding for the seed topic
                try:
                    topic_embedding = embedding_service.generate_embedding(seed_topic)
                except Exception:
                    continue

                # Cosine similarity
                similarity = self._cosine_similarity(embedding, topic_embedding)
                if similarity >= 0.5:
                    thread_service.reinforce_from_conversation(seed_topic)
                    logger.debug(
                        f"{LOG_PREFIX} Reinforced thread '{seed_topic}' "
                        f"from episode (similarity={similarity:.3f})"
                    )

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} on_new_episode failed: {e}")

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity."""
        import math

        if not a or not b:
            return 0.0
        if isinstance(a, str):
            a = [float(x) for x in a.strip('[]').split(',')]
        if isinstance(b, str):
            b = [float(x) for x in b.strip('[]').split(',')]
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def curiosity_pursuit_worker(shared_state=None):
    """Module-level wrapper for multiprocessing."""
    logging.basicConfig(level=logging.INFO)
    try:
        config = ConfigService.resolve_agent_config("cognitive-drift")
        cycle_interval = config.get('curiosity_pursuit_interval', 21600)
    except Exception:
        cycle_interval = 21600

    service = CuriosityPursuitService(cycle_interval=cycle_interval)
    service.run(shared_state)
