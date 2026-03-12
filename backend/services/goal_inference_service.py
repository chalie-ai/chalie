# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Goal Inference Service — detects latent goals from interaction patterns.

Emits: goal_inferred
Consumes: Called by ReasoningLoopService during idle time
Trigger: Idle-time check (cooldown-gated, default every 6 hours)
Fail mode: Returns empty list on any failure, never raises

Scans interaction_log for topics recurring across multiple conversations.
Uses a lightweight LLM call to validate whether a pattern represents a
genuine goal vs. routine conversation. Creates PROPOSED persistent tasks
that the user must accept before autonomous pursuit.
"""

import json
import logging
from typing import Optional, List, Dict

from services.database_service import get_lightweight_db_service
from services.config_service import ConfigService
from services.background_llm_queue import create_background_llm_proxy

logger = logging.getLogger(__name__)

LOG_PREFIX = "[GOAL INFERENCE]"

# Topics that are too generic to be goals
ROUTINE_TOPICS = frozenset({
    'general', 'greeting', 'greetings', 'meta', 'chitchat', 'chat',
    'small-talk', 'thanks', 'goodbye', 'hello', 'help', 'test',
    'testing', 'debug', 'debugging', 'error', 'errors',
})


def _get_account_id(db) -> int:
    """Resolve the primary account ID. Returns 1 as a safe fallback."""
    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM master_account LIMIT 1")
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else 1
    except Exception:
        return 1


class GoalInferenceService:
    """Detects latent goals from recurring interaction patterns.

    The detection pipeline:
    1. SQL query finds topics with >= N conversations in the lookback window
    2. Filter out topics with active tasks, routine topics, and recently-proposed topics
    3. LLM validates the top candidate and generates a goal statement
    4. Creates a PROPOSED persistent task (user must accept)
    5. Emits goal_inferred signal to the reasoning loop
    """

    def __init__(self, db_service=None):
        self.db = db_service or get_lightweight_db_service()
        config = ConfigService.resolve_agent_config("cognitive-drift")
        goal_config = config.get('goal_inference', {})

        self.min_conversations = goal_config.get('min_conversations', 3)
        self.min_messages = goal_config.get('min_messages', 5)
        self.lookback_days = goal_config.get('lookback_days', 14)
        self.max_proposals_per_cycle = goal_config.get('max_proposals_per_cycle', 1)
        self.confidence_threshold = goal_config.get('confidence_threshold', 0.6)
        self.max_evidence_messages = goal_config.get('max_evidence_messages', 8)

        # LLM for goal validation — reuses the cognitive-drift provider config
        self.llm = create_background_llm_proxy("cognitive-drift")

    def detect_and_propose(self) -> List[Dict]:
        """Full pipeline: detect candidates → validate → propose.

        Returns list of proposed goals (may be empty).
        """
        proposed = []

        try:
            candidates = self._find_candidate_topics()
            if not candidates:
                logger.debug(f"{LOG_PREFIX} No candidate topics found")
                return []

            candidates = self._filter_existing_goals(candidates)
            candidates = self._filter_routine_topics(candidates)
            candidates = self._filter_recently_proposed(candidates)

            if not candidates:
                logger.debug(f"{LOG_PREFIX} All candidates filtered out")
                return []

            logger.info(
                f"{LOG_PREFIX} {len(candidates)} candidate(s) after filtering: "
                f"{[c['topic'] for c in candidates[:3]]}"
            )

            # Validate top candidates via LLM
            for candidate in candidates[:self.max_proposals_per_cycle]:
                result = self._validate_and_name(candidate)
                if result and result.get('confidence', 0) >= self.confidence_threshold:
                    success = self._propose_goal(
                        goal_statement=result['goal'],
                        topic=candidate['topic'],
                        evidence=candidate,
                        reasoning=result.get('reasoning', ''),
                    )
                    if success:
                        proposed.append({
                            'topic': candidate['topic'],
                            'goal': result['goal'],
                            'confidence': result['confidence'],
                        })
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Detection pipeline failed: {e}")

        return proposed

    def _find_candidate_topics(self) -> List[Dict]:
        """Find topics appearing in >= N conversations over lookback window."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT topic,
                           COUNT(DISTINCT thread_id) as conversation_count,
                           COUNT(*) as message_count,
                           MAX(created_at) as last_seen,
                           MIN(created_at) as first_seen
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND topic IS NOT NULL
                      AND topic != 'general'
                      AND thread_id IS NOT NULL
                      AND created_at > datetime('now', ? || ' days')
                    GROUP BY topic
                    HAVING COUNT(DISTINCT thread_id) >= ?
                       AND COUNT(*) >= ?
                    ORDER BY conversation_count DESC, message_count DESC
                    LIMIT 10
                """, (str(-self.lookback_days), self.min_conversations, self.min_messages))
                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    'topic': row[0],
                    'conversation_count': row[1],
                    'message_count': row[2],
                    'last_seen': row[3],
                    'first_seen': row[4],
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Candidate query failed: {e}")
            return []

    def _filter_existing_goals(self, candidates: List[Dict]) -> List[Dict]:
        """Remove topics that already have active persistent tasks."""
        try:
            from services.persistent_task_service import PersistentTaskService
            task_service = PersistentTaskService(self.db)
            account_id = _get_account_id(self.db)

            filtered = []
            for c in candidates:
                # Check if there's already an active task for this topic using a
                # synthetic goal string that will surface Jaccard matches
                dup = task_service.find_duplicate(account_id, f"Goal related to {c['topic']}")
                if not dup:
                    filtered.append(c)
                else:
                    logger.debug(f"{LOG_PREFIX} Skipping '{c['topic']}' — active task exists")
            return filtered
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Existing goal filter failed: {e}")
            return candidates  # Fail-open: return unfiltered

    def _filter_routine_topics(self, candidates: List[Dict]) -> List[Dict]:
        """Remove topics that are too generic to represent goals."""
        return [
            c for c in candidates
            if c['topic'].lower() not in ROUTINE_TOPICS
            and len(c['topic']) > 2  # Skip very short topic names
        ]

    def _filter_recently_proposed(self, candidates: List[Dict]) -> List[Dict]:
        """Remove topics that were proposed as goals in the last 7 days."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                # Scope strings written by _propose_goal always contain 'goal inference'
                cursor.execute("""
                    SELECT scope FROM persistent_tasks
                    WHERE created_at > datetime('now', '-7 days')
                      AND scope LIKE '%goal inference%'
                """)
                recent_scopes = [row[0] for row in cursor.fetchall()]
                cursor.close()

            filtered = []
            for c in candidates:
                topic_lower = c['topic'].lower()
                already_proposed = any(
                    topic_lower in scope.lower()
                    for scope in recent_scopes
                )
                if not already_proposed:
                    filtered.append(c)
                else:
                    logger.debug(f"{LOG_PREFIX} Skipping '{c['topic']}' — recently proposed")
            return filtered
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Recent proposal filter failed: {e}")
            return candidates  # Fail-open

    def _gather_evidence_messages(self, topic: str) -> List[str]:
        """Fetch recent user messages about this topic for LLM validation."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT payload FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND topic = ?
                      AND created_at > datetime('now', ? || ' days')
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (topic, str(-self.lookback_days), self.max_evidence_messages))
                rows = cursor.fetchall()
                cursor.close()

            messages = []
            for row in rows:
                try:
                    payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    # Extract user message text from payload
                    text = payload.get('text', payload.get('message', ''))
                    if text:
                        messages.append(text[:300])  # Cap length
                except (json.JSONDecodeError, AttributeError):
                    pass
            return messages
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Evidence gathering failed: {e}")
            return []

    def _validate_and_name(self, candidate: Dict) -> Optional[Dict]:
        """Use LLM to validate a candidate and generate a goal statement.

        Returns dict with 'goal', 'confidence', 'reasoning' or None on failure.
        """
        messages = self._gather_evidence_messages(candidate['topic'])
        if not messages:
            logger.debug(f"{LOG_PREFIX} No evidence messages for '{candidate['topic']}'")
            return None

        evidence_text = "\n".join(f"- {m}" for m in messages)

        prompt = f"""You are analyzing recurring conversation topics to detect latent goals.

Topic: {candidate['topic']}
Appeared in {candidate['conversation_count']} separate conversations over {self.lookback_days} days
Total messages: {candidate['message_count']}

Recent user messages about this topic:
{evidence_text}

Determine if this recurring topic represents a genuine goal the user is working toward — something they want to achieve, learn, plan, or resolve — as opposed to routine conversation, casual interest, or already-completed work.

A goal is something the user would benefit from having tracked, decomposed into steps, and worked on over time.

Respond with ONLY a JSON object:
{{"goal": "A clear, actionable goal statement (or null if not a goal)", "confidence": 0.0, "reasoning": "Brief explanation"}}"""

        try:
            import re
            raw = self.llm.send_message("You analyze user behavior patterns.", prompt)
            if not raw:
                return None

            # Strip thinking tags that some models emit
            cleaned = re.sub(r'<think>.*?</think>', '', raw.text, flags=re.DOTALL).strip()
            if '<think>' in cleaned:
                cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()

            result = json.loads(cleaned)

            if not isinstance(result, dict):
                return None

            goal = result.get('goal')
            confidence = float(result.get('confidence', 0))
            reasoning = result.get('reasoning', '')

            if not goal or confidence < 0.1:
                logger.info(
                    f"{LOG_PREFIX} LLM rejected '{candidate['topic']}': {reasoning[:100]}"
                )
                return None

            logger.info(
                f"{LOG_PREFIX} LLM validated '{candidate['topic']}' → "
                f"'{goal}' (confidence={confidence:.2f})"
            )
            return {'goal': goal, 'confidence': confidence, 'reasoning': reasoning}

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"{LOG_PREFIX} LLM response parse failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM validation failed: {e}")
            return None

    def _propose_goal(self, goal_statement: str, topic: str,
                      evidence: Dict, reasoning: str = '') -> bool:
        """Create a PROPOSED persistent task for the inferred goal.

        Tasks are created in PROPOSED state by default — the user must explicitly
        accept before autonomous pursuit begins.
        """
        try:
            from services.persistent_task_service import PersistentTaskService
            task_service = PersistentTaskService(self.db)
            account_id = _get_account_id(self.db)

            # Final duplicate check with the actual goal statement
            dup = task_service.find_duplicate(account_id, goal_statement)
            if dup:
                logger.info(f"{LOG_PREFIX} Duplicate task exists for '{goal_statement[:50]}...'")
                return False

            scope = (
                f"Inferred from {evidence['conversation_count']} conversations "
                f"about '{topic}' over {self.lookback_days} days (goal inference)"
            )

            task = task_service.create_task(
                account_id=account_id,
                goal=goal_statement,
                scope=scope,
                priority=7,  # Same as drift-created tasks
            )

            if not task:
                logger.warning(f"{LOG_PREFIX} Task creation failed for '{goal_statement[:50]}...'")
                return False

            task_id = task.get('id', 'unknown')

            # Store inference evidence in task progress checkpoint
            try:
                task_service.checkpoint(
                    task_id=task_id,
                    progress={
                        'inferred_from': 'goal_inference_m4',
                        'topic': topic,
                        'conversation_count': evidence['conversation_count'],
                        'message_count': evidence['message_count'],
                        'reasoning': reasoning[:200],
                        'first_seen': evidence.get('first_seen', ''),
                        'last_seen': evidence.get('last_seen', ''),
                    },
                )
            except Exception:
                pass  # Non-fatal — task was created successfully

            # Emit goal_inferred signal to the reasoning loop
            try:
                from services.reasoning_loop_service import emit_reasoning_signal, ReasoningSignal
                emit_reasoning_signal(ReasoningSignal(
                    signal_type='goal_inferred',
                    source='goal_inference',
                    topic=topic,
                    content=goal_statement[:200],
                    activation_energy=0.7,
                ))
            except Exception:
                pass  # Non-fatal

            # Surface to user via proactive notification
            try:
                from services.output_service import OutputService
                OutputService().enqueue_proactive(
                    topic=topic,
                    response=(
                        f"I've noticed you keep coming back to **{topic}** — "
                        f"it's appeared in {evidence['conversation_count']} conversations recently. "
                        f"I think there might be a goal here: *{goal_statement}*\n\n"
                        f"I've created a proposed task for this. "
                        f"You can accept, modify, or dismiss it."
                    ),
                    source='goal_inference',
                )
            except Exception:
                pass  # Non-fatal — task still created

            logger.info(
                f"{LOG_PREFIX} Proposed goal: '{goal_statement}' "
                f"(task={task_id}, topic={topic})"
            )
            return True

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Goal proposal failed: {e}")
            return False
