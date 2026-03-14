"""
Failure Analysis Service — Root-cause attribution and lesson extraction for the ACT loop.

Analyses failed or critic-rejected actions to extract generalisable lessons.
Lessons are stored in ``procedural_memory.context_stats["__failure_lessons"]`` with
embedding-based deduplication so semantically identical lessons are merged rather than
duplicated.

Embedding vectors are NOT stored in the JSON blob — they are recomputed on-the-fly
during dedup (the embedding service caches results internally, so this is cheap).

Intended usage pattern:
    fas = FailureAnalysisService(db_service)
    analysis = fas.analyze(failure_context)
    if analysis:
        fas.store_lesson(analysis, action_type)
"""

import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[FAILURE-ANALYSIS]"

# Minimum confidence below which analysis results are discarded.
CONFIDENCE_THRESHOLD = 0.65

# Cosine similarity threshold above which two lessons are considered duplicates.
DEDUP_SIMILARITY_THRESHOLD = 0.85

# Maximum number of lessons stored per action_name row.
MAX_LESSONS_PER_ACTION = 50

# Valid blame categories (mirrors the prompt schema).
VALID_BLAME_CATEGORIES = frozenset({
    "tool_choice",
    "input_quality",
    "stale_memory",
    "wrong_assumption",
    "external",
    "ambiguous_goal",
})

# Error signal keywords that support an "external" blame attribution.
_EXTERNAL_KEYWORDS = frozenset({
    "timeout", "timed out", "connection", "refused", "network",
    "http", "503", "502", "504", "unavailable", "unreachable",
})

# Error signal keywords that support a "tool_choice" blame attribution.
_TOOL_KEYWORDS = frozenset({
    "tool", "function", "handler", "not found", "unknown action",
    "unsupported", "invalid tool",
})

# Action type fragments that hint at memory operations.
_MEMORY_ACTION_FRAGMENTS = frozenset({
    "recall", "memory", "memorize", "remember", "retrieve",
    "store", "lookup", "search_memory",
})


class FailureAnalysisService:
    """
    Analyses ACT-loop failures and stores generalisable lessons in procedural memory.

    Follows the same dependency-injection pattern as :class:`ProceduralMemoryService`
    (receives a ``DatabaseService`` instance in the constructor).  The LLM is lazily
    instantiated on first use via :meth:`_get_llm`.
    """

    def __init__(self, db_service: DatabaseService):
        """
        Initialise the failure analysis service.

        Args:
            db_service: Injected :class:`DatabaseService` for all SQLite I/O.
        """
        self.db_service = db_service
        self._llm = None
        self._prompt_template: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────

    def analyze(self, failure_context: dict) -> Optional[dict]:
        """
        Invoke the failure-analysis LLM to produce a structured root-cause assessment.

        Returns ``None`` when:
        - The LLM response cannot be parsed as valid JSON.
        - Sanity-check rules reduce ``confidence`` below :data:`CONFIDENCE_THRESHOLD`.
        - Any unexpected exception is raised (errors are logged, never re-raised).

        Args:
            failure_context: Dict with keys ``original_request``, ``action_type``,
                ``action_intent``, ``action_result``, ``error_signals``, and
                optionally ``plan_context``.

        Returns:
            Structured analysis dict (``blame``, ``root_cause``, ``lesson``,
            ``affected_skill``, ``severity``, ``confidence``, ``generalizable``)
            or ``None`` if analysis is inconclusive.
        """
        try:
            llm = self._get_llm()
            prompt = self._build_prompt(failure_context)
            response_text = llm.send_message("", prompt).text
            analysis = self._parse_response(response_text)
            if analysis is None:
                logger.debug(f"{LOG_PREFIX} Could not parse LLM response")
                return None

            error_signals = failure_context.get("error_signals", {})
            analysis = self._sanity_check(analysis, error_signals)

            if analysis.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"{LOG_PREFIX} Discarding low-confidence analysis "
                    f"(confidence={analysis.get('confidence', 0):.2f})"
                )
                return None

            return analysis

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} analyze() failed: {e}")
            return None

    def store_lesson(self, analysis: dict, action_name: str) -> bool:
        """
        Persist a failure lesson into ``procedural_memory.context_stats["__failure_lessons"]``.

        Deduplication is embedding-based: if any existing lesson has cosine similarity
        ≥ :data:`DEDUP_SIMILARITY_THRESHOLD` to the new lesson, the existing entry is
        updated (``times_seen`` incremented, ``confidence`` and ``updated_at`` refreshed)
        rather than a new record being inserted.

        Embeddings are recomputed on-the-fly and are NOT stored in the JSON blob,
        keeping the ``context_stats`` column compact.

        The lesson list is capped at :data:`MAX_LESSONS_PER_ACTION` entries.  When the
        cap is exceeded the entry with the lowest ``times_seen`` (ties broken by oldest
        ``created_at``) is evicted.

        Args:
            analysis: Structured dict returned by :meth:`analyze`.
            action_name: The ``action_name`` key used in the ``procedural_memory`` row.

        Returns:
            ``True`` if the lesson was stored or merged successfully, ``False`` on error.
        """
        lesson_text = analysis.get("lesson", "").strip()
        if not lesson_text:
            logger.debug(f"{LOG_PREFIX} store_lesson called with empty lesson text — skipped")
            return False

        lesson_hash = hashlib.md5(lesson_text.encode()).hexdigest()[:32]

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Ensure the action_name row exists.
                cursor.execute(
                    """
                    INSERT INTO procedural_memory (id, action_name, total_attempts, total_successes, weight)
                    VALUES (?, ?, 0, 0, 1.0)
                    ON CONFLICT (action_name) DO NOTHING
                    """,
                    (str(uuid.uuid4()), action_name),
                )

                # Read existing context_stats.
                cursor.execute(
                    "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                    (action_name,),
                )
                row = cursor.fetchone()
                context_stats = {}
                if row:
                    raw = row[0] if not isinstance(row, dict) else row["context_stats"]
                    if raw:
                        if isinstance(raw, str):
                            context_stats = json.loads(raw)
                        elif isinstance(raw, dict):
                            context_stats = raw

                lessons: List[dict] = context_stats.get("__failure_lessons", [])

                # Embedding-based dedup: compute similarity against existing lessons.
                new_vec = self._get_embedding_np(lesson_text)
                merged = False
                for existing in lessons:
                    existing_text = existing.get("lesson", "")
                    if not existing_text:
                        continue
                    existing_vec = self._get_embedding_np(existing_text)
                    similarity = float(np.dot(new_vec, existing_vec))
                    if similarity >= DEDUP_SIMILARITY_THRESHOLD:
                        # Merge: update in place.
                        existing["times_seen"] = existing.get("times_seen", 1) + 1
                        existing["confidence"] = max(
                            existing.get("confidence", 0.0),
                            analysis.get("confidence", 0.0),
                        )
                        from services.time_utils import utc_now
                        existing["updated_at"] = utc_now().isoformat()
                        merged = True
                        logger.debug(
                            f"{LOG_PREFIX} Merged duplicate lesson for '{action_name}' "
                            f"(similarity={similarity:.3f})"
                        )
                        break

                if not merged:
                    from services.time_utils import utc_now
                    now_iso = utc_now().isoformat()
                    new_lesson = {
                        "lesson_hash": lesson_hash,
                        "blame": analysis.get("blame", "ambiguous_goal"),
                        "root_cause": analysis.get("root_cause", ""),
                        "lesson": lesson_text,
                        "affected_skill": analysis.get("affected_skill", action_name),
                        "severity": analysis.get("severity", "minor"),
                        "confidence": analysis.get("confidence", 0.0),
                        "generalizable": analysis.get("generalizable", True),
                        "times_seen": 1,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                    lessons.append(new_lesson)
                    logger.info(
                        f"{LOG_PREFIX} Stored new lesson for '{action_name}': "
                        f"blame={new_lesson['blame']}, severity={new_lesson['severity']}"
                    )

                # Enforce cap: evict lowest times_seen, then oldest created_at.
                if len(lessons) > MAX_LESSONS_PER_ACTION:
                    lessons.sort(
                        key=lambda l: (l.get("times_seen", 1), l.get("created_at", "")),
                        reverse=False,
                    )
                    evicted = lessons.pop(0)
                    logger.debug(
                        f"{LOG_PREFIX} Evicted oldest/least-seen lesson for "
                        f"'{action_name}': {evicted.get('lesson', '')[:60]}"
                    )

                context_stats["__failure_lessons"] = lessons
                cursor.execute(
                    "UPDATE procedural_memory SET context_stats = ?, updated_at = datetime('now') "
                    "WHERE action_name = ?",
                    (json.dumps(context_stats), action_name),
                )
                cursor.close()

            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} store_lesson failed for '{action_name}': {e}")
            return False

    def get_relevant_lessons(self, action_name: str, action_context: str = "") -> list:
        """
        Retrieve failure lessons relevant to a given action type.

        Filters to lessons where ``severity == 'major'`` **or** ``times_seen >= 2``.
        Results are sorted by ``times_seen`` descending then ``updated_at`` descending,
        and only the top 3 are returned.  Internal bookkeeping fields (``lesson_hash``)
        are stripped from output.

        Args:
            action_name: The ``action_name`` key to look up in ``procedural_memory``.
            action_context: Optional contextual string (reserved for future semantic
                matching — not currently used for filtering).

        Returns:
            List of up to 3 lesson dicts, each containing ``blame``, ``root_cause``,
            ``lesson``, ``affected_skill``, ``severity``, ``confidence``,
            ``generalizable``, ``times_seen``, ``created_at``, and ``updated_at``.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                    (action_name,),
                )
                row = cursor.fetchone()
                cursor.close()

            if not row:
                return []

            raw = row[0] if not isinstance(row, dict) else row["context_stats"]
            if not raw:
                return []
            if isinstance(raw, str):
                context_stats = json.loads(raw)
            else:
                context_stats = raw

            lessons: List[dict] = context_stats.get("__failure_lessons", [])

            # Filter: major severity OR seen at least twice.
            qualifying = [
                l for l in lessons
                if l.get("severity") == "major" or l.get("times_seen", 1) >= 2
            ]

            # Sort: times_seen DESC, updated_at DESC.
            qualifying.sort(
                key=lambda l: (l.get("times_seen", 1), l.get("updated_at", "")),
                reverse=True,
            )

            # Return top 3, stripping internal fields.
            _internal = {"lesson_hash"}
            return [
                {k: v for k, v in l.items() if k not in _internal}
                for l in qualifying[:3]
            ]

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_relevant_lessons failed for '{action_name}': {e}")
            return []

    def get_stats(self) -> dict:
        """
        Aggregate failure-lesson statistics across all ``procedural_memory`` rows.

        Returns:
            Dict with keys:
            - ``blame_distribution``: Counter mapping blame category → total count.
            - ``lesson_counts_by_action``: Dict mapping action_name → lesson count.
            - ``top_lessons_by_frequency``: Top 10 lessons ordered by ``times_seen`` desc.
            - ``recent_lessons``: 10 most recently updated lessons.
            - ``total_lessons``: Total lesson count across all actions.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT action_name, context_stats FROM procedural_memory"
                )
                rows = cursor.fetchall()
                cursor.close()

            blame_dist: Dict[str, int] = {}
            lessons_by_action: Dict[str, int] = {}
            all_lessons: List[dict] = []

            for row in rows:
                if isinstance(row, dict):
                    action_name = row["action_name"]
                    raw_stats = row["context_stats"]
                else:
                    action_name = row[0]
                    raw_stats = row[1]

                if isinstance(raw_stats, str):
                    try:
                        stats = json.loads(raw_stats)
                    except json.JSONDecodeError:
                        stats = {}
                else:
                    stats = raw_stats or {}

                lessons: List[dict] = stats.get("__failure_lessons", [])
                if lessons:
                    lessons_by_action[action_name] = len(lessons)
                    for lesson in lessons:
                        blame = lesson.get("blame", "unknown")
                        blame_dist[blame] = blame_dist.get(blame, 0) + 1
                        all_lessons.append({"action_name": action_name, **lesson})

            top_lessons = sorted(
                all_lessons,
                key=lambda x: x.get("times_seen", 1),
                reverse=True,
            )[:10]

            recent_lessons = sorted(
                all_lessons,
                key=lambda x: x.get("updated_at", ""),
                reverse=True,
            )[:10]

            _internal = {"lesson_hash"}

            def _strip(lesson: dict) -> dict:
                """Remove internal bookkeeping fields from a lesson dict for output."""
                return {k: v for k, v in lesson.items() if k not in _internal}

            return {
                "blame_distribution": blame_dist,
                "lesson_counts_by_action": lessons_by_action,
                "top_lessons_by_frequency": [_strip(l) for l in top_lessons],
                "recent_lessons": [_strip(l) for l in recent_lessons],
                "total_lessons": len(all_lessons),
            }

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_stats failed: {e}")
            return {}

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_llm(self):
        """
        Return the lazily-initialised LLM service for failure-analysis evaluations.

        Uses the ``failure-analysis`` agent config (lightweight/triage-tier model).
        The instance is cached on ``self._llm`` so subsequent calls are free.

        Returns:
            LLM service instance configured for failure analysis.
        """
        if self._llm is None:
            from services.llm_service import create_llm_service
            from services.config_service import ConfigService
            agent_cfg = ConfigService.resolve_agent_config("failure-analysis")
            self._llm = create_llm_service(agent_cfg)
        return self._llm

    def _load_prompt(self) -> str:
        """
        Load the failure-analysis prompt template from ``prompts/failure-analysis.md``.

        The result is cached on ``self._prompt_template``; subsequent calls return
        the cached string without touching the filesystem.

        Returns:
            Prompt template string with ``{{...}}`` placeholders for substitution.
        """
        if self._prompt_template is None:
            prompts_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "prompts"
            )
            path = os.path.join(prompts_dir, "failure-analysis.md")
            with open(path, "r") as fh:
                self._prompt_template = fh.read()
        return self._prompt_template

    def _build_prompt(self, failure_context: dict) -> str:
        """
        Render the prompt template with values from ``failure_context``.

        Substitutes ``{{original_request}}``, ``{{action_type}}``,
        ``{{action_intent}}``, ``{{action_result}}``, ``{{error_signals}}``,
        and ``{{plan_context}}`` placeholders.

        Args:
            failure_context: Dict containing failure details.  Missing keys default
                to empty string / empty dict.

        Returns:
            Fully rendered prompt string ready for the LLM.
        """
        template = self._load_prompt()

        def _j(v: Any) -> str:
            """JSON-encode a value, falling back to str() for non-serialisable objects."""
            if isinstance(v, (dict, list)):
                return json.dumps(v, default=str, indent=2)
            return str(v)

        return (
            template
            .replace("{{original_request}}", str(failure_context.get("original_request", "")))
            .replace("{{action_type}}", str(failure_context.get("action_type", "")))
            .replace("{{action_intent}}", _j(failure_context.get("action_intent", {})))
            .replace("{{action_result}}", _j(failure_context.get("action_result", {})))
            .replace("{{error_signals}}", _j(failure_context.get("error_signals", {})))
            .replace("{{plan_context}}", str(failure_context.get("plan_context", "")))
        )

    def _parse_response(self, response_text: str) -> Optional[dict]:
        """
        Parse the LLM's raw text response into a structured dict.

        Attempts three strategies in order:
        1. Direct :func:`json.loads`.
        2. Extract JSON from a markdown code fence (`` ```json ... ``` ``).
        3. Keyword detection: if ``blame`` or ``lesson`` appear in the text, try to
           extract the first ``{...}`` block via regex.

        Args:
            response_text: Raw text response from the LLM call.

        Returns:
            Parsed dict or ``None`` if no valid JSON could be extracted.
        """
        # Strategy 1: direct parse.
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from markdown fence.
        try:
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 3: keyword-hinted raw JSON block.
        lower = response_text.lower()
        if any(kw in lower for kw in ("blame", "lesson", "root_cause")):
            try:
                import re
                match = re.search(r"\{[\s\S]*\}", response_text)
                if match:
                    return json.loads(match.group(0))
            except (json.JSONDecodeError, AttributeError):
                pass

        return None

    def _sanity_check(self, analysis: dict, error_signals: dict) -> dict:
        """
        Apply heuristic rules to validate the LLM's blame attribution.

        Each failed rule reduces ``confidence`` by 0.20.  When ``severity`` is
        ``'major'`` but the adjusted confidence is below 0.50, the severity is
        demoted to ``'minor'``.

        Rules:
        - ``tool_choice`` blame requires a recognisable tool-related term in the
          error signals or a non-success status.
        - ``external`` blame requires timeout / network / HTTP keywords in the
          error signal text.
        - ``stale_memory`` blame requires a memory-related action type.
        - ``input_quality`` blame requires non-empty ``action_intent`` content,
          suggesting user-provided input was present.
        - ``wrong_assumption`` blame requires a non-empty ``root_cause`` string.

        Args:
            analysis: Structured dict from :meth:`_parse_response`.
            error_signals: Dict of error indicators (``status``, ``error_text``, etc.)

        Returns:
            The mutated analysis dict with adjusted ``confidence`` and possibly
            demoted ``severity``.
        """
        analysis = dict(analysis)  # shallow copy — avoid mutating caller's dict
        blame = analysis.get("blame", "")
        confidence = float(analysis.get("confidence", 0.7))

        # Build a searchable string from all error signal values.
        signal_text = " ".join(str(v) for v in error_signals.values()).lower()
        action_type = analysis.get("affected_skill", "").lower()
        status = str(error_signals.get("status", "")).lower()

        # Rule 1: tool_choice requires tool-related evidence.
        if blame == "tool_choice":
            has_tool_signal = (
                any(kw in signal_text for kw in _TOOL_KEYWORDS)
                or status not in ("success", "")
            )
            if not has_tool_signal:
                confidence -= 0.20
                logger.debug(
                    f"{LOG_PREFIX} Sanity: tool_choice blame lacks tool evidence "
                    f"(confidence → {confidence:.2f})"
                )

        # Rule 2: external requires network/timeout evidence.
        elif blame == "external":
            has_external_signal = any(kw in signal_text for kw in _EXTERNAL_KEYWORDS)
            if not has_external_signal:
                confidence -= 0.20
                logger.debug(
                    f"{LOG_PREFIX} Sanity: external blame lacks network evidence "
                    f"(confidence → {confidence:.2f})"
                )

        # Rule 3: stale_memory requires a memory-related action type.
        elif blame == "stale_memory":
            is_memory_action = any(
                frag in action_type for frag in _MEMORY_ACTION_FRAGMENTS
            )
            if not is_memory_action:
                confidence -= 0.20
                logger.debug(
                    f"{LOG_PREFIX} Sanity: stale_memory blame for non-memory action "
                    f"'{action_type}' (confidence → {confidence:.2f})"
                )

        # Rule 4: input_quality requires non-empty action intent.
        elif blame == "input_quality":
            action_intent = analysis.get("root_cause", "")
            if not action_intent or len(action_intent.strip()) < 5:
                confidence -= 0.10
                logger.debug(
                    f"{LOG_PREFIX} Sanity: input_quality blame with thin root_cause "
                    f"(confidence → {confidence:.2f})"
                )

        # Rule 5: wrong_assumption requires a non-empty root_cause.
        elif blame == "wrong_assumption":
            root_cause = analysis.get("root_cause", "").strip()
            if len(root_cause) < 10:
                confidence -= 0.10
                logger.debug(
                    f"{LOG_PREFIX} Sanity: wrong_assumption blame with thin root_cause "
                    f"(confidence → {confidence:.2f})"
                )

        # Clamp confidence to [0, 1].
        confidence = max(0.0, min(1.0, confidence))
        analysis["confidence"] = confidence

        # Severity demotion: major requires at least 0.50 confidence.
        if analysis.get("severity") == "major" and confidence < 0.50:
            analysis["severity"] = "minor"
            logger.debug(
                f"{LOG_PREFIX} Sanity: demoted severity major→minor "
                f"(confidence={confidence:.2f} < 0.50)"
            )

        return analysis

    def _get_embedding_np(self, text: str) -> np.ndarray:
        """
        Return an L2-normalised embedding for ``text`` as a numpy array.

        Delegates to :func:`services.embedding_service.get_embedding_service` which
        caches results internally.  Falls back to a zero vector on error, which will
        never exceed the cosine similarity threshold (dot product of zero vector is 0).

        Args:
            text: The text string to embed.

        Returns:
            1-D L2-normalised numpy float32 array.  Shape is model-dependent.
        """
        try:
            from services.embedding_service import get_embedding_service
            emb_svc = get_embedding_service()
            return emb_svc.generate_embedding_np(text)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Embedding generation failed, using zero vector: {e}")
            return np.zeros(256, dtype=np.float32)
