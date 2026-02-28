"""
Autobiography Service - Synthesis of Chalie's self-narrative from memory layers.

Background service that periodically synthesizes prose from episode, trait,
concept, and relationship data via LLM. Supports incremental updates with
concurrency protection via PostgreSQL advisory locks.
"""

import time
import logging
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from sqlalchemy import text

logger = logging.getLogger(__name__)


class AutobiographyService:
    """Manages autobiography synthesis and retrieval."""

    def __init__(self, db_service):
        """
        Initialize autobiography service.

        Args:
            db_service: DatabaseService instance for database access
        """
        self.db = db_service
        logger.info("[AUTOBIOGRAPHY] Service initialized")

    def get_current_narrative(self, user_id: str = "primary") -> Optional[Dict[str, Any]]:
        """
        Fetch the latest version of Chalie's self-narrative.

        Args:
            user_id: User identifier (default: "primary")

        Returns:
            Dict with narrative, version, created_at, or None if not synthesized
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, version, narrative, created_at, episodes_since
                    FROM autobiography
                    WHERE user_id = :user_id
                    ORDER BY version DESC
                    LIMIT 1
                    """),
                    {"user_id": user_id}
                )
                row = result.fetchone()

                if not row:
                    return None

                return {
                    "id": str(row[0]),
                    "version": row[1],
                    "narrative": row[2],
                    "created_at": row[3],
                    "episodes_since": row[4]
                }
        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY] Error fetching narrative: {e}")
            return None

    def should_synthesize(self, user_id: str = "primary") -> bool:
        """
        Check if enough new material exists to warrant synthesis.

        Returns True if:
        - No autobiography exists and ≥5 total episodes exist, OR
        - Autobiography exists and ≥3 new episodes since last cursor

        Args:
            user_id: User identifier (default: "primary")

        Returns:
            True if synthesis is warranted
        """
        try:
            with self.db.get_session() as session:
                # Check for existing autobiography
                result = session.execute(
                    text("""
                    SELECT version, episode_cursor, episodes_since
                    FROM autobiography
                    WHERE user_id = :user_id
                    ORDER BY version DESC
                    LIMIT 1
                    """),
                    {"user_id": user_id}
                )
                current = result.fetchone()

                # Count total episodes
                result = session.execute(
                    text("SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"),
                    {}
                )
                total_episodes = result.scalar() or 0

                if not current:
                    # No autobiography yet — need ≥5 episodes
                    return total_episodes >= 5

                # Autobiography exists — check for new episodes
                if current[1] is None:
                    # No cursor set (shouldn't happen, but handle it)
                    return False

                result = session.execute(
                    text("""
                    SELECT COUNT(*) FROM episodes
                    WHERE created_at > :cursor AND deleted_at IS NULL
                    """),
                    {"cursor": current[1]}
                )
                new_episode_count = result.scalar() or 0

                return new_episode_count >= 3
        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY] Error checking synthesis threshold: {e}")
            return False

    def gather_synthesis_inputs(
        self,
        user_id: str = "primary",
        since_cursor: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Gather all inputs for synthesis (episodes, traits, concepts, relationships).

        Args:
            user_id: User identifier
            since_cursor: If set, only include episodes after this timestamp

        Returns:
            Dict with episodes, traits, concepts, relationships
        """
        try:
            with self.db.get_session() as session:
                inputs = {
                    "episodes": [],
                    "traits": [],
                    "concepts": [],
                    "relationships": [],
                }

                # Gather episodes (≤50, sorted by salience DESC)
                if since_cursor:
                    query = text("""
                        SELECT gist, action, outcome, emotion, salience, topic, created_at
                        FROM episodes
                        WHERE created_at > :cursor AND deleted_at IS NULL
                        ORDER BY salience DESC
                        LIMIT 50
                    """)
                    params = {"cursor": since_cursor}
                else:
                    query = text("""
                        SELECT gist, action, outcome, emotion, salience, topic, created_at
                        FROM episodes
                        WHERE deleted_at IS NULL
                        ORDER BY salience DESC
                        LIMIT 50
                    """)
                    params = {}

                result = session.execute(query, params)
                for row in result.fetchall():
                    gist = row[0] if row[0] else ""
                    # Truncate gist to 500 chars
                    if len(gist) > 500:
                        gist = gist[:500] + "..."

                    inputs["episodes"].append({
                        "gist": gist,
                        "action": row[1],
                        "outcome": row[2],
                        "emotion": row[3],
                        "salience": row[4],
                        "topic": row[5],
                        "created_at": row[6].isoformat() if row[6] else None
                    })

                # Gather traits (confidence > 0.3)
                result = session.execute(
                    text("""
                    SELECT trait_key, trait_value, category, confidence, reinforcement_count
                    FROM user_traits
                    WHERE user_id = :user_id AND confidence > 0.3
                    ORDER BY confidence DESC
                    """),
                    {"user_id": user_id}
                )
                for row in result.fetchall():
                    inputs["traits"].append({
                        "key": row[0],
                        "value": row[1],
                        "category": row[2],
                        "confidence": row[3],
                        "reinforcement_count": row[4]
                    })

                # Gather concepts (top 30 by strength)
                result = session.execute(
                    text("""
                    SELECT concept_name, concept_type, definition, domain, strength
                    FROM semantic_concepts
                    WHERE deleted_at IS NULL
                    ORDER BY strength DESC
                    LIMIT 30
                    """),
                    {}
                )
                for row in result.fetchall():
                    inputs["concepts"].append({
                        "name": row[0],
                        "type": row[1],
                        "definition": row[2],
                        "domain": row[3],
                        "strength": row[4]
                    })

                # Gather relationships (join to resolve UUIDs → names)
                result = session.execute(
                    text("""
                    SELECT sc1.concept_name, sc2.concept_name, sr.relationship_type, sr.strength
                    FROM semantic_relationships sr
                    JOIN semantic_concepts sc1 ON sr.source_concept_id = sc1.id
                    JOIN semantic_concepts sc2 ON sr.target_concept_id = sc2.id
                    WHERE sr.deleted_at IS NULL
                    ORDER BY sr.strength DESC
                    LIMIT 50
                    """),
                    {}
                )
                for row in result.fetchall():
                    inputs["relationships"].append({
                        "source": row[0],
                        "target": row[1],
                        "type": row[2],
                        "strength": row[3]
                    })

                return inputs
        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY] Error gathering synthesis inputs: {e}")
            return {
                "episodes": [],
                "traits": [],
                "concepts": [],
                "relationships": [],
            }

    def synthesize(self, user_id: str = "primary") -> bool:
        """
        Full synthesis pipeline: acquire lock → gather → prompt → LLM → store.

        Acquires a PostgreSQL advisory lock to prevent concurrent synthesis.

        Args:
            user_id: User identifier

        Returns:
            True if synthesis succeeded, False otherwise
        """
        try:
            with self.db.get_session() as session:
                # Try to acquire advisory lock (non-blocking)
                lock_result = session.execute(
                    text("SELECT pg_try_advisory_lock(hashtext('autobiography'))")
                )
                lock_acquired = lock_result.scalar()

                if not lock_acquired:
                    logger.info("[AUTOBIOGRAPHY] Another worker is synthesizing, skipping")
                    return False

                try:
                    # Get current narrative for context
                    current = self.get_current_narrative(user_id)
                    since_cursor = None

                    if current:
                        # Incremental update — fetch episodes since last cursor
                        result = session.execute(
                            text("""
                            SELECT episode_cursor FROM autobiography
                            WHERE user_id = :user_id
                            ORDER BY version DESC
                            LIMIT 1
                            """),
                            {"user_id": user_id}
                        )
                        cursor_row = result.fetchone()
                        if cursor_row and cursor_row[0]:
                            since_cursor = cursor_row[0]

                    # Gather inputs
                    inputs = self.gather_synthesis_inputs(user_id, since_cursor)

                    if not inputs["episodes"]:
                        logger.debug("[AUTOBIOGRAPHY] No episodes to synthesize")
                        return False

                    # Build and execute synthesis prompt
                    start_time = time.time()
                    narrative = self._synthesize_via_llm(inputs, current)
                    synthesis_ms = int((time.time() - start_time) * 1000)

                    if not narrative:
                        logger.error("[AUTOBIOGRAPHY] LLM synthesis returned empty")
                        return False

                    # Get newest episode timestamp for cursor
                    result = session.execute(
                        text("""
                        SELECT MAX(created_at) FROM episodes WHERE deleted_at IS NULL
                        """),
                        {}
                    )
                    newest_episode = result.scalar()

                    # Store new version
                    self._store_narrative(
                        session,
                        user_id,
                        narrative,
                        newest_episode,
                        len(inputs["episodes"]),
                        synthesis_ms
                    )

                    logger.info(
                        f"[AUTOBIOGRAPHY] Synthesis complete: "
                        f"v{self.get_current_narrative(user_id)['version']} "
                        f"({synthesis_ms}ms)"
                    )

                    # Post-synthesis: compute growth delta and reinforce stable traits (non-fatal)
                    try:
                        from services.autobiography_delta_service import AutobiographyDeltaService
                        from sqlalchemy import text as _text
                        import json as _json

                        delta_service = AutobiographyDeltaService(self.db)

                        # Compute growth delta
                        delta = delta_service.compute_growth_delta(user_id)
                        if delta and delta.get('section_deltas'):
                            # Store delta_summary on latest autobiography row
                            with self.db.get_session() as delta_session:
                                delta_session.execute(
                                    _text("""
                                    UPDATE autobiography
                                    SET delta_summary = :delta
                                    WHERE user_id = :user_id
                                      AND version = (
                                          SELECT MAX(version) FROM autobiography
                                          WHERE user_id = :user_id
                                      )
                                    """),
                                    {"user_id": user_id, "delta": _json.dumps(delta)}
                                )
                                delta_session.commit()
                            logger.info(
                                f"[AUTOBIOGRAPHY] Delta computed: "
                                f"{len(delta['section_deltas'])} sections changed"
                            )

                        # Reinforce stable traits
                        delta_service.reinforce_stable_traits(user_id)

                    except Exception as de:
                        logger.warning(f"[AUTOBIOGRAPHY] Delta computation non-fatal error: {de}")

                    return True

                finally:
                    # Always release lock
                    session.execute(text("SELECT pg_advisory_unlock(hashtext('autobiography'))"))

        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY] Synthesis failed: {e}", exc_info=True)
            return False

    def _synthesize_via_llm(
        self,
        inputs: Dict[str, Any],
        current_narrative: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Call LLM to synthesize narrative from inputs.

        Args:
            inputs: Dict with episodes, traits, concepts, relationships
            current_narrative: Current narrative for incremental updates

        Returns:
            Synthesized narrative string or None on failure
        """
        try:
            from services.background_llm_queue import create_background_llm_proxy
            from services.config_service import ConfigService

            llm = create_background_llm_proxy("autobiography")

            # Read synthesis prompt
            import os
            prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "prompts",
                "autobiography-synthesis.md"
            )

            with open(prompt_path, 'r') as f:
                system_prompt = f.read()

            # Build user prompt with input data
            user_prompt = self._build_synthesis_prompt(inputs, current_narrative)

            # Call LLM
            response = llm.send_message(system_prompt, user_prompt)
            if response:
                return response.text.strip()

            return None

        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY] LLM synthesis error: {e}", exc_info=True)
            return None

    def _build_synthesis_prompt(
        self,
        inputs: Dict[str, Any],
        current: Optional[Dict[str, Any]]
    ) -> str:
        """
        Format memory data and current narrative for LLM synthesis.

        Args:
            inputs: Gathered episodes, traits, concepts, relationships
            current: Current autobiography version

        Returns:
            Formatted prompt string
        """
        lines = []

        if current:
            lines.append("## Current Narrative (for incremental update)\n")
            lines.append(current["narrative"])
            lines.append("\n\n## New Episodes Since Last Synthesis\n")
        else:
            lines.append("## New Episodes\n")

        for ep in inputs["episodes"]:
            lines.append(f"- {ep['gist']} (emotion: {ep['emotion']}, topic: {ep['topic']})")

        if inputs["traits"]:
            lines.append("\n## Observed Traits\n")
            for trait in inputs["traits"]:
                lines.append(
                    f"- {trait['key']}: {trait['value']} "
                    f"(confidence: {trait['confidence']:.2f}, category: {trait['category']})"
                )

        if inputs["concepts"]:
            lines.append("\n## Key Concepts\n")
            for concept in inputs["concepts"]:
                lines.append(
                    f"- {concept['name']}: {concept['definition']} "
                    f"(strength: {concept['strength']:.2f}, domain: {concept['domain']})"
                )

        return "\n".join(lines)

    def _compute_section_hashes(self, narrative: str) -> dict:
        """
        Compute SHA-256 hashes for each ## section in the narrative.

        Args:
            narrative: Synthesized narrative text

        Returns:
            Dict mapping section_name (lowercase snake_case) → hash string
        """
        import re as _re

        hashes = {}
        parts = _re.split(r'^(##\s+.+)$', narrative, flags=_re.MULTILINE)

        current_section = None
        current_content = []

        for part in parts:
            if part.startswith('## '):
                if current_section is not None:
                    content = '\n'.join(current_content).strip()
                    hashes[current_section] = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                current_section = part.lstrip('#').strip().lower().replace(' ', '_')
                current_content = []
            else:
                if current_section is not None:
                    current_content.append(part)

        if current_section is not None:
            content = '\n'.join(current_content).strip()
            hashes[current_section] = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

        return hashes

    def _store_narrative(
        self,
        session,
        user_id: str,
        narrative: str,
        episode_cursor: Optional[datetime],
        episodes_since: int,
        synthesis_ms: int
    ) -> None:
        """
        Insert new autobiography version into database.

        Args:
            session: SQLAlchemy session
            user_id: User identifier
            narrative: Synthesized narrative text
            episode_cursor: Timestamp of newest episode included
            episodes_since: Count of episodes in this synthesis
            synthesis_ms: Milliseconds taken for synthesis
        """
        import json as _json

        # Get next version number
        result = session.execute(
            text("SELECT MAX(version) FROM autobiography WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        max_version = result.scalar() or 0
        next_version = max_version + 1

        # Compute section hashes for delta tracking
        section_hashes = self._compute_section_hashes(narrative)

        session.execute(
            text("""
            INSERT INTO autobiography
            (user_id, version, narrative, episode_cursor, episodes_since, synthesis_ms, section_hashes)
            VALUES (:user_id, :version, :narrative, :cursor, :episodes, :synthesis_ms, :section_hashes)
            """),
            {
                "user_id": user_id,
                "version": next_version,
                "narrative": narrative,
                "cursor": episode_cursor,
                "episodes": episodes_since,
                "synthesis_ms": synthesis_ms,
                "section_hashes": _json.dumps(section_hashes),
            }
        )
        session.commit()


def autobiography_synthesis_worker(shared_state=None) -> None:
    """
    Background worker: periodically synthesize autobiography.

    Infinite loop: sleeps 6 hours, checks should_synthesize(), runs if needed.

    Args:
        shared_state: Optional shared state dict (for consumer integration)
    """
    logger.info("[AUTOBIOGRAPHY] Worker started")

    synthesis_interval = 6 * 3600  # 6 hours
    check_interval = 300  # Check every 5 minutes if synthesis needed

    try:
        from services.database_service import get_lightweight_db_service

        db = get_lightweight_db_service()
        service = AutobiographyService(db)

        last_synthesis = time.time()

        while True:
            try:
                time.sleep(check_interval)

                # Check if it's time to attempt synthesis
                if time.time() - last_synthesis < synthesis_interval:
                    continue

                # Check if synthesis is needed and run if so
                if service.should_synthesize():
                    logger.info("[AUTOBIOGRAPHY] Synthesis threshold met, running synthesis...")
                    if service.synthesize():
                        last_synthesis = time.time()
                    # else: lock held by another worker, will retry next interval

            except KeyboardInterrupt:
                logger.info("[AUTOBIOGRAPHY] Worker shutting down...")
                break
            except Exception as e:
                logger.error(f"[AUTOBIOGRAPHY] Worker error: {e}", exc_info=True)
                time.sleep(60)

    except Exception as e:
        logger.error(f"[AUTOBIOGRAPHY] Worker initialization failed: {e}", exc_info=True)
