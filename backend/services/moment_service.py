"""
Moment Service — pinned message bookmarks with LLM-enriched context.

Stores user-pinned Chalie responses as permanent, semantically searchable
moments. Each moment is enriched in the background with surrounding gists
and an LLM-generated summary. Pinning boosts related episode salience.
"""

import json
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MOMENTS]"


class MomentService:
    """Manages moment CRUD, search, enrichment, and salience boosting."""

    def __init__(self, db_service):
        self.db = db_service

    # ─────────────────────────────────────────────
    # Create
    # ─────────────────────────────────────────────

    def create_moment(
        self,
        message_text: str,
        exchange_id: Optional[str] = None,
        topic: Optional[str] = None,
        thread_id: Optional[str] = None,
        title: Optional[str] = None,
        user_id: str = "primary",
    ) -> Dict[str, Any]:
        """
        Pin a message as a moment.

        Returns dict with moment data. If a near-duplicate exists,
        returns {duplicate: True, existing_id: ...} alongside the existing moment.
        """
        moment_id = secrets.token_hex(4)

        # Duplicate check via embedding similarity
        try:
            from services.embedding_service import get_embedding_service
            embedding = get_embedding_service().generate_embedding(message_text)

            dup = self._check_duplicate(embedding, user_id)
            if dup:
                logger.info(f"{LOG_PREFIX} Near-duplicate detected (existing={dup['id']})")
                return {"duplicate": True, "existing_id": dup["id"], **dup}
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding/duplicate check failed: {e}")
            embedding = None

        # Auto-generate title if not provided
        if not title:
            title = self._generate_title(message_text)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO moments
                        (id, user_id, title, message_text, exchange_id, topic,
                         thread_id, embedding, status, pinned_at,
                         metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                            'enriching', NOW(), '{}'::jsonb, NOW(), NOW())
                """, (
                    moment_id, user_id, title, message_text,
                    exchange_id, topic, thread_id,
                    embedding,
                ))
                cursor.close()

            logger.info(f"{LOG_PREFIX} Created moment '{title}' (id={moment_id})")

            # Boost related episodes
            try:
                self.boost_related_salience(moment_id)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Salience boost failed (non-fatal): {e}")

            return self.get_moment(moment_id)

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to create moment: {e}")
            raise

    # ─────────────────────────────────────────────
    # Read
    # ─────────────────────────────────────────────

    def get_moment(self, moment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single moment by ID."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, title, message_text, exchange_id, topic,
                           thread_id, gists, summary, status, pinned_at, sealed_at,
                           last_enriched_at, metadata, created_at, updated_at
                    FROM moments
                    WHERE id = %s AND deleted_at IS NULL
                """, (moment_id,))
                row = cursor.fetchone()
                cursor.close()

            if not row:
                return None
            return self._row_to_dict(row)

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_moment failed: {e}")
            return None

    def get_all_moments(self, user_id: str = "primary") -> List[Dict[str, Any]]:
        """Get all active moments ordered by pinned_at DESC."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, title, message_text, exchange_id, topic,
                           thread_id, gists, summary, status, pinned_at, sealed_at,
                           last_enriched_at, metadata, created_at, updated_at
                    FROM moments
                    WHERE user_id = %s AND status != 'forgotten' AND deleted_at IS NULL
                    ORDER BY pinned_at DESC
                """, (user_id,))
                rows = cursor.fetchall()
                cursor.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_all_moments failed: {e}")
            return []

    def get_enriching_moments(self) -> List[Dict[str, Any]]:
        """Get all moments that need enrichment."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, title, message_text, exchange_id, topic,
                           thread_id, gists, summary, status, pinned_at, sealed_at,
                           last_enriched_at, metadata, created_at, updated_at
                    FROM moments
                    WHERE status = 'enriching' AND deleted_at IS NULL
                    ORDER BY pinned_at ASC
                """)
                rows = cursor.fetchall()
                cursor.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_enriching_moments failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Forget / Delete
    # ─────────────────────────────────────────────

    def forget_moment(self, moment_id: str) -> bool:
        """
        Forget a moment — sets status to 'forgotten', reverses salience boosts.
        Moment data preserved but invisible to all retrieval paths.
        """
        moment = self.get_moment(moment_id)
        if not moment:
            return False

        try:
            # Reverse salience boosts
            self._reverse_salience_boost(moment)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE moments
                    SET status = 'forgotten', updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (moment_id,))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                logger.info(f"{LOG_PREFIX} Forgot moment (id={moment_id})")
            return updated

        except Exception as e:
            logger.error(f"{LOG_PREFIX} forget_moment failed: {e}")
            return False

    def delete_moment(self, moment_id: str) -> bool:
        """Hard soft-delete (set deleted_at). Admin/future use only."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE moments
                    SET deleted_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (moment_id,))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                logger.info(f"{LOG_PREFIX} Deleted moment (id={moment_id})")
            return updated

        except Exception as e:
            logger.error(f"{LOG_PREFIX} delete_moment failed: {e}")
            return False

    # ─────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────

    def search_moments(
        self,
        query: str,
        limit: int = 3,
        user_id: str = "primary",
    ) -> List[Dict[str, Any]]:
        """Semantic search via pgvector cosine similarity."""
        try:
            from services.embedding_service import get_embedding_service
            query_embedding = get_embedding_service().generate_embedding(query)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, title, message_text, exchange_id, topic,
                           thread_id, gists, summary, status, pinned_at, sealed_at,
                           last_enriched_at, metadata, created_at, updated_at,
                           embedding <=> %s::vector AS distance
                    FROM moments
                    WHERE user_id = %s
                      AND status != 'forgotten'
                      AND deleted_at IS NULL
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, user_id, query_embedding, limit))
                rows = cursor.fetchall()
                cursor.close()

            results = []
            for row in rows:
                moment = self._row_to_dict(row[:16])
                moment["distance"] = row[16]
                results.append(moment)
            return results

        except Exception as e:
            logger.error(f"{LOG_PREFIX} search_moments failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Enrichment
    # ─────────────────────────────────────────────

    def enrich_moment(self, moment_id: str, new_gists: List[str]) -> bool:
        """Merge new gists into moment (Jaccard dedup at 0.7 threshold)."""
        moment = self.get_moment(moment_id)
        if not moment:
            return False

        existing_gists = moment.get("gists") or []
        merged = list(existing_gists)

        for gist in new_gists:
            if not self._is_duplicate_gist(gist, merged, threshold=0.7):
                merged.append(gist)

        if len(merged) == len(existing_gists):
            return False  # No new gists added

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE moments
                    SET gists = %s, last_enriched_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (json.dumps(merged), moment_id))
                cursor.close()

            logger.info(f"{LOG_PREFIX} Enriched moment {moment_id} ({len(existing_gists)} → {len(merged)} gists)")
            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} enrich_moment failed: {e}")
            return False

    def generate_summary(self, moment_id: str) -> Optional[str]:
        """
        LLM-generated summary from pinned message + gists.
        Regenerates embedding from message_text + summary.
        """
        moment = self.get_moment(moment_id)
        if not moment:
            return None

        message_text = moment["message_text"]
        gists = moment.get("gists") or []

        gist_text = "\n".join(f"- {g}" for g in gists) if gists else "No surrounding context available."

        system_prompt = (
            "You generate concise moment summaries. Write 1-2 sentences that capture "
            "WHY this mattered to the person. Be direct and specific — never use generic "
            "framing like 'User discussed...', 'This conversation covered...', or "
            "'The assistant provided...'. Instead produce actionable specifics. "
            "Example good: 'Quick air fryer lemon chicken with crisp skin and minimal prep.' "
            "Example bad: 'Discussion about chicken recipes.'"
        )
        user_message = (
            f"Pinned message:\n{message_text}\n\n"
            f"Surrounding conversation context:\n{gist_text}\n\n"
            f"Write a 1-2 sentence summary of why this moment was worth remembering."
        )

        try:
            from services.llm_service import create_refreshable_llm_service
            llm = create_refreshable_llm_service("moment-enrichment")
            response = llm.send_message(system_prompt, user_message)
            summary = response.text.strip()

            # Regenerate embedding from enriched content
            from services.embedding_service import get_embedding_service
            combined_text = f"{message_text} {summary}"
            embedding = get_embedding_service().generate_embedding(combined_text)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE moments
                    SET summary = %s, embedding = %s, updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (summary, embedding, moment_id))
                cursor.close()

            logger.info(f"{LOG_PREFIX} Generated summary for moment {moment_id}")
            return summary

        except Exception as e:
            logger.error(f"{LOG_PREFIX} generate_summary failed: {e}")
            return None

    def seal_moment(self, moment_id: str) -> bool:
        """Seal a moment — marks enrichment complete."""
        moment = self.get_moment(moment_id)
        if not moment:
            return False

        gists = moment.get("gists") or []

        # Sparse-seal guard: if < 2 gists, keep summary minimal
        if len(gists) < 2 and not moment.get("summary"):
            logger.info(f"{LOG_PREFIX} Sparse-sealing moment {moment_id} (< 2 gists, no summary)")

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE moments
                    SET status = 'sealed', sealed_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (moment_id,))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                logger.info(f"{LOG_PREFIX} Sealed moment {moment_id}")

                # Final semantic salience boost on seal
                try:
                    self._semantic_salience_boost(moment_id)
                except Exception as e:
                    logger.warning(f"{LOG_PREFIX} Seal salience boost failed (non-fatal): {e}")

            return updated

        except Exception as e:
            logger.error(f"{LOG_PREFIX} seal_moment failed: {e}")
            return False

    # ─────────────────────────────────────────────
    # Salience
    # ─────────────────────────────────────────────

    def boost_related_salience(self, moment_id: str) -> int:
        """
        Boost salience of episodes matching topic + time window.
        Records pre-boost salience in moment metadata for safe reversal.
        """
        moment = self.get_moment(moment_id)
        if not moment:
            return 0

        topic = moment.get("topic")
        pinned_at = moment.get("pinned_at")
        if not topic or not pinned_at:
            return 0

        window_start = pinned_at - timedelta(hours=4)
        window_end = pinned_at + timedelta(hours=4)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Find matching episodes
                cursor.execute("""
                    SELECT id, salience FROM episodes
                    WHERE topic = %s
                      AND created_at BETWEEN %s AND %s
                      AND deleted_at IS NULL
                """, (topic, window_start, window_end))
                episodes = cursor.fetchall()

                if not episodes:
                    cursor.close()
                    return 0

                # Record pre-boost salience
                boosted_episodes = []
                for ep_id, salience in episodes:
                    boosted_episodes.append({
                        "episode_id": ep_id,
                        "pre_boost_salience": float(salience) if salience else 0,
                    })

                # Boost salience (capped at 10)
                ep_ids = [ep[0] for ep in episodes]
                cursor.execute("""
                    UPDATE episodes
                    SET salience = LEAST(10, salience + 1.0)
                    WHERE id = ANY(%s)
                """, (ep_ids,))
                boosted_count = cursor.rowcount

                # Store boost records in moment metadata
                cursor.execute("""
                    UPDATE moments
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{boosted_episodes}',
                        %s::jsonb
                    ),
                    updated_at = NOW()
                    WHERE id = %s
                """, (json.dumps(boosted_episodes), moment_id))

                cursor.close()

            logger.info(f"{LOG_PREFIX} Boosted salience for {boosted_count} episodes (moment={moment_id})")
            return boosted_count

        except Exception as e:
            logger.error(f"{LOG_PREFIX} boost_related_salience failed: {e}")
            return 0

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _check_duplicate(
        self,
        embedding: list,
        user_id: str,
        threshold: float = 0.15,
    ) -> Optional[Dict[str, Any]]:
        """Check for near-duplicate moments via cosine distance."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, title, message_text, exchange_id, topic,
                           thread_id, gists, summary, status, pinned_at, sealed_at,
                           last_enriched_at, metadata, created_at, updated_at,
                           embedding <=> %s::vector AS distance
                    FROM moments
                    WHERE user_id = %s
                      AND status != 'forgotten'
                      AND deleted_at IS NULL
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """, (embedding, user_id, embedding))
                row = cursor.fetchone()
                cursor.close()

            if row and row[16] < threshold:
                return self._row_to_dict(row[:16])
            return None

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _check_duplicate failed: {e}")
            return None

    def _generate_title(self, message_text: str) -> str:
        """Auto-generate a short title from message text via LLM."""
        try:
            from services.llm_service import create_refreshable_llm_service
            llm = create_refreshable_llm_service("moment-enrichment")
            response = llm.send_message(
                "Generate a short title (3-6 words) for this pinned message. "
                "Be specific and direct. No quotes, no punctuation at the end. "
                "Example: 'Air fryer lemon chicken recipe'",
                message_text[:500],
            )
            title = response.text.strip().strip('"\'.')
            if title:
                return title[:100]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Title generation failed, using fallback: {e}")

        # Fallback: first 60 chars of message
        return message_text[:60].strip() + ("..." if len(message_text) > 60 else "")

    def _reverse_salience_boost(self, moment: Dict[str, Any]) -> None:
        """Reverse salience boosts applied when this moment was pinned."""
        metadata = moment.get("metadata") or {}
        boosted_episodes = metadata.get("boosted_episodes") or []

        if not boosted_episodes:
            return

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                for record in boosted_episodes:
                    ep_id = record.get("episode_id")
                    pre_boost = record.get("pre_boost_salience", 0)
                    if ep_id:
                        # Decrease by 1.0 but never below pre-boost value
                        cursor.execute("""
                            UPDATE episodes
                            SET salience = GREATEST(%s, salience - 1.0)
                            WHERE id = %s AND deleted_at IS NULL
                        """, (pre_boost, ep_id))
                cursor.close()

            logger.info(f"{LOG_PREFIX} Reversed salience for {len(boosted_episodes)} episodes")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _reverse_salience_boost failed: {e}")

    def _semantic_salience_boost(self, moment_id: str) -> int:
        """On seal: boost semantically similar episodes by +0.5."""
        moment = self.get_moment(moment_id)
        if not moment:
            return 0

        try:
            from services.embedding_service import get_embedding_service
            text = moment["message_text"]
            summary = moment.get("summary") or ""
            combined = f"{text} {summary}".strip()
            embedding = get_embedding_service().generate_embedding(combined)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE episodes
                    SET salience = LEAST(10, salience + 0.5)
                    WHERE deleted_at IS NULL
                      AND embedding IS NOT NULL
                      AND embedding <=> %s::vector < 0.5
                """, (embedding,))
                count = cursor.rowcount
                cursor.close()

            if count > 0:
                logger.info(f"{LOG_PREFIX} Semantic boost: {count} episodes (moment={moment_id})")
            return count

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _semantic_salience_boost failed: {e}")
            return 0

    def _is_duplicate_gist(
        self,
        new_gist: str,
        existing: List[str],
        threshold: float = 0.7,
    ) -> bool:
        """Jaccard similarity dedup for gist strings."""
        new_tokens = set(new_gist.lower().split())
        for existing_gist in existing:
            existing_tokens = set(existing_gist.lower().split())
            if not new_tokens or not existing_tokens:
                continue
            intersection = new_tokens & existing_tokens
            union = new_tokens | existing_tokens
            if len(intersection) / len(union) >= threshold:
                return True
        return False

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row tuple to a dict."""
        return {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "message_text": row[3],
            "exchange_id": row[4],
            "topic": row[5],
            "thread_id": row[6],
            "gists": row[7] if isinstance(row[7], list) else json.loads(row[7] or "[]"),
            "summary": row[8],
            "status": row[9],
            "pinned_at": row[10],
            "sealed_at": row[11],
            "last_enriched_at": row[12],
            "metadata": row[13] if isinstance(row[13], dict) else json.loads(row[13] or "{}"),
            "created_at": row[14],
            "updated_at": row[15],
        }
