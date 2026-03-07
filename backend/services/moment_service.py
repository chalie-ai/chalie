"""
Moment Service — pinned messages stored as documents with cognitive side effects.

Stores user-pinned Chalie responses as markdown documents in the document pipeline.
Each "moment" is a document with source_type='moment' and doc_category='Memory'.
Enrichment metadata (gists, pinned_at, topic, etc.) stored in extracted_metadata.

Side effects preserved:
  - Salience boosting of related episodes on pin
  - Salience reversal on forget (soft-delete)
  - Background enrichment via moment_enrichment_service
"""

import json
import logging
import struct
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MOMENTS]"


def _pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list of floats into a binary blob for sqlite-vec."""
    if embedding is None:
        return None
    return struct.pack(f'{len(embedding)}f', *embedding)


class MomentService:
    """Manages moment creation, search, enrichment, and salience — backed by documents."""

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
    ) -> Dict[str, Any]:
        """
        Pin a message as a moment (stored as a document).

        Returns dict with moment data. If a near-duplicate exists,
        returns {duplicate: True, existing_id: ...}.
        """
        from services.document_service import DocumentService
        from services.document_queue import enqueue_document_processing

        doc_service = DocumentService(self.db)

        # Duplicate check via embedding similarity on existing moment documents
        try:
            from services.embedding_service import get_embedding_service
            embedding = get_embedding_service().generate_embedding(message_text)

            dup = self._check_duplicate(doc_service, embedding)
            if dup:
                logger.info(f"{LOG_PREFIX} Near-duplicate detected (existing={dup['id']})")
                moment_data = self._doc_to_moment(dup)
                moment_data["duplicate"] = True
                moment_data["existing_id"] = dup["id"]
                return moment_data
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding/duplicate check failed: {e}")

        # Auto-generate title if not provided
        if not title:
            title = self._generate_title(message_text)

        # Create document via the document pipeline
        file_name = f"Memory — {title}.md"
        doc_id = doc_service.create_document_from_text(
            original_name=file_name,
            text_content=message_text,
            source_type='moment',
        )

        # Set classification immediately (don't wait for pipeline)
        doc_service.update_classification(
            doc_id,
            category='Memory',
            doc_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        )

        # Store moment-specific metadata in extracted_metadata
        moment_meta = {
            'moment_status': 'enriching',
            'moment_title': title,
            'moment_gists': [],
            'moment_pinned_at': datetime.now(timezone.utc).isoformat(),
            'moment_exchange_id': exchange_id,
            'moment_topic': topic,
            'moment_thread_id': thread_id,
            'boosted_episodes': [],
        }
        doc_service.update_extracted_metadata(
            doc_id, metadata=moment_meta, summary='',
        )

        # Enqueue for document processing (chunking, embedding, classification)
        enqueue_document_processing(doc_id)

        logger.info(f"{LOG_PREFIX} Created moment '{title}' as document {doc_id}")

        # Boost related episode salience
        try:
            self._boost_related_salience(doc_id, topic)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Salience boost failed (non-fatal): {e}")

        doc = doc_service.get_document(doc_id)
        return self._doc_to_moment(doc) if doc else {'id': doc_id, 'title': title}

    # ─────────────────────────────────────────────
    # Read
    # ─────────────────────────────────────────────

    def get_moment(self, moment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single moment by document ID."""
        from services.document_service import DocumentService
        doc_service = DocumentService(self.db)
        doc = doc_service.get_document(moment_id)
        if not doc or doc.get('source_type') != 'moment':
            return None
        if doc.get('deleted_at'):
            return None
        return self._doc_to_moment(doc)

    def get_all_moments(self) -> List[Dict[str, Any]]:
        """Get all active moment documents ordered by creation date DESC."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, extracted_metadata, summary,
                           doc_category, doc_project, doc_date, created_at,
                           clean_text
                    FROM documents
                    WHERE source_type = 'moment'
                      AND deleted_at IS NULL
                      AND json_extract(extracted_metadata, '$.moment_status') != 'forgotten'
                    ORDER BY created_at DESC
                """)
                rows = cursor.fetchall()
                cursor.close()

            moments = []
            for row in rows:
                meta = json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {})
                moments.append({
                    'id': row[0],
                    'title': meta.get('moment_title') or row[1],
                    'message_text': row[8] or '',
                    'summary': row[3] or '',
                    'gists': meta.get('moment_gists', []),
                    'status': meta.get('moment_status', 'sealed'),
                    'pinned_at': meta.get('moment_pinned_at') or row[7],
                    'created_at': row[7],
                })
            return moments

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_all_moments failed: {e}")
            return []

    def get_enriching_moments(self) -> List[Dict[str, Any]]:
        """Get all moment documents that need enrichment."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, extracted_metadata, summary,
                           clean_text, created_at
                    FROM documents
                    WHERE source_type = 'moment'
                      AND deleted_at IS NULL
                      AND json_extract(extracted_metadata, '$.moment_status') = 'enriching'
                    ORDER BY created_at ASC
                """)
                rows = cursor.fetchall()
                cursor.close()

            moments = []
            for row in rows:
                meta = json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {})
                moments.append({
                    'id': row[0],
                    'title': meta.get('moment_title') or row[1],
                    'message_text': row[4] or '',
                    'summary': row[3] or '',
                    'gists': meta.get('moment_gists', []),
                    'status': meta.get('moment_status', 'enriching'),
                    'pinned_at': meta.get('moment_pinned_at') or row[5],
                    'topic': meta.get('moment_topic'),
                    'exchange_id': meta.get('moment_exchange_id'),
                    'thread_id': meta.get('moment_thread_id'),
                    'created_at': row[5],
                    '_extracted_metadata': meta,
                })
            return moments

        except Exception as e:
            logger.error(f"{LOG_PREFIX} get_enriching_moments failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Forget / Delete
    # ─────────────────────────────────────────────

    def forget_moment(self, moment_id: str) -> bool:
        """Forget a moment — soft-deletes the document and reverses salience boosts."""
        from services.document_service import DocumentService
        doc_service = DocumentService(self.db)
        doc = doc_service.get_document(moment_id)
        if not doc or doc.get('source_type') != 'moment':
            return False

        try:
            # Reverse salience boosts
            meta = doc.get('extracted_metadata') or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            self._reverse_salience_boost(meta)

            # Soft-delete the document
            doc_service.soft_delete(moment_id)
            logger.info(f"{LOG_PREFIX} Forgot moment (id={moment_id})")
            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} forget_moment failed: {e}")
            return False

    # ─────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────

    def search_moments(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Semantic search over moment documents via document_chunks_vec."""
        try:
            from services.embedding_service import get_embedding_service
            query_embedding = get_embedding_service().generate_embedding(query)
            packed_query = _pack_embedding(query_embedding)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.id, d.original_name, d.extracted_metadata, d.summary,
                           d.clean_text, d.created_at, v.distance
                    FROM documents d
                    JOIN documents_vec v ON v.rowid = d.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                      AND d.source_type = 'moment'
                      AND d.deleted_at IS NULL
                    ORDER BY v.distance
                """, (packed_query, limit * 2))
                rows = cursor.fetchall()
                cursor.close()

            results = []
            for row in rows[:limit]:
                meta = json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {})
                moment = {
                    'id': row[0],
                    'title': meta.get('moment_title') or row[1],
                    'message_text': row[4] or '',
                    'summary': row[3] or '',
                    'gists': meta.get('moment_gists', []),
                    'status': meta.get('moment_status', 'sealed'),
                    'pinned_at': meta.get('moment_pinned_at') or row[5],
                    'created_at': row[5],
                    'distance': row[6],
                }
                results.append(moment)
            return results

        except Exception as e:
            logger.error(f"{LOG_PREFIX} search_moments failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Enrichment
    # ─────────────────────────────────────────────

    def enrich_moment(self, moment_id: str, new_gists: List[str]) -> bool:
        """Merge new gists into moment's extracted_metadata (Jaccard dedup)."""
        from services.document_service import DocumentService
        doc_service = DocumentService(self.db)
        doc = doc_service.get_document(moment_id)
        if not doc:
            return False

        meta = doc.get('extracted_metadata') or {}
        if isinstance(meta, str):
            meta = json.loads(meta)

        existing_gists = meta.get('moment_gists') or []
        merged = list(existing_gists)

        for gist in new_gists:
            if not self._is_duplicate_gist(gist, merged, threshold=0.7):
                merged.append(gist)

        if len(merged) == len(existing_gists):
            return False

        meta['moment_gists'] = merged
        doc_service.update_extracted_metadata(
            moment_id, metadata=meta, summary=doc.get('summary') or '',
        )

        logger.info(f"{LOG_PREFIX} Enriched moment {moment_id} ({len(existing_gists)} → {len(merged)} gists)")
        return True

    def generate_summary(self, moment_id: str) -> Optional[str]:
        """LLM-generated summary from pinned message + gists. Updates document."""
        from services.document_service import DocumentService
        doc_service = DocumentService(self.db)
        doc = doc_service.get_document(moment_id)
        if not doc:
            return None

        meta = doc.get('extracted_metadata') or {}
        if isinstance(meta, str):
            meta = json.loads(meta)

        message_text = doc.get('clean_text') or ''
        gists = meta.get('moment_gists') or []
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
            from services.background_llm_queue import create_background_llm_proxy
            llm = create_background_llm_proxy("moment-enrichment")
            response = llm.send_message(system_prompt, user_message)
            summary = response.text.strip()

            doc_service.update_extracted_metadata(
                moment_id, metadata=meta, summary=summary,
            )

            logger.info(f"{LOG_PREFIX} Generated summary for moment {moment_id}")
            return summary

        except Exception as e:
            logger.error(f"{LOG_PREFIX} generate_summary failed: {e}")
            return None

    def seal_moment(self, moment_id: str) -> bool:
        """Seal a moment — marks enrichment complete, updates the markdown file."""
        from services.document_service import DocumentService, DOCUMENTS_ROOT
        doc_service = DocumentService(self.db)
        doc = doc_service.get_document(moment_id)
        if not doc:
            return False

        meta = doc.get('extracted_metadata') or {}
        if isinstance(meta, str):
            meta = json.loads(meta)

        try:
            # Update status to sealed
            meta['moment_status'] = 'sealed'
            meta['moment_sealed_at'] = datetime.now(timezone.utc).isoformat()

            # Rewrite the markdown file with enriched content
            gists = meta.get('moment_gists') or []
            summary = doc.get('summary') or ''
            message_text = doc.get('clean_text') or ''

            enriched_content = message_text
            if summary:
                enriched_content += f"\n\n---\n**Summary:** {summary}"
            if gists:
                enriched_content += "\n\n**Context:**\n"
                enriched_content += "\n".join(f"- {g}" for g in gists)

            # Write updated file
            import os
            file_path = os.path.join(DOCUMENTS_ROOT, doc['file_path'])
            if os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(enriched_content)

            # Update metadata and re-embed
            import hashlib
            new_hash = hashlib.sha256(enriched_content.encode()).hexdigest()
            doc_service.update_extracted_metadata(
                moment_id, metadata=meta, summary=summary,
                clean_text=enriched_content,
            )

            # Update file hash
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents SET file_hash = ?, updated_at = datetime('now')
                    WHERE id = ?
                """, (new_hash, moment_id))
                cursor.close()

            # Re-embed the summary for vector search
            try:
                from services.embedding_service import get_embedding_service
                embedding = get_embedding_service().generate_embedding(
                    f"{message_text} {summary}".strip()
                )
                if embedding:
                    packed = _pack_embedding(embedding)
                    with self.db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT rowid FROM documents WHERE id = ?", (moment_id,))
                        row = cursor.fetchone()
                        if row:
                            cursor.execute(
                                "INSERT OR REPLACE INTO documents_vec(rowid, embedding) VALUES (?, ?)",
                                (row[0], packed)
                            )
                        cursor.close()
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Re-embed on seal failed (non-fatal): {e}")

            # Semantic salience boost on seal
            try:
                self._semantic_salience_boost(moment_id)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Seal salience boost failed (non-fatal): {e}")

            logger.info(f"{LOG_PREFIX} Sealed moment {moment_id}")
            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} seal_moment failed: {e}")
            return False

    # ─────────────────────────────────────────────
    # Salience
    # ─────────────────────────────────────────────

    def _boost_related_salience(self, doc_id: str, topic: Optional[str]) -> int:
        """Boost salience of episodes matching topic + time window."""
        if not topic:
            return 0

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=4)
        window_end = now + timedelta(hours=4)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, salience FROM episodes
                    WHERE topic = ?
                      AND created_at BETWEEN ? AND ?
                      AND deleted_at IS NULL
                """, (topic, window_start.isoformat(), window_end.isoformat()))
                episodes = cursor.fetchall()

                if not episodes:
                    cursor.close()
                    return 0

                boosted_episodes = []
                for ep in episodes:
                    boosted_episodes.append({
                        "episode_id": ep[0],
                        "pre_boost_salience": float(ep[1]) if ep[1] else 0,
                    })

                for ep in episodes:
                    cursor.execute("""
                        UPDATE episodes SET salience = MIN(10, salience + 1.0)
                        WHERE id = ?
                    """, (ep[0],))

                cursor.close()

            # Store boost records in document metadata
            from services.document_service import DocumentService
            doc_service = DocumentService(self.db)
            doc = doc_service.get_document(doc_id)
            if doc:
                meta = doc.get('extracted_metadata') or {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta['boosted_episodes'] = boosted_episodes
                doc_service.update_extracted_metadata(
                    doc_id, metadata=meta, summary=doc.get('summary') or '',
                )

            logger.info(f"{LOG_PREFIX} Boosted salience for {len(episodes)} episodes (doc={doc_id})")
            return len(episodes)

        except Exception as e:
            logger.error(f"{LOG_PREFIX} _boost_related_salience failed: {e}")
            return 0

    def _reverse_salience_boost(self, meta: dict) -> None:
        """Reverse salience boosts from extracted_metadata."""
        boosted_episodes = meta.get("boosted_episodes") or []
        if not boosted_episodes:
            return

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                for record in boosted_episodes:
                    ep_id = record.get("episode_id")
                    pre_boost = record.get("pre_boost_salience", 0)
                    if ep_id:
                        cursor.execute("""
                            UPDATE episodes SET salience = MAX(?, salience - 1.0)
                            WHERE id = ? AND deleted_at IS NULL
                        """, (pre_boost, ep_id))
                cursor.close()

            logger.info(f"{LOG_PREFIX} Reversed salience for {len(boosted_episodes)} episodes")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _reverse_salience_boost failed: {e}")

    def _semantic_salience_boost(self, doc_id: str) -> int:
        """On seal: boost semantically similar episodes by +0.5."""
        from services.document_service import DocumentService
        doc_service = DocumentService(self.db)
        doc = doc_service.get_document(doc_id)
        if not doc:
            return 0

        try:
            from services.embedding_service import get_embedding_service
            text = doc.get('clean_text') or ''
            summary = doc.get('summary') or ''
            combined = f"{text} {summary}".strip()
            embedding = get_embedding_service().generate_embedding(combined)
            packed = _pack_embedding(embedding)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.id FROM episodes e
                    JOIN episodes_vec v ON v.rowid = e.rowid
                    WHERE v.embedding MATCH ? AND k = 50
                      AND e.deleted_at IS NULL
                      AND v.distance < 0.5
                """, (packed,))
                matching_ids = [row[0] for row in cursor.fetchall()]

                count = 0
                for ep_id in matching_ids:
                    cursor.execute("""
                        UPDATE episodes
                        SET salience = MIN(10, salience + 0.5)
                        WHERE id = ? AND deleted_at IS NULL
                    """, (ep_id,))
                    count += cursor.rowcount

                cursor.close()

            if count > 0:
                logger.info(f"{LOG_PREFIX} Semantic boost: {count} episodes (doc={doc_id})")
            return count

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _semantic_salience_boost failed: {e}")
            return 0

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _check_duplicate(
        self,
        doc_service,
        embedding: list,
        threshold: float = 0.15,
    ) -> Optional[Dict[str, Any]]:
        """Check for near-duplicate moments via documents_vec cosine distance."""
        try:
            packed = _pack_embedding(embedding)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.id, v.distance
                    FROM documents d
                    JOIN documents_vec v ON v.rowid = d.rowid
                    WHERE v.embedding MATCH ? AND k = 1
                      AND d.source_type = 'moment'
                      AND d.deleted_at IS NULL
                    ORDER BY v.distance
                """, (packed,))
                row = cursor.fetchone()
                cursor.close()

            if row and row[1] < threshold:
                return doc_service.get_document(row[0])
            return None

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _check_duplicate failed: {e}")
            return None

    def _generate_title(self, message_text: str) -> str:
        """Auto-generate a short title from message text via LLM."""
        try:
            from services.background_llm_queue import create_background_llm_proxy
            llm = create_background_llm_proxy("moment-enrichment")
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

        return "Pinned Moment"

    def _doc_to_moment(self, doc: dict) -> dict:
        """Convert a document dict to the moment API shape."""
        meta = doc.get('extracted_metadata') or {}
        if isinstance(meta, str):
            meta = json.loads(meta)

        return {
            'id': doc['id'],
            'title': meta.get('moment_title') or doc.get('original_name', ''),
            'message_text': doc.get('clean_text') or '',
            'summary': doc.get('summary') or '',
            'gists': meta.get('moment_gists', []),
            'status': meta.get('moment_status', 'sealed'),
            'pinned_at': meta.get('moment_pinned_at') or doc.get('created_at'),
            'sealed_at': meta.get('moment_sealed_at'),
            'created_at': doc.get('created_at'),
            'updated_at': doc.get('updated_at'),
        }

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
