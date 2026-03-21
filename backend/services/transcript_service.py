"""
Transcript Service — persistent conversation record.

Stores every exchange turn (user, assistant, tool, internal) in SQLite
with optional vector embeddings for semantic search.

Key operations:
- append(): Write a turn to the transcript
- search(): Semantic search via topic_transcript_vec (supports cross-topic)
- get_recent(): Retrieve the most recent N entries for a topic
- prune_old(): Delete entries older than TTL (90 days default)
"""

import struct
import logging
from typing import List, Dict, Optional

from services.llm_service import estimate_tokens

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TRANSCRIPT]"

# Embedding threshold — entries with fewer estimated tokens are not embedded
_EMBED_TOKEN_THRESHOLD = 50

# Default TTL for pruning (90 days in seconds)
_PRUNE_TTL_DAYS = 90


def append(
    topic: str,
    role: str,
    content: str,
    tool_call_id: str = None,
    tool_name: str = None,
    internal: bool = False,
) -> Optional[int]:
    """Append a turn to the topic transcript.

    Generates an embedding for substantive entries (>50 estimated tokens)
    and inserts it into the companion vec table.

    Returns the rowid of the inserted entry, or None on failure.
    """
    if not topic or not content:
        return None

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO topic_transcript (topic, role, content, tool_call_id, tool_name, internal)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (topic, role, content, tool_call_id, tool_name, 1 if internal else 0),
            )
            rowid = cursor.lastrowid
            cursor.close()

        # Generate embedding for substantive entries
        if estimate_tokens(content) >= _EMBED_TOKEN_THRESHOLD:
            _embed_entry(rowid, content)

        return rowid

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to append: {e}")
        return None


def append_batch(entries: List[Dict]) -> int:
    """Append multiple turns in a single transaction.

    Each entry dict should have: topic, role, content, and optionally
    tool_call_id, tool_name, internal.

    Returns the number of entries successfully inserted.
    """
    if not entries:
        return 0

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        inserted = 0
        rowids_to_embed = []

        with db.connection() as conn:
            cursor = conn.cursor()
            for entry in entries:
                topic = entry.get('topic', '')
                content = entry.get('content', '')
                if not topic or not content:
                    continue
                cursor.execute(
                    """
                    INSERT INTO topic_transcript (topic, role, content, tool_call_id, tool_name, internal)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic,
                        entry.get('role', 'user'),
                        content,
                        entry.get('tool_call_id'),
                        entry.get('tool_name'),
                        1 if entry.get('internal') else 0,
                    ),
                )
                rowid = cursor.lastrowid
                inserted += 1
                if estimate_tokens(content) >= _EMBED_TOKEN_THRESHOLD:
                    rowids_to_embed.append((rowid, content))
            cursor.close()

        # Generate embeddings outside the transaction
        for rowid, content in rowids_to_embed:
            _embed_entry(rowid, content)

        return inserted

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Batch append failed: {e}")
        return 0


def search(
    topic: Optional[str],
    query: str,
    limit: int = 5,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict]:
    """Semantic search over transcript entries.

    Uses embedding similarity via topic_transcript_vec.

    Args:
        topic: Filter to a specific topic, or None for cross-topic (global) search.
        query: Search text.
        limit: Max results (1-20).
        date_from: ISO datetime lower bound (inclusive). Optional.
        date_to: ISO datetime upper bound (inclusive). Optional.

    Returns list of dicts with: id, role, content, tool_name, created_at, topic, similarity.
    """
    limit = min(max(limit, 1), 20)

    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        query_embedding = emb_service.generate_embedding(query)
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Embedding failed, falling back to keyword: {e}")
        return _keyword_search(topic, query, limit, date_from, date_to)

    blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        # Build WHERE clause dynamically
        conditions = ["v.embedding MATCH ?", "k = ?"]
        params: list = [blob, limit + 10]

        if topic:
            conditions.append("tt.topic = ?")
            params.append(topic)

        if date_from:
            conditions.append("tt.created_at >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("tt.created_at <= ?")
            params.append(date_to)

        where = " AND ".join(conditions)

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT tt.id, tt.role, tt.content, tt.tool_name, tt.created_at,
                       v.distance, tt.topic
                FROM topic_transcript_vec v
                JOIN topic_transcript tt ON tt.rowid = v.rowid
                WHERE {where}
                ORDER BY v.distance
                """,
                params,
            )
            rows = cursor.fetchall()
            cursor.close()

        results = []
        for row in rows[:limit]:
            distance = row[5]
            similarity = max(0.0, 1.0 - distance / 2.0)
            results.append({
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'tool_name': row[3],
                'created_at': row[4],
                'similarity': similarity,
                'topic': row[6],
            })
        return results

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Vector search failed: {e}")
        return _keyword_search(topic, query, limit, date_from, date_to)


def get_recent(topic: str, limit: int = 20, since_id: int = None) -> List[Dict]:
    """Get the most recent transcript entries for a topic.

    Args:
        topic: Topic to retrieve entries for.
        limit: Maximum entries to return (default 20).
        since_id: If provided, only return entries with id > since_id.

    Returns list of dicts with: id, role, content, tool_call_id, tool_name, internal, created_at.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            if since_id:
                cursor.execute(
                    """
                    SELECT id, role, content, tool_call_id, tool_name, internal, created_at
                    FROM topic_transcript
                    WHERE topic = ? AND id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (topic, since_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, role, content, tool_call_id, tool_name, internal, created_at
                    FROM topic_transcript
                    WHERE topic = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (topic, limit),
                )
            rows = cursor.fetchall()
            cursor.close()

        results = [
            {
                'id': r[0],
                'role': r[1],
                'content': r[2],
                'tool_call_id': r[3],
                'tool_name': r[4],
                'internal': bool(r[5]),
                'created_at': r[6],
            }
            for r in rows
        ]

        # Reverse if fetched DESC (no since_id) so oldest is first
        if not since_id:
            results.reverse()

        return results

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} get_recent failed: {e}")
        return []


def get_latest_id(topic: str) -> Optional[int]:
    """Get the highest transcript entry ID for a topic (compaction watermark)."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(id) FROM topic_transcript WHERE topic = ?",
                (topic,),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row and row[0] else None

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} get_latest_id failed: {e}")
        return None


def prune_old(ttl_days: int = _PRUNE_TTL_DAYS) -> int:
    """Delete transcript entries older than ttl_days.

    Also removes corresponding vec table entries.
    Returns the number of entries deleted.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()

            # Find entries to delete
            cursor.execute(
                """
                SELECT rowid FROM topic_transcript
                WHERE created_at < datetime('now', ?)
                """,
                (f'-{ttl_days} days',),
            )
            old_rowids = [r[0] for r in cursor.fetchall()]

            if not old_rowids:
                cursor.close()
                return 0

            # Delete from vec table first (FK-safe)
            placeholders = ','.join('?' * len(old_rowids))
            cursor.execute(
                f"DELETE FROM topic_transcript_vec WHERE rowid IN ({placeholders})",
                old_rowids,
            )

            # Delete from main table
            cursor.execute(
                f"DELETE FROM topic_transcript WHERE rowid IN ({placeholders})",
                old_rowids,
            )

            deleted = len(old_rowids)
            cursor.close()

        logger.info(f"{LOG_PREFIX} Pruned {deleted} entries older than {ttl_days} days")
        return deleted

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Pruning failed: {e}")
        return 0


# ── Internal helpers ─────────────────────────────────────────────────


def _embed_entry(rowid: int, content: str) -> None:
    """Generate and store embedding for a transcript entry."""
    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        embedding = emb_service.generate_embedding(content)
        blob = struct.pack(f'{len(embedding)}f', *embedding)

        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO topic_transcript_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
            cursor.close()

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Embedding failed for rowid {rowid}: {e}")


def _keyword_search(
    topic: Optional[str],
    query: str,
    limit: int,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict]:
    """Keyword fallback when embedding search is unavailable."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        conditions = ["content LIKE ?"]
        params: list = [f'%{query}%']

        if topic:
            conditions.append("topic = ?")
            params.append(topic)

        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = " AND ".join(conditions)
        params.append(limit)

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, role, content, tool_name, created_at, topic
                FROM topic_transcript
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
            rows = cursor.fetchall()
            cursor.close()

        return [
            {
                'id': r[0],
                'role': r[1],
                'content': r[2],
                'tool_name': r[3],
                'created_at': r[4],
                'similarity': 0.5,
                'topic': r[5],
            }
            for r in rows
        ]

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword search failed: {e}")
        return []
