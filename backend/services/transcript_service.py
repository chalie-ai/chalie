"""
Transcript Service — persistent conversation record.

Stores every exchange turn (user, assistant, tool, internal) in SQLite
with optional vector embeddings for semantic search.

Key operations:
- append(): Write a turn to the transcript
- search(): Semantic search via transcript_vec (supports cross-topic)
- get_recent(): Retrieve the most recent N entries for a topic
- prune_old(): Delete entries older than TTL (90 days default)
"""

import logging
import threading
from typing import List, Dict, Optional

from services.embedding_utils import pack_embedding

from services.llm_service import estimate_tokens

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TRANSCRIPT]"

# Embedding threshold — entries with fewer estimated tokens are not embedded
_EMBED_TOKEN_THRESHOLD = 50

# Default TTL for pruning (90 days in seconds)
_PRUNE_TTL_DAYS = 90

# Per-channel insert counters for rolling episode extraction.
# First extraction fires when a channel has accumulated FIRST_EXTRACTION
# inserts (25). Subsequent extractions fire every EXTRACTION_INTERVAL (20).
# Each window covers 25 entries (20 new + 5 overlap with the prior window)
# so consecutive windows share 5 entries of context.
_channel_insert_counts: dict[str, int] = {}
_channel_first_fired: dict[str, bool] = {}
_channel_insert_lock = threading.Lock()
_FIRST_EXTRACTION = 25
_EXTRACTION_INTERVAL = 20
_EXTRACTION_WINDOW = 25
_EXTRACTION_OVERLAP = 5


def append(
    channel: str,
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
    if not channel:
        return None
    if not content and role != 'assistant':
        return None

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcript (channel, role, content, tool_call_id, tool_name, internal)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (channel, role, content, tool_call_id, tool_name, 1 if internal else 0),
            )
            rowid = cursor.lastrowid
            cursor.close()

        # Generate embedding for substantive entries
        if estimate_tokens(content) >= _EMBED_TOKEN_THRESHOLD:
            _embed_entry(rowid, content)

        _maybe_trigger_extraction(channel, rowid)

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
                topic = entry.get('channel') or entry.get('topic', '')
                content = entry.get('content', '')
                if not topic or not content:
                    continue
                cursor.execute(
                    """
                    INSERT INTO transcript (channel, role, content, tool_call_id, tool_name, internal)
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
    channel: Optional[str],
    query: str,
    limit: int = 5,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict]:
    """Semantic search over transcript entries.

    Uses embedding similarity via transcript_vec.

    Args:
        channel: Filter to a specific channel, or None for cross-channel (global) search.
        query: Search text.
        limit: Max results (1-20).
        date_from: ISO datetime lower bound (inclusive). Optional.
        date_to: ISO datetime upper bound (inclusive). Optional.

    Returns list of dicts with: id, role, content, tool_name, created_at, channel, similarity.
    """
    limit = min(max(limit, 1), 20)

    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        query_embedding = emb_service.generate_embedding(query)
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Embedding failed, falling back to keyword: {e}")
        return _keyword_search(channel, query, limit, date_from, date_to)

    blob = pack_embedding(query_embedding)

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        # Build WHERE clause dynamically
        conditions = ["v.embedding MATCH ?", "k = ?"]
        params: list = [blob, limit + 10]

        if channel:
            conditions.append("tt.channel = ?")
            params.append(channel)

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
                       v.distance, tt.channel
                FROM transcript_vec v
                JOIN transcript tt ON tt.rowid = v.rowid
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
                'channel': row[6],
            })
        return results

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Vector search failed: {e}")
        return _keyword_search(channel, query, limit, date_from, date_to)


def get_recent(channel: str, limit: int = 20, since_id: int = None, _context=None) -> List[Dict]:
    """Get the most recent transcript entries for a channel.

    Args:
        channel: Channel key to retrieve entries for.
        limit: Maximum entries to return (default 20).
        since_id: If provided, only return entries with id > since_id.

    Returns list of dicts with: id, role, content, tool_call_id, tool_name, internal, created_at.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            if since_id is not None:
                cursor.execute(
                    """
                    SELECT id, role, content, tool_call_id, tool_name, internal, created_at
                    FROM transcript
                    WHERE channel = ? AND id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (channel, since_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, role, content, tool_call_id, tool_name, internal, created_at
                    FROM transcript
                    WHERE channel = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (channel, limit),
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
        if since_id is None:
            results.reverse()

        return results

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} get_recent failed: {e}")
        return []


def get_latest_id(channel: str) -> Optional[int]:
    """Get the highest transcript entry ID for a channel (compaction watermark)."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(id) FROM transcript WHERE channel = ?",
                (channel,),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row and row[0] else None

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} get_latest_id failed: {e}")
        return None


def cleanup_unlinked_entries(channel: str = None) -> int:
    """Delete transcript entries not linked to any episode and below compaction watermark.

    Returns number of entries deleted.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()

            if channel:
                cursor.execute(
                    "SELECT channel, compacted_up_to_id FROM compactions WHERE channel = ?",
                    (channel,),
                )
            else:
                cursor.execute("SELECT channel, compacted_up_to_id FROM compactions")

            watermarks = cursor.fetchall()

            if not watermarks:
                cursor.close()
                return 0

            total_deleted = 0

            for row in watermarks:
                t, watermark = row[0], row[1]
                if not watermark:
                    continue

                # Collect transcript IDs referenced by any episode for this topic
                cursor.execute(
                    """
                    SELECT transcript_ids FROM episodes
                    WHERE channel = ? AND deleted_at IS NULL
                      AND transcript_ids IS NOT NULL AND transcript_ids != '[]'
                    """,
                    (t,),
                )
                referenced_ids = set()
                import json as _json
                for ep_row in cursor.fetchall():
                    try:
                        ids = _json.loads(ep_row[0])
                        if isinstance(ids, list):
                            referenced_ids.update(int(i) for i in ids if i is not None)
                    except Exception:
                        pass

                # Find transcript rowids below watermark that are not referenced
                cursor.execute(
                    """
                    SELECT id, rowid FROM transcript
                    WHERE channel = ? AND id < ?
                    """,
                    (t, watermark),
                )
                candidate_rows = cursor.fetchall()

                to_delete_ids = []
                to_delete_rowids = []
                for entry_id, entry_rowid in candidate_rows:
                    if entry_id not in referenced_ids:
                        to_delete_ids.append(entry_id)
                        to_delete_rowids.append(entry_rowid)

                if not to_delete_ids:
                    continue

                placeholders = ','.join('?' * len(to_delete_rowids))
                cursor.execute(
                    f"DELETE FROM transcript_vec WHERE rowid IN ({placeholders})",
                    to_delete_rowids,
                )
                id_placeholders = ','.join('?' * len(to_delete_ids))
                cursor.execute(
                    f"DELETE FROM transcript WHERE id IN ({id_placeholders})",
                    to_delete_ids,
                )
                total_deleted += len(to_delete_ids)

            cursor.close()

        if total_deleted > 0:
            logger.info(f"{LOG_PREFIX} Cleaned up {total_deleted} unlinked transcript entries")
        return total_deleted

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} cleanup_unlinked_entries failed: {e}")
        return 0


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
                SELECT rowid FROM transcript
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
                f"DELETE FROM transcript_vec WHERE rowid IN ({placeholders})",
                old_rowids,
            )

            # Delete from main table
            cursor.execute(
                f"DELETE FROM transcript WHERE rowid IN ({placeholders})",
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


def _maybe_trigger_extraction(channel: str, rowid: int) -> None:
    with _channel_insert_lock:
        count = _channel_insert_counts.get(channel, 0) + 1
        _channel_insert_counts[channel] = count
        first_fired = _channel_first_fired.get(channel, False)
        # First fire at FIRST_EXTRACTION, subsequent every EXTRACTION_INTERVAL
        threshold = _EXTRACTION_INTERVAL if first_fired else _FIRST_EXTRACTION
        if count >= threshold:
            _channel_insert_counts[channel] = 0
            _channel_first_fired[channel] = True
            trigger = True
        else:
            trigger = False
    if trigger:
        _trigger_episode_extraction(channel, rowid)


def _trigger_episode_extraction(channel: str, rowid: int) -> None:
    """Fire-and-forget episode extraction for the window ending at rowid.

    Queries the last _EXTRACTION_WINDOW (25) transcript entries up to and
    including rowid for the given channel — 20 new entries plus a
    5-entry overlap with the prior window. Runs episode extraction, computes
    embeddings, calculates salience, stores episodes + traits. Never raises —
    any failure is logged only.
    """
    def _run():
        try:
            from services.database_service import get_shared_db_service
            from services.episode_extractor_service import EpisodeExtractorService
            from services.episodic_service import EpisodicService
            from services.salience_service import SalienceService
            from services.embedding_service import get_embedding_service
            from services.data_graph_service import get_data_graph_service
            from services.config_service import ConfigService

            db = get_shared_db_service()

            # Fetch the window: up to 35 entries for this channel up to rowid
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, role, content, tool_name, created_at
                    FROM transcript
                    WHERE channel = ? AND id <= ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (channel, rowid, _EXTRACTION_WINDOW),
                )
                rows = cursor.fetchall()
                cursor.close()

            if not rows:
                return

            entries = [
                {
                    'id': r[0],
                    'role': r[1],
                    'content': r[2],
                    'tool_name': r[3],
                    'created_at': r[4],
                    'channel': channel,
                }
                for r in reversed(rows)
            ]

            extractor = EpisodeExtractorService()
            episodes = extractor.extract(entries, channel)

            if not episodes:
                return

            try:
                episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            except Exception:
                episodic_config = {}

            episodic_svc = EpisodicService(db, episodic_config)
            salience_svc = SalienceService()
            emb_svc = get_embedding_service()
            dgs = get_data_graph_service()

            for ep in episodes:
                try:
                    factors = ep.get('salience_factors', {}) or {}
                    # Map prompt-side keys to SalienceService-side keys
                    mapped_factors = {
                        'novelty': factors.get('novelty', 0),
                        'emotional': factors.get('emotional_weight', factors.get('emotional', 0)),
                        'commitment': factors.get('goal_relevance', factors.get('commitment', 0)),
                        'unresolved': factors.get('open_loop_created', factors.get('unresolved', False)),
                    }
                    salience_f = salience_svc.calculate_salience(mapped_factors)  # 0.1..1.0
                    # DB requires INTEGER 1..10
                    ep['salience'] = max(1, min(10, int(round(salience_f * 10))))
                    ep['channel'] = channel

                    gist = ep.get('gist', '')
                    if gist:
                        ep['embedding'] = emb_svc.generate_embedding(gist)

                    episodic_svc.store_episode(ep)

                    for trait in ep.get('traits', []):
                        if not isinstance(trait, dict):
                            continue
                        key = trait.get('key', '')
                        value = trait.get('value')
                        if key and value:
                            dgs.store(kind='user_specific', key=key, value=value,
                                      source='episode_extraction')

                except Exception as ep_err:
                    logger.warning(f"{LOG_PREFIX} Episode store failed in trigger: {ep_err}")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Episode extraction trigger failed (channel={channel}, rowid={rowid}): {e}")

    threading.Thread(target=_run, daemon=True).start()


def _embed_entry(rowid: int, content: str) -> None:
    """Generate and store embedding for a transcript entry."""
    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        embedding = emb_service.generate_embedding(content)
        blob = pack_embedding(embedding)

        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcript_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
            cursor.close()

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Embedding failed for rowid {rowid}: {e}")


def write_input_row(channel: str, role: str, content: str) -> int:
    """Write the input transcript row. Returns the row ID.

    Fires an embedding hook in a daemon thread if the content is long enough.
    """
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            (channel, role, content),
        )
        row_id = cursor.lastrowid
        cursor.close()

    def _embed():
        try:
            if estimate_tokens(content) >= _EMBED_TOKEN_THRESHOLD:
                _embed_entry(row_id, content)
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} embed-input hook crashed: {exc}")

    threading.Thread(target=_embed, daemon=True).start()

    _maybe_trigger_extraction(channel, row_id)

    return row_id


def write_assistant_row(channel: str, content: str) -> int:
    """Write the assistant transcript row. Returns the row ID.

    Fires an embedding hook and rolling episode extraction trigger.
    """
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            (channel, 'assistant', content),
        )
        row_id = cursor.lastrowid
        cursor.close()

    def _embed():
        try:
            if estimate_tokens(content) >= _EMBED_TOKEN_THRESHOLD:
                _embed_entry(row_id, content)
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} embed-assistant hook crashed: {exc}")

    threading.Thread(target=_embed, daemon=True).start()

    _maybe_trigger_extraction(channel, row_id)

    return row_id


def _keyword_search(
    channel: Optional[str],
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

        if channel:
            conditions.append("channel = ?")
            params.append(channel)

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
                SELECT id, role, content, tool_name, created_at, channel
                FROM transcript
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
                'channel': r[5],
            }
            for r in rows
        ]

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword search failed: {e}")
        return []
