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
    5-entry overlap with the prior window. Runs EpisodeEncoderProcessor,
    computes novelty + salience in pure code, and stores resulting episodes.
    Never raises — any failure is logged only.
    """
    def _run():
        try:
            import json as _json
            from services.database_service import get_shared_db_service
            from services.episode_encoder_processor import EpisodeEncoderProcessor
            from services.episodic_service import (
                EpisodicService, _fetch_novelty_comparison_set, compute_novelty,
            )
            from services.salience_service import compute_salience
            from services.embedding_service import get_embedding_service
            from services.config_service import ConfigService

            db = get_shared_db_service()

            # ── 1. Fetch the window ──────────────────────────────────────────
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

            # ── 2. Format window string ──────────────────────────────────────
            window_str = _format_window_entries(entries)

            # ── 3. Fetch referenced episodes via tool_calls ──────────────────
            referenced_episodes = _fetch_referenced_episodes(entries, db)
            referenced_str = _format_episodes_for_prompt(referenced_episodes)

            # ── 4. Call EpisodeEncoderProcessor ─────────────────────────────
            response = EpisodeEncoderProcessor(window_str, referenced_str).send()
            snapshots = _safe_json_load(response)
            if not snapshots:
                return

            # ── 5. Resolve service handles ───────────────────────────────────
            try:
                episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            except Exception:
                episodic_config = {}

            episodic_svc = EpisodicService(db, episodic_config)
            emb_svc = get_embedding_service()

            valid_ids = {e['id'] for e in entries}

            # Hoist novelty comparison set ONCE — shared across all snapshots
            prior_embeddings = _fetch_novelty_comparison_set(channel)

            # ── 6. Store snapshots ───────────────────────────────────────────
            for ep in snapshots:
                try:
                    if _is_delete_only(ep):
                        delete_id = ep.get('delete_id')
                        if delete_id:
                            episodic_svc.soft_delete(delete_id)
                        continue

                    # Filter transcript_ids to valid window IDs
                    raw_ids = ep.get('transcript_ids') or []
                    ep['transcript_ids'] = [i for i in raw_ids if i in valid_ids]
                    if not ep['transcript_ids']:
                        continue

                    ep['transcript_id_start'] = min(ep['transcript_ids'])
                    ep['transcript_id_end'] = max(ep['transcript_ids'])
                    ep['channel'] = channel

                    gist = ep.get('gist', '') or ''
                    embedding = emb_svc.generate_embedding(gist) if gist else None

                    novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
                    ep['salience'] = compute_salience(
                        valence=float(ep.get('emotional_valence') or 0.0),
                        arousal=float(ep.get('emotional_arousal') or 0.0),
                        has_open_loop=bool(ep.get('has_open_loop', False)),
                        novelty=novelty,
                    )

                    # Pop transient fields — not persisted
                    has_open_loop = ep.pop('has_open_loop', False)  # noqa: F841 (consumed above)
                    update_id = ep.pop('update_id', None)
                    ep.pop('delete_id', None)  # defensive — should be None here

                    if update_id:
                        episodic_svc.update_episode(update_id, ep, embedding=embedding)
                    else:
                        episodic_svc.store_episode(ep, embedding=embedding)

                except Exception as ep_err:
                    logger.warning(f"{LOG_PREFIX} Episode store failed in trigger: {ep_err}")

            # After storing all snapshots, check for super-episode clusters.
            _maybe_trigger_super_episode(channel, db, episodic_svc, emb_svc)

        except Exception as e:
            logger.warning(
                f"{LOG_PREFIX} Episode extraction trigger failed "
                f"(channel={channel}, rowid={rowid}): {e}"
            )

    threading.Thread(target=_run, daemon=True).start()


# ── Episode extraction helpers ───────────────────────────────────────────────


def _format_window_entries(entries: list) -> str:
    """Format transcript entries as `[id] (timestamp) role: content` lines."""
    lines = []
    for entry in entries:
        entry_id = entry.get('id', '?')
        role = entry.get('role', 'unknown')
        content = entry.get('content', '')
        tool_name = entry.get('tool_name')
        created_at = entry.get('created_at', '')
        if tool_name:
            lines.append(f"[{entry_id}] ({created_at}) {role} [{tool_name}]: {content}")
        else:
            lines.append(f"[{entry_id}] ({created_at}) {role}: {content}")
    return "\n".join(lines)


def _fetch_referenced_episodes(entries: list, db) -> list:
    """Query tool_calls for memory skill invocations within the window.

    Parses each result for episode IDs in the format `[id:{uuid},...]`
    (the output format of _format_results in memory_skill.py). Fetches
    the matching episodes from the DB and returns them as dicts.

    tool_name='memory' covers both auto-seed (tool_name='memory') and
    LLM-invoked recall (also dispatched as tool_name='memory').
    """
    import re as _re
    import json as _json

    t_ids = [e['id'] for e in entries if e.get('id')]
    if not t_ids:
        return []

    try:
        placeholders = ','.join('?' * len(t_ids))
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT result FROM tool_calls
                WHERE transcript_id IN ({placeholders})
                  AND tool_name = 'memory'
                  AND result IS NOT NULL
                """,
                t_ids,
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} _fetch_referenced_episodes query failed: {exc}")
        return []

    # Parse episode IDs from memory result strings: [id:{uuid},relevance:...]
    episode_ids = set()
    _id_pattern = _re.compile(r'\[id:([^,\]]+)')
    for row in rows:
        result_text = row[0] or ''
        for match in _id_pattern.finditer(result_text):
            eid = match.group(1).strip()
            if eid:
                episode_ids.add(eid)

    if not episode_ids:
        return []

    try:
        from services.episodic_service import EpisodicService
        from services.config_service import ConfigService
        try:
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        except Exception:
            episodic_config = {}
        episodic_svc = EpisodicService(db, episodic_config)
        episodes = []
        for eid in episode_ids:
            ep = episodic_svc.get_episode_by_id(eid)
            if ep:
                episodes.append(ep)
        return episodes
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} _fetch_referenced_episodes fetch failed: {exc}")
        return []


def _format_episodes_for_prompt(episodes: list) -> str:
    """Format referenced episodes for the EpisodeEncoderProcessor user prompt."""
    if not episodes:
        return ''
    lines = []
    for ep in episodes:
        eid = ep.get('id', '')
        gist = ep.get('gist', '')
        created_at = ep.get('created_at', '')
        lines.append(f"id: {eid} | gist: {gist} | created: {created_at}")
    return "\n".join(lines)


def _safe_json_load(text: str) -> list:
    """Parse a JSON array from LLM response text. Returns [] on any failure."""
    import json as _json
    import re as _re

    if not text:
        return []
    text = text.strip()
    # Strip markdown code fences if present
    match = _re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, list):
            return parsed
        logger.warning(f"{LOG_PREFIX} EpisodeEncoder returned non-list JSON")
        return []
    except (_json.JSONDecodeError, ValueError):
        logger.warning(f"{LOG_PREFIX} EpisodeEncoder returned unparseable JSON")
        return []


def _is_delete_only(ep: dict) -> bool:
    """Return True if the snapshot is a delete-only directive (no other data)."""
    if not ep.get('delete_id'):
        return False
    # All other meaningful fields must be absent or null/empty
    meaningful = ('gist', 'transcript_ids', 'update_id')
    return not any(ep.get(f) for f in meaningful)


def _fetch_transcript_spans(sources: list[dict], db) -> str:
    """Fetch and format the raw transcript rows spanning all source episodes.

    Collects the union of transcript IDs from the source episodes, then
    fetches those rows and formats them in the same `[id] (ts) role: content`
    style used by the EpisodeEncoderProcessor prompt.

    Args:
        sources: List of episode dicts (from episodic_svc.get_episode_by_id).
        db:      Shared DatabaseService instance.

    Returns:
        Formatted multi-line string of transcript entries, oldest first.
        Returns empty string if no transcript IDs are found.
    """
    import json as _json

    # Collect union of transcript IDs across all sources.
    all_t_ids: set[int] = set()
    for ep in sources:
        raw_ids = ep.get('transcript_ids')
        if isinstance(raw_ids, str):
            try:
                ids = _json.loads(raw_ids)
            except Exception:
                ids = []
        elif isinstance(raw_ids, list):
            ids = raw_ids
        else:
            ids = []
        for tid in ids:
            if tid is not None:
                try:
                    all_t_ids.add(int(tid))
                except (TypeError, ValueError):
                    pass

    # Also cover the full start–end range (inclusive) so overlap rows are included.
    starts = [ep.get('transcript_id_start') for ep in sources if ep.get('transcript_id_start') is not None]
    ends = [ep.get('transcript_id_end') for ep in sources if ep.get('transcript_id_end') is not None]
    if starts and ends:
        span_min = min(starts)
        span_max = max(ends)
        all_t_ids.update(range(span_min, span_max + 1))

    if not all_t_ids:
        return ''

    try:
        placeholders = ','.join('?' * len(all_t_ids))
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, role, content, tool_name, created_at
                FROM transcript
                WHERE id IN ({placeholders})
                ORDER BY id ASC
                """,
                list(all_t_ids),
            )
            rows = cursor.fetchall()
            cursor.close()

        entries = [
            {
                'id': r[0],
                'role': r[1],
                'content': r[2],
                'tool_name': r[3],
                'created_at': r[4],
            }
            for r in rows
        ]
        return _format_window_entries(entries)

    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} _fetch_transcript_spans failed: {exc}")
        return ''


def _maybe_trigger_super_episode(channel: str, db, episodic_svc, emb_svc) -> None:
    """Check for tight semantic clusters in the apex pool and create super-episodes.

    Called at the end of each successful episode extraction run.  For each
    cluster of 3+ apex episodes with all-pairwise cosine >= SUPER_EPISODE_THRESHOLD:

    1. Fetches raw transcript spans for the cluster.
    2. Calls SuperEpisodeEncoderProcessor to synthesise a consolidated gist.
    3. Embeds the gist, computes novelty + salience.
    4. Stores the super-episode via EpisodicService.
    5. Sets consolidated_into back-pointers on each source episode.

    Any per-cluster failure is logged and skipped — the remaining clusters still run.
    """
    from services.episodic_service import find_super_candidates, _fetch_novelty_comparison_set, compute_novelty
    from services.episodic_constants import SUPER_EPISODE_MIN_CLUSTER
    from services.salience_service import compute_salience
    from services.super_episode_encoder_processor import SuperEpisodeEncoderProcessor

    try:
        clusters = find_super_candidates(channel)
        if not clusters:
            return
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} find_super_candidates failed: {exc}")
        return

    # Hoisted out of the loop: the comparison set is channel-scoped and does
    # not need to be recomputed for every cluster in a single extraction run.
    # Minor staleness (cluster N doesn't see cluster N-1's super as prior art)
    # is acceptable — novelty is a soft bias on salience, not a correctness gate.
    try:
        prior_embeddings = _fetch_novelty_comparison_set(channel)
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} _fetch_novelty_comparison_set failed: {exc}")
        prior_embeddings = []

    for cluster_ids in clusters:
        try:
            sources = []
            for eid in cluster_ids:
                ep = episodic_svc.get_episode_by_id(eid)
                if ep:
                    sources.append(ep)

            if len(sources) < SUPER_EPISODE_MIN_CLUSTER:
                continue

            transcript_spans = _fetch_transcript_spans(sources, db)

            response = SuperEpisodeEncoderProcessor(sources, transcript_spans).send()
            if not response:
                logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response for cluster {cluster_ids}")
                continue

            super_ep = _safe_json_load_object(response)
            if not super_ep or not super_ep.get('gist'):
                logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty gist for cluster {cluster_ids}")
                continue

            super_ep['channel'] = channel

            # Union of all source transcript_ids.
            import json as _json
            all_t_ids: list[int] = []
            for ep in sources:
                raw_ids = ep.get('transcript_ids')
                if isinstance(raw_ids, str):
                    try:
                        ids = _json.loads(raw_ids)
                    except Exception:
                        ids = []
                elif isinstance(raw_ids, list):
                    ids = raw_ids
                else:
                    ids = []
                for tid in ids:
                    if tid is not None:
                        try:
                            all_t_ids.append(int(tid))
                        except (TypeError, ValueError):
                            pass

            unique_t_ids = sorted(set(all_t_ids))
            super_ep['transcript_ids'] = unique_t_ids
            super_ep['transcript_id_start'] = min(unique_t_ids) if unique_t_ids else None
            super_ep['transcript_id_end'] = max(unique_t_ids) if unique_t_ids else None
            super_ep['consolidated_from'] = [ep['id'] for ep in sources]

            gist = super_ep.get('gist', '')
            embedding = emb_svc.generate_embedding(gist) if gist else None

            novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
            super_ep['salience'] = compute_salience(
                valence=float(super_ep.get('emotional_valence') or 0.0),
                arousal=float(super_ep.get('emotional_arousal') or 0.0),
                has_open_loop=bool(super_ep.get('has_open_loop', False)),
                novelty=novelty,
            )

            # Pop transient field not persisted.
            super_ep.pop('has_open_loop', None)

            new_id = episodic_svc.store_episode(super_ep, embedding=embedding)

            for src_id in cluster_ids:
                episodic_svc.set_consolidated_into(src_id, new_id)

            logger.info(
                f"{LOG_PREFIX} Super-episode {new_id} created from cluster {cluster_ids}"
            )

        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} Super-episode creation failed for cluster {cluster_ids}: {exc}"
            )


def _safe_json_load_object(text: str) -> dict:
    """Parse a JSON object from LLM response text. Returns {} on any failure."""
    import json as _json
    import re as _re

    if not text:
        return {}
    text = text.strip()
    match = _re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned non-dict JSON")
        return {}
    except (_json.JSONDecodeError, ValueError):
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable JSON")
        return {}


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
