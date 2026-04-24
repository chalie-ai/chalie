"""
Transcript Service — persistent conversation record.

Stores every exchange turn (user, assistant, tool, internal) in SQLite.

Key operations:
- append(): Write a turn to the transcript
- get_recent(): Retrieve the most recent N entries for a topic
- prune_old(): Delete entries older than TTL (90 days default)
"""

import logging
import threading
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TRANSCRIPT]"

# Default TTL for pruning (90 days in seconds)
_PRUNE_TTL_DAYS = 90

# Rolling episode extraction — DB-state-driven.
#
# On every transcript append, count the transcripts in the channel whose id
# is greater than the channel's latest episode.transcript_id_end. When the
# tail reaches _EXTRACTION_THRESHOLD (20), fire extraction. The extractor
# pulls the last _EXTRACTION_WINDOW (25) rows — 20 new + 5 overlap with
# the prior window — producing one episode per ~20 new transcript turns.
#
# No process-local counter, no boot catch-up: trigger state lives entirely
# in the DB (transcript.id vs episodes.transcript_id_end) so restarts never
# desync from accumulated history.
_EXTRACTION_THRESHOLD = 20
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

        _maybe_trigger_extraction(channel, rowid)

        return rowid

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to append: {e}")
        return None



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

                # Find transcript IDs below watermark that are not referenced
                cursor.execute(
                    """
                    SELECT id FROM transcript
                    WHERE channel = ? AND id < ?
                    """,
                    (t, watermark),
                )
                candidate_rows = cursor.fetchall()

                to_delete_ids = []
                for (entry_id,) in candidate_rows:
                    if entry_id not in referenced_ids:
                        to_delete_ids.append(entry_id)

                if not to_delete_ids:
                    continue

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
    """Delete transcript entries older than ttl_days. Returns the number deleted."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()

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

            placeholders = ','.join('?' * len(old_rowids))
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
    """Fire episode extraction when the channel has accumulated
    _EXTRACTION_THRESHOLD untriggered transcripts.

    DB-state-driven: counts transcripts with id > MAX(episodes.transcript_id_end)
    for the channel. When count >= threshold, fire. No process-local state,
    so restarts and container rebuilds cannot desync the trigger from the
    actual tail of accumulated history.

    Never raises — failures logged and silently ignored.
    """
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM transcript
                WHERE channel = ?
                  AND id > COALESCE(
                      (SELECT MAX(transcript_id_end)
                       FROM episodes
                       WHERE channel = ? AND deleted_at IS NULL
                         AND transcript_id_end IS NOT NULL),
                      0
                  )
                """,
                (channel, channel),
            ).fetchone()
        untriggered = row[0] if row else 0
        if untriggered >= _EXTRACTION_THRESHOLD:
            _trigger_episode_extraction(channel, rowid)
    except Exception as e:
        logger.warning(
            f"{LOG_PREFIX} _maybe_trigger_extraction failed "
            f"(channel={channel}, rowid={rowid}): {e}"
        )


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
                    ep.pop('has_open_loop', None)
                    update_id = ep.pop('update_id', None)
                    ep.pop('delete_id', None)  # defensive — should be None here

                    if update_id:
                        episodic_svc.update_episode(update_id, ep, embedding=embedding)
                    else:
                        episodic_svc.store_episode(ep, embedding=embedding)

                    # Update the novelty comparison set so subsequent snapshots
                    # in this same extraction run are compared against already-stored
                    # episodes, not just the stale pre-run set.
                    if embedding is not None:
                        from services.embedding_utils import pack_embedding as _pack
                        blob = _pack(embedding)
                        if blob is not None:
                            prior_embeddings.append(blob)

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


def _strip_code_fence(text: str) -> str:
    """Remove a single markdown code fence wrapper (```[lang]...```) from text."""
    open_end = text.find("```")
    if open_end == -1:
        return text
    # Skip the opening fence line (```json or ```)
    newline = text.find("\n", open_end)
    if newline == -1:
        return text
    close_start = text.rfind("```", newline)
    if close_start <= newline:
        return text
    return text[newline + 1 : close_start].strip()


def _safe_json_load(text: str) -> list:
    """Parse a JSON array from LLM response text. Returns [] on any failure."""
    import json as _json

    if not text:
        return []
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = _strip_code_fence(text)
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, list):
            return parsed
        logger.warning(f"{LOG_PREFIX} EpisodeEncoder returned non-list JSON")
        return []
    except ValueError:
        logger.warning(f"{LOG_PREFIX} EpisodeEncoder returned unparseable JSON")
        return []


def _is_delete_only(ep: dict) -> bool:
    """Return True if the snapshot is a delete-only directive (no other data)."""
    if not ep.get('delete_id'):
        return False
    # All other meaningful fields must be absent or null/empty
    meaningful = ('gist', 'transcript_ids', 'update_id')
    return not any(ep.get(f) for f in meaningful)


def _parse_transcript_ids_field(raw) -> list:
    """Normalize an episode.transcript_ids field (JSON string or list) to a list."""
    import json as _json
    if isinstance(raw, str):
        try:
            return _json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    return []


def _collect_transcript_ids(episodes: list[dict]) -> set[int]:
    """Return the union of all transcript IDs referenced by *episodes*.

    Handles both JSON-string and list forms of the ``transcript_ids`` field,
    and also expands the ``transcript_id_start``–``transcript_id_end`` range so
    that overlap rows are always included.
    """
    ids: set[int] = set()
    for ep in episodes:
        for tid in _parse_transcript_ids_field(ep.get('transcript_ids')):
            if tid is None:
                continue
            try:
                ids.add(int(tid))
            except (TypeError, ValueError):
                pass

    starts = [ep['transcript_id_start'] for ep in episodes if ep.get('transcript_id_start') is not None]
    ends = [ep['transcript_id_end'] for ep in episodes if ep.get('transcript_id_end') is not None]
    if starts and ends:
        ids.update(range(min(starts), max(ends) + 1))

    return ids


def _fetch_transcript_spans(sources: list[dict], db) -> str:
    """Fetch and format the raw transcript rows spanning all source episodes.

    Collects the union of transcript IDs from the source episodes, then
    fetches those rows and formats them in the same ``[id] (ts) role: content``
    style used by the EpisodeEncoderProcessor prompt.

    Args:
        sources: List of episode dicts (from episodic_svc.get_episode_by_id).
        db:      Shared DatabaseService instance.

    Returns:
        Formatted multi-line string of transcript entries, oldest first.
        Returns empty string if no transcript IDs are found.
    """
    all_t_ids = _collect_transcript_ids(sources)
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
            {'id': r[0], 'role': r[1], 'content': r[2], 'tool_name': r[3], 'created_at': r[4]}
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
    2. Calls super_episode_encoder_cls to synthesise a consolidated gist.
    3. Embeds the gist, computes novelty + salience.
    4. Stores the super-episode via EpisodicService.
    5. Sets consolidated_into back-pointers on each source episode.

    Any per-cluster failure is logged and skipped — the remaining clusters still run.
    """
    from services.episodic_service import find_super_candidates, _fetch_novelty_comparison_set, compute_novelty
    from services.episodic_constants import SUPER_EPISODE_MIN_CLUSTER
    from services.salience_service import compute_salience
    from services.super_episode_encoder_processor import (
        SuperEpisodeEncoderProcessor as super_episode_encoder_cls,
    )

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
            _process_super_cluster(
                cluster_ids, channel, db, episodic_svc, emb_svc, prior_embeddings,
                super_episode_encoder_cls, compute_novelty, compute_salience,
                SUPER_EPISODE_MIN_CLUSTER,
            )
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} Super-episode creation failed for cluster {cluster_ids}: {exc}"
            )


def _process_super_cluster(
    cluster_ids, channel, db, episodic_svc, emb_svc, prior_embeddings,
    super_episode_encoder_cls, compute_novelty, compute_salience, min_cluster,
) -> None:
    """Assemble sources, invoke encoder, compute salience, and persist one super-episode."""
    sources = [ep for ep in (episodic_svc.get_episode_by_id(eid) for eid in cluster_ids) if ep]
    if len(sources) < min_cluster:
        return

    transcript_spans = _fetch_transcript_spans(sources, db)
    response = super_episode_encoder_cls(sources, transcript_spans).send()
    if not response:
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response for cluster {cluster_ids}")
        return

    super_ep = _safe_json_load_object(response)
    if not super_ep or not super_ep.get('gist'):
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty gist for cluster {cluster_ids}")
        return

    super_ep['channel'] = channel
    unique_t_ids = sorted(_collect_transcript_ids(sources))
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
    super_ep.pop('has_open_loop', None)

    new_id = episodic_svc.store_episode(super_ep, embedding=embedding)
    for src_id in cluster_ids:
        episodic_svc.set_consolidated_into(src_id, new_id)

    logger.info(f"{LOG_PREFIX} Super-episode {new_id} created from cluster {cluster_ids}")


def _safe_json_load_object(text: str) -> dict:
    """Parse a JSON object from LLM response text. Returns {} on any failure."""
    import json as _json

    if not text:
        return {}
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = _strip_code_fence(text)
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned non-dict JSON")
        return {}
    except ValueError:
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable JSON")
        return {}


def write_input_row(channel: str, role: str, content: str) -> int:
    """Write the input transcript row. Returns the row ID."""
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

    _maybe_trigger_extraction(channel, row_id)

    return row_id


def write_assistant_row(channel: str, content: str) -> int:
    """Write the assistant transcript row and fire the rolling episode extraction trigger."""
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

    _maybe_trigger_extraction(channel, row_id)

    return row_id
