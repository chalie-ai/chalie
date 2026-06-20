"""
Transcript Service — persistent conversation record.
"""

import logging
import threading
from collections import Counter
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from services.database_service import DatabaseService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TRANSCRIPT]"

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


def _resolve_location(lat: float | None, lon: float | None, name: str | None, channel: str) -> tuple[float | None, float | None, str | None]:
    """Back-fill live location ONLY for channels whose source profile permits it
    (``location_backfill`` — user activity). Muted / non-user-activity channels
    store NULL location so background work never corrupts the geo signal.
    Returns (None, None, None) on any failure.
    """
    if lat is not None or lon is not None or name is not None:
        return lat, lon, name
    # Bidirectional dependency: the per-source allowlist lives in
    # services/source_profiles.py; this is the location-backfill consumer.
    from services.source_profiles import profile_for

    if not profile_for(channel).location_backfill:
        return None, None, None
    try:
        from services.locale_service import get_location
        loc = get_location()
        return cast("float | None", loc.get('lat')), cast("float | None", loc.get('lon')), cast("str | None", loc.get('name'))
    except Exception:
        return None, None, None


def append(
    channel: str,
    role: str,
    content: str,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    internal: bool = False,
    location_lat: float | None = None,
    location_lon: float | None = None,
    location_name: str | None = None,
) -> int | None:
    if not channel:
        return None
    if not content and role != 'assistant':
        return None

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        lat, lon, loc_name = _resolve_location(
            location_lat, location_lon, location_name, channel
        )

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcript (
                    channel, role, content, tool_call_id, tool_name, internal,
                    xml_migrated, location_lat, location_lon, location_name
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (channel, role, content, tool_call_id, tool_name, 1 if internal else 0,
                 lat, lon, loc_name),
            )
            rowid = cursor.lastrowid
            cursor.close()

        _maybe_trigger_extraction(channel, rowid)

        return rowid

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to append: {e}")
        return None



def get_recent(channel: str, limit: int = 20, since_id: int | None = None, _context: object = None) -> list[dict[str, object]]:
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            if since_id is not None:
                cursor.execute(
                    """
                    SELECT id, role, content, tool_call_id, tool_name, internal, created_at, turn_id
                    FROM transcript
                    WHERE channel = ? AND id > ?
                    ORDER BY id ASC
                    """,
                    (channel, since_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, role, content, tool_call_id, tool_name, internal, created_at, turn_id
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
                'turn_id': r[7],
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


def cleanup_unlinked_entries(channel: str | None = None) -> int:
    try:
        from services import compaction_persistence
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        # Build a list of (channel, watermark) pairs from the append-only
        # tool_calls compaction audit rows.
        if channel:
            row = compaction_persistence.get_compaction(channel)
            watermarks = [(channel, row['compacted_up_to_id'])] if row else []
        else:
            with db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT t.channel
                    FROM tool_calls tc
                    JOIN transcript t ON t.id = tc.transcript_id
                    WHERE tc.tool_name = 'compaction'
                      AND json_extract(tc.params, '$.status') = 'success'
                    """
                ).fetchall()
            channels = [r[0] for r in rows]
            watermarks = []
            for ch in channels:
                row = compaction_persistence.get_compaction(ch)
                if row:
                    watermarks.append((ch, row['compacted_up_to_id']))

        if not watermarks:
            return 0

        with db.connection() as conn:
            cursor = conn.cursor()
            total_deleted = 0

            for t, watermark in watermarks:
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
                referenced_ids: set[int] = set()
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


# ── Internal helpers ─────────────────────────────────────────────────


def _maybe_trigger_extraction(channel: str, rowid: int | None) -> None:
    """Fire episode extraction when the channel has accumulated
    _EXTRACTION_THRESHOLD untriggered transcripts.

    Gate: only channels whose source profile has ``extract_episodes`` produce
    episodes (user, dmn, external-agent:*). Every other channel (delegate:*,
    skills_building, scheduled, …) resolves to the muted default and returns
    here before any DB work, keeping the episodes table free of housekeeping
    noise and bounding which channels SubconsciousWorker._step_consolidate()
    iterates.

    DB-state-driven: counts transcripts with id > MAX(episodes.transcript_id_end)
    for the channel. When count >= threshold, fire. No process-local state,
    so restarts and container rebuilds cannot desync the trigger from the
    actual tail of accumulated history.

    Never raises — failures logged and silently ignored.
    """
    if rowid is None:
        return
    # Bidirectional dependency: the per-source allowlist lives in
    # services/source_profiles.py; this is the episode-gate consumer.
    from services.source_profiles import profile_for

    if not profile_for(channel).extract_episodes:
        return

    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM transcript
                WHERE channel = ?
                  AND role != 'compaction'
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

    Never raises — any failure is logged only.
    """
    def _run() -> None:
        try:
            import threading

            from configs.channels import EpisodeEncoderConfig
            from services.database_service import get_shared_db_service
            from services.message_processor import MessageProcessor
            from services.episodic_service import (
                EpisodicService, _fetch_novelty_comparison_set, compute_novelty,
            )
            from services.salience_service import compute_salience
            from services.embedding_service import get_embedding_service

            db = get_shared_db_service()

            # ── 1. Fetch the window ──────────────────────────────────────────
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, role, content, tool_name, created_at,
                           location_lat, location_lon, location_name
                    FROM transcript
                    WHERE channel = ? AND id <= ? AND role != 'compaction'
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
                    'location_lat': r[5],
                    'location_lon': r[6],
                    'location_name': r[7],
                    'channel': channel,
                }
                for r in reversed(rows)
            ]

            # ── 2. Format window string ──────────────────────────────────────
            window_str = _format_window_entries(entries)

            # ── 3. Fetch referenced episodes via tool_calls ──────────────────
            referenced_episodes = _fetch_referenced_episodes(entries, db)
            referenced_str = _format_episodes_for_prompt(referenced_episodes)

            # ── 4. Encode the window via the flat episode-encoder channel ────
            # _window / _referenced are read by EpisodeEncoderConfig's
            # get_user_prompt; set them on the instance before _run().
            emp = object.__new__(MessageProcessor)
            MessageProcessor.__init__(emp, "", None)
            emp.config = EpisodeEncoderConfig()
            emp.uid = None
            emp.cancel_event = threading.Event()
            emp.thinking_level = "low"
            setattr(emp, '_window', window_str)
            setattr(emp, '_referenced', referenced_str)
            response = emp._run()
            snapshots = cast("list[dict[str, object]]", _safe_json_load(response))
            if not snapshots:
                return

            # ── 5. Resolve service handles ───────────────────────────────────
            episodic_svc = EpisodicService(db)
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
                            episodic_svc.soft_delete(cast("str", delete_id))
                        continue

                    # Filter transcript_ids to valid window IDs
                    raw_ids = cast("list[object]", ep.get('transcript_ids') or [])
                    ep['transcript_ids'] = [i for i in raw_ids if i in valid_ids]
                    if not ep['transcript_ids']:
                        continue

                    ep['transcript_id_start'] = min(cast("list[int]", ep['transcript_ids']))
                    ep['transcript_id_end'] = max(cast("list[int]", ep['transcript_ids']))
                    ep['channel'] = channel

                    dominant_location = _aggregate_dominant_location(entries)
                    if dominant_location.get('lat') is not None:
                        ep['location_lat'] = dominant_location['lat']
                        ep['location_lon'] = dominant_location['lon']
                        ep['location_name'] = dominant_location.get('name')

                    gist = cast("str", ep.get('gist', '') or '')
                    embedding = emb_svc.generate_embedding(gist) if gist else None

                    novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
                    ep['salience'] = compute_salience(
                        valence=float(cast("float", ep.get('emotional_valence') or 0.0)),
                        arousal=float(cast("float", ep.get('emotional_arousal') or 0.0)),
                        has_open_loop=bool(ep.get('has_open_loop', False)),
                        novelty=novelty,
                    )

                    # Pop transient fields — not persisted
                    ep.pop('has_open_loop', None)
                    update_id = ep.pop('update_id', None)
                    ep.pop('delete_id', None)  # defensive — should be None here

                    if update_id:
                        episodic_svc.update_episode(cast("str", update_id), ep, embedding=embedding)
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

        except Exception as e:
            logger.warning(
                f"{LOG_PREFIX} Episode extraction trigger failed "
                f"(channel={channel}, rowid={rowid}): {e}"
            )

    threading.Thread(target=_run, daemon=True).start()


# ── Episode extraction helpers ───────────────────────────────────────────────


def _aggregate_dominant_location(entries: list[dict[str, object]]) -> dict[str, object]:
    """When all location names are unique (no majority), falls back to the most recent non-null row."""
    located = [
        e for e in entries
        if e.get('location_lat') is not None or e.get('location_name') is not None
    ]
    if not located:
        return {'lat': None, 'lon': None, 'name': None}

    name_counts = Counter(
        e['location_name'] for e in located if e.get('location_name') is not None
    )

    if name_counts:
        top_name, top_count = name_counts.most_common(1)[0]
        if top_count > 1:
            # Clear dominant name — use any row with that name for coords
            for e in reversed(located):
                if e.get('location_name') == top_name:
                    return {
                        'lat': e.get('location_lat'),
                        'lon': e.get('location_lon'),
                        'name': top_name,
                    }

    # All names are unique (or no names) — use the most recent located row
    most_recent = located[-1]
    return {
        'lat': most_recent.get('location_lat'),
        'lon': most_recent.get('location_lon'),
        'name': most_recent.get('location_name'),
    }


def _format_window_entries(entries: list[dict[str, object]]) -> str:
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


def _parse_episode_ids_from_results(result_texts: list[str]) -> set[str]:
    """Extract episode IDs from rendered memory recall envelopes.

    Only rows with ``kind == "episode"`` are collected — data-graph rows are
    keyed by data-graph key (not an episode id) and are intentionally skipped.
    Malformed / non-recall envelopes are skipped silently.
    """
    import json as _json

    episode_ids: set[str] = set()
    for text in result_texts:
        if not text or "[end:memory]" not in text:
            continue
        nl = text.find("]\n")
        end = text.find("\n[end:memory]")
        if nl == -1 or end == -1 or end <= nl:
            continue
        body = text[nl + 2:end]
        try:
            parsed = _json.loads(body)
        except (ValueError, TypeError):
            continue
        rows = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("kind") == "episode":
                eid = str(row.get("id", "")).strip()
                if eid:
                    episode_ids.add(eid)
    return episode_ids


def _fetch_referenced_episodes(entries: list[dict[str, object]], db: "DatabaseService") -> list[dict[str, object]]:
    """Query tool_calls for memory skill invocations within the window.

    ``tool_name='memory'`` covers both auto-seed and LLM-invoked recall —
    both are dispatched under the same tool name.
    """

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

    # Parse episode IDs out of each memory recall envelope. recall renders a
    # structured JSON body — ``[memory(status=success, …)]\n{"results": [{"id":…,
    # "kind":"episode", …}, …]}\n[end:memory]`` — so we pull the JSON between the
    # open tag and ``[end:memory]`` and collect the ``id`` of every row whose
    # ``kind`` is ``episode`` (data-graph rows carry their own kind and are keyed
    # by data-graph key, not an episode id, so they are skipped).
    episode_ids = _parse_episode_ids_from_results([row[0] or '' for row in rows])

    if not episode_ids:
        return []

    try:
        from services.episodic_service import EpisodicService
        episodic_svc = EpisodicService(db)
        episodes = []
        for eid in episode_ids:
            ep = episodic_svc.get_episode_by_id(eid)
            if ep:
                episodes.append(ep)
        return episodes
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} _fetch_referenced_episodes fetch failed: {exc}")
        return []


def _format_episodes_for_prompt(episodes: list[dict[str, object]]) -> str:
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


def _safe_json_load(text: str) -> list[object]:
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


def _is_delete_only(ep: dict[str, object]) -> bool:
    if not ep.get('delete_id'):
        return False
    # All other meaningful fields must be absent or null/empty
    meaningful = ('gist', 'transcript_ids', 'update_id')
    return not any(ep.get(f) for f in meaningful)


def write_input_row(channel: str, role: str, content: str) -> int:
    """Write a turn's anchoring input row, opening the next turn.

    turn_id is the per-channel monotonic turn boundary. Every input row opens a
    fresh turn — there is no caller-supplied turn_id: fresh user / external
    input, an async re-entry and a compaction checkpoint each open their OWN
    turn. The next value, ``MAX(turn_id)+1`` for the channel, is computed inside
    the INSERT so the allocation is atomic with the write — no read-then-insert
    race when two same-channel turns open concurrently. Read it back with
    :func:`turn_id_of_row`.
    """
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    lat, lon, loc_name = _resolve_location(None, None, None, channel)
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transcript (channel, role, content, xml_migrated,
                                    location_lat, location_lon, location_name, turn_id)
            VALUES (?, ?, ?, 1, ?, ?, ?,
                    (SELECT COALESCE(MAX(turn_id), 0) + 1
                     FROM transcript WHERE channel = ?))
            """,
            (channel, role, content, lat, lon, loc_name, channel),
        )
        row_id = cursor.lastrowid
        cursor.close()

    _maybe_trigger_extraction(channel, row_id)

    return cast("int", row_id)


def turn_id_of_row(row_id: int) -> int:
    """The turn_id the INSERT's COALESCE subquery opened for a transcript row.

    write_input_row allocates the next turn atomically inside the write (under
    the SQLite writer lock), so the caller cannot know the value up front — it
    reads it back here by row id. Reading MAX(turn_id) instead would race: a
    concurrent same-channel turn could advance the max between the write and the
    read. Returns 0 if the row is gone or still unnumbered."""
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        row = conn.execute(
            "SELECT turn_id FROM transcript WHERE id = ?", (row_id,)
        ).fetchone()
    return row[0] if row and row[0] is not None else 0


def link_transcript_doc(transcript_id: int, doc_id: str) -> None:
    """Link an uploaded document to the transcript turn that carried it.

    Powers chat-attachment persistence across page refresh: the live preview is a
    browser-only blob: URL, so on reload the rebuild (api.conversation
    .get_recent_history) joins this table to re-render the image/file from
    /documents/<id>/preview.  INSERT OR IGNORE keeps it idempotent against the
    composite primary key.  Called from message_processor._seed_upload_attachment.
    """
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO transcript_docs (transcript_id, doc_id) VALUES (?, ?)",
            (transcript_id, doc_id),
        )


def latest_input_content(channel: str) -> "str | None":
    """The most recent input-row content on a channel — the post-compaction
    continuation's "the user query was: …".

    An input row is any NON-assistant, NON-compaction row: there are many input
    roles (user / proactive_thought / external_agent / vision / …), so we exclude
    the two output-shaped roles rather than hardcode role='user'. Compaction
    checkpoints are written via write_input_row(channel, 'compaction', …) so they
    ARE non-assistant rows and must be excluded explicitly. Ordered by monotonic
    id (not created_at — one-second granularity ties). Returns None on an empty
    channel."""
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    with db.connection() as conn:
        row = conn.execute(
            "SELECT content FROM transcript "
            "WHERE channel = ? AND role != 'assistant' AND role != 'compaction' "
            "ORDER BY id DESC LIMIT 1",
            (channel,),
        ).fetchone()
    return row[0] if row else None


def write_assistant_row(channel: str, content: str, turn_id: "int | None" = None) -> int:
    """Write one assistant row for a single chain step, grounded to its turn.

    Under the recursive turn chain (TKT-1070) every step persists its own row
    via ``MessageProcessor._store_row`` — the prose of each tool-bearing step
    plus the final settle text — so one turn produces MULTIPLE assistant rows
    that share a ``turn_id``, not a single end message. The MP supplies its
    current ``turn_id`` so each row shares the boundary of the turn's input row.
    A ``None`` turn_id falls back to the same fresh-turn allocation as
    write_input_row — the path an anchorless re-entry (no input row) takes to
    open its own turn."""
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    lat, lon, loc_name = _resolve_location(None, None, None, channel)
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transcript (channel, role, content, xml_migrated,
                                    location_lat, location_lon, location_name, turn_id)
            VALUES (?, ?, ?, 1, ?, ?, ?,
                    COALESCE(?, (SELECT COALESCE(MAX(turn_id), 0) + 1
                                 FROM transcript WHERE channel = ?)))
            """,
            (channel, 'assistant', content, lat, lon, loc_name, turn_id, channel),
        )
        row_id = cursor.lastrowid
        cursor.close()

    _maybe_trigger_extraction(channel, row_id)

    return cast("int", row_id)
