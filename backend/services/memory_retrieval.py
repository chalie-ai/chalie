"""Memory retrieval + mutation engine — the substance behind the ``memory`` ability.

The ``memory`` tool (``abilities/memory.py``) is a thin adapter: its
``run()`` reads the action and the bound channel, then delegates to the
handlers here. This module owns everything non-ability — the
store/recall/reflect/forget handlers, the episode recall engine, the
data-graph search, the reflection layer expansion, response formatting,
and recall telemetry — so the same engine is reachable from non-ability
callers without importing an ability.

Every handler returns an ``abilities._result.ToolResult`` — never a
string. ``recall`` returns a STRUCTURED body
(``{"results": [{id, content, score, kind, created_at}, …]}``, +
``fallback`` on explicit recalls). When some (not all) backend lanes
error, recall succeeds with ``meta degraded=true`` so the partial result
is honest. A dead retrieval backend surfaces as
``ToolResult.err(code='memory-backend-error')`` rather than a silent
``results=0`` — the model must never be told "nothing is stored" when the
store simply failed.

Episode recall is cross-channel: the read path never filters by the
caller's own channel, so a memory encoded on any episode-producing channel
is recallable from any turn — exactly as facts already cross-pollinate via
the channel-agnostic ``data_graph.recall``. Muted channels write no episodes,
so the channel-agnostic read naturally scopes to the set that actually holds
memories. The caller's channel is recorded only for ``memory_recall_log``
provenance and the per-channel feeling-of-knowing signal, never as a recall
scope.

Two recall callers stay distinct only in their side-effects:
``caller='seed'`` is the silent turn-0 auto-seed (no fallback hint, no
fan-out, no user-curated moments lane). ``caller='llm_recall'`` is the
explicit recall — appends the fallback hint naming ``document`` and
``schedule`` tools, and adds the labeled moments lane (``kind='moment'``)
so pinned bookmarks surface alongside facts/episodes.
"""

import hashlib
import json
import logging
from typing import TYPE_CHECKING, cast

from abilities._result import ToolResult

if TYPE_CHECKING:
    from services.database_service import DatabaseService
    from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)
LOG_PREFIX = "[MEMORY]"


# ── Store ────────────────────────────────────────────────────────────


def handle_store(channel: str, params: dict[str, object]) -> ToolResult:
    key = cast("str | None", params.get("key"))
    value = params.get("value")
    kind = cast("str", params.get("kind", "user_specific"))

    # The proactive-research loop never picks a kind: any store on its channel is
    # a discovery memory. Routing by channel keeps a weak model from misfiling it.
    from services.source_profiles import CHANNEL_DISCOVERY
    if channel == CHANNEL_DISCOVERY:
        from services.data_graph_service import KIND_DISCOVERY
        kind = KIND_DISCOVERY

    if not key:
        return ToolResult.err(
            "store needs a 'key' naming the fact.",
            code="key-required",
            action="store",
            hint="pass a canonical 'key' (e.g. 'residence', 'employment').",
        )
    if value is None:
        return ToolResult.err(
            "store needs a 'value' — the fact to remember.",
            code="value-required",
            action="store",
            hint="pass the atomic 'value' to store under the key.",
        )

    from services.data_graph_service import get_data_graph_service

    dgs = get_data_graph_service()
    result = dgs.store(kind=kind, key=key, value=str(value), source=f"skill:memory:store:{channel}")

    if result is None:
        return ToolResult.err(
            f"Could not store '{key}': '{kind}' is not a valid memory kind.",
            code="invalid-kind",
            action="store",
            key=key,
            valid=("user_specific", "system", "misc"),
        )

    body = _format_store_response(result)
    return ToolResult.ok(body, action="store", key=key)


def _format_store_response(result: dict[str, object]) -> str:
    status = result.get("status", "")
    canonical = result.get("canonical_key", "")
    provided = result.get("provided_key", "")
    value = result.get("value", "")
    date = result.get("date")

    if canonical != provided:
        key_display = f"'{canonical}' (canonical of '{provided}')"
    else:
        key_display = f"'{canonical}'"

    if status == "created":
        rule = result.get("rule")
        if rule == "coexist":
            return f"{key_display} saved as ['{value}']."
        return f"{key_display} saved as '{value}'."

    if status == "reinforced":
        return f"{key_display} was already set on {date}. Memory reinforced."

    if status == "superseded":
        old = result.get("old_value", "")
        return f"{key_display} updated to '{value}'. Supersedes '{old}' (previously set on {date})."

    if status == "conflict":
        old = result.get("old_value", "")
        return (
            f"{key_display} is immutable. Existing value '{old}' (set {date}) kept. "
            f"New value '{value}' rejected. Use 'forget' first if you're sure."
        )

    if status == "appended":
        all_vals = cast("list[object]", result.get("all_values") or [])
        vals_str = ", ".join(f"'{v}'" for v in all_vals)
        return f"{key_display} updated. Values now: [{vals_str}] (previously updated on {date})."

    if status == "lut_miss_created":
        return f"'{provided}' saved as '{value}'."

    if status == "lut_miss_reinforced":
        return f"'{provided}' already set to '{value}'. Memory reinforced."

    if status == "lut_miss_appended":
        all_vals = cast("list[object]", result.get("all_values") or [])
        vals_str = ", ".join(f"'{v}'" for v in all_vals)
        return f"'{provided}' updated. Values now: [{vals_str}]."

    return f"'{provided}' stored."


# ── Forget ───────────────────────────────────────────────────────────


def handle_forget(params: dict[str, object]) -> ToolResult:
    key = cast("str | None", params.get("key"))
    value = cast("str | None", params.get("value"))
    kind = cast("str", params.get("kind", "user_specific"))

    if not key:
        return ToolResult.err(
            "forget needs a 'key' naming the memory to remove.",
            code="key-required",
            action="forget",
            hint="pass the canonical 'key' of the fact to forget.",
        )

    from services.data_graph_service import get_data_graph_service

    dgs = get_data_graph_service()
    result = dgs.forget(kind=kind, key=key, value=value)

    if result is None:
        return ToolResult.err(
            f"Could not forget '{key}': '{kind}' is not a valid memory kind.",
            code="invalid-kind",
            action="forget",
            key=key,
            valid=("user_specific", "system", "misc"),
        )

    body = _format_forget_response(result)
    return ToolResult.ok(body, action="forget", key=key)


def _format_forget_response(result: dict[str, object]) -> str:
    status = result.get("status", "")
    canonical = result.get("canonical_key", "")
    provided = result.get("provided_key", "")
    value = result.get("value")
    date = result.get("date")

    if canonical and provided and canonical != provided:
        key_display = f"'{canonical}' (canonical of '{provided}')"
    else:
        key_display = f"'{canonical or provided}'"

    if status == "forgotten":
        rule = result.get("rule")
        if rule == "coexist":
            remaining = cast("list[object]", result.get("remaining_values") or [])
            vals_str = ", ".join(f"'{v}'" for v in remaining)
            return f"'{value}' removed from {key_display}. Remaining: [{vals_str}]."
        old = result.get("old_value") or value
        return f"{key_display} forgotten (was '{old}', set {date})."

    if status == "forgotten_all":
        n = result.get("versions_removed", 0)
        return f"{key_display} forgotten. All {n} versions removed."

    if status == "forgotten_empty":
        return f"'{value}' removed from {key_display}. No values remain — key fully forgotten."

    if status == "value_not_found":
        remaining = cast("list[object]", result.get("remaining_values") or [])
        vals_str = ", ".join(f"'{v}'" for v in remaining)
        return f"'{value}' not found in {key_display}. Currently stored: [{vals_str}]."

    if status == "not_found":
        return f"No memory stored under {key_display}. Nothing to forget."

    if status == "error":
        return f"{LOG_PREFIX} {result.get('message', 'Unknown error')}"

    return f"{key_display} forget operation completed."


# ── Recall ───────────────────────────────────────────────────────────

# Guardrail appended to every explicit (model-invoked) recall so the model
# falls back to the document/schedule stores ON ITS OWN JUDGEMENT rather than
# memory.recall silently dispatching those searches behind its back. The
# turn-0 auto-seed recall (_auto=True) stays silent — no hint, no fan-out.
# Tool names are exact: `document` and `schedule`, each with action="search".
_RECALL_FALLBACK_HINT = (
    "If you cannot find the information in memory, try using the "
    "`document` (action: search) or `schedule` (action: search) tools."
)

# Stable, machine-readable code surfaced when EVERY retrieval backend a recall
# touched failed (e.g. a dead sqlite-vec extension). A weak model must be able to
# tell "the store is broken" apart from "nothing is stored" — otherwise it
# confidently asserts the user never said something it simply could not look up.
_BACKEND_ERROR_CODE = "memory-backend-error"
_BACKEND_ERROR_HINT = (
    "The memory store is unavailable right now — this is an infrastructure "
    "failure, NOT a confirmation that nothing is stored. Do not tell the user "
    "you have no record; say you could not reach memory and try again later."
)

# Label stamped on the user-curated moments lane in an explicit recall, so the
# model (and the transcript back-reference) can tell a pinned bookmark apart from
# a generic data_graph fact. Moments are surfaced ONLY on explicit recall, never
# in the silent turn-0 auto-seed.
_MOMENT_KIND = "moment"
# A pinned moment is a deliberate, high-signal user bookmark, so its lane rows
# carry a "high" relevance label regardless of lexical/semantic distance.
_MOMENT_RELEVANCE = "high"
_MOMENT_CONFIDENCE = 0.9


def _is_backend_error(status: str) -> bool:
    """A search helper signals an infra failure with a status prefixed ``error:``.

    Genuine empties report ``0 matches`` / ``0 matches (N candidates evaluated)``;
    only a real backend exception yields ``error: …`` — the single discriminator
    between "the store is broken" and "nothing matched".
    """
    return isinstance(status, str) and status.startswith("error:")


def handle_recall(mp: "MessageProcessor | None", channel: str, params: dict[str, object]) -> ToolResult:
    query = cast("str", params.get("query", ""))
    location = cast("str", params.get("location", ""))
    if not query and not location:
        return ToolResult.err(
            "Recall needs a 'query' or a 'location' to search for.",
            code="no-query-or-location",
            action="recall",
            hint="pass a 'query' (a topic) and/or a 'location'.",
        )

    # The turn-0 background seed (_auto=True) stays silent (no fallback hint);
    # an explicit model-invoked recall carries the fallback guardrail. Both run
    # the identical ranked retrieval — the radius split is gone.
    caller = "seed" if params.get("_auto") else "llm_recall"

    limit = 10
    results: list[dict[str, object]] = []
    # Track every backend the recall actually queried so a dead store surfaces as
    # a loud error (all-failed) or a degraded success (some-failed) instead of a
    # silent "0 results". ``statuses`` only holds the lanes we ran for this call.
    statuses: list[str] = []

    if query and location:
        # AND gate: only episodes that satisfy both location AND semantic query.
        loc_hits, loc_status = _search_episodes_by_location(location, limit * 3)
        loc_ids = {h["id"] for h in loc_hits}
        loc_by_id = {h["id"]: h for h in loc_hits}

        sem_hits, sem_status = _search_episodes(mp, query, limit * 3, caller=caller)
        sem_ids = {h["id"] for h in sem_hits}
        sem_by_id = {h["id"]: h for h in sem_hits}

        matched_ids = loc_ids & sem_ids
        for ep_id in matched_ids:
            hit = dict(loc_by_id[ep_id])
            sem_hit = sem_by_id[ep_id]
            hit["confidence"] = sem_hit["confidence"]
            hit["relevance"] = sem_hit["relevance"]
            results.append(hit)

        dg_hits, dg_status = _search_data_graph(query, limit)
        results.extend(dg_hits)
        statuses.extend([loc_status, sem_status, dg_status])

        # The user-curated moments lane is explicit-recall only — never the
        # silent turn-0 seed (caller='seed'), so pinned bookmarks stay out of the
        # auto-flashback.
        if caller == "llm_recall":
            mom_hits, mom_status = _search_moments(query, limit)
            results.extend(mom_hits)
            statuses.append(mom_status)

    elif location:
        hits, loc_status = _search_episodes_by_location(location, limit)
        results.extend(hits)
        statuses.append(loc_status)

    else:
        dg_hits, dg_status = _search_data_graph(query, limit)
        results.extend(dg_hits)

        ep_hits, sem_status = _search_episodes(mp, query, limit, caller=caller)
        results.extend(ep_hits)
        statuses.extend([dg_status, sem_status])

        # Explicit-recall-only moments lane (see the AND-gate branch above): the
        # turn-0 auto-seed never surfaces pinned bookmarks.
        if caller == "llm_recall":
            mom_hits, mom_status = _search_moments(query, limit)
            results.extend(mom_hits)
            statuses.append(mom_status)

    errored = [s for s in statuses if _is_backend_error(s)]
    # All lanes failed → the store is down. Surface a loud, stable error so the
    # model knows it could not look up rather than that nothing is stored.
    if statuses and len(errored) == len(statuses):
        logger.warning(
            "%s recall hit a dead backend (all %d lane(s) errored): %s",
            LOG_PREFIX, len(statuses), "; ".join(errored),
        )
        return ToolResult.err(
            "Could not search memory — the retrieval backend failed.",
            code=_BACKEND_ERROR_CODE,
            hint=_BACKEND_ERROR_HINT,
            query=query or location,
        )

    degraded = bool(errored)
    if degraded:
        logger.warning(
            "%s recall degraded — %d/%d backend lane(s) errored: %s",
            LOG_PREFIX, len(errored), len(statuses), "; ".join(errored),
        )

    partial = sum(1 for r in results if cast("float", r.get("confidence", 0)) < 0.5)
    _store_fok_signal(channel, partial)

    body: dict[str, object] = {"results": _recall_payload(results)}
    # Explicit recalls carry the fallback guardrail in the structured body; the
    # silent turn-0 seed (caller='seed') carries no fallback and fires no fan-out.
    if caller != "seed":
        body["fallback"] = _RECALL_FALLBACK_HINT

    return ToolResult.ok(
        body,
        query=query or location,
        results=len(results),
        degraded=degraded,
    )


# ── Reflect ──────────────────────────────────────────────────────────


def handle_reflect(mp: "MessageProcessor | None", params: dict[str, object]) -> ToolResult:
    query = cast("str", params.get("query", ""))
    if not query:
        return ToolResult.err(
            "reflect needs a 'query' — the topic to deep-search.",
            code="no-query",
            action="reflect",
            hint="pass a 'query' naming the topic to reflect on.",
        )

    raw_episodes, _ = recall_episodes(
        mp,
        query=query,
        caller="llm_recall",
        limit=1,
        return_raw=True,
    )

    if not raw_episodes:
        dg_hits, _ = _search_data_graph(query, 2)
        if not dg_hits:
            return ToolResult.ok("", action="reflect", query=query, results=0)
        body = _format_reflect(query, None, [], dg_hits)
        return ToolResult.ok(body, action="reflect", query=query)

    top = raw_episodes[0]

    supporting = _expand_episode_layers(top)

    dg_hits, _ = _search_data_graph(query, 2)

    body = _format_reflect(query, top, supporting, dg_hits)
    return ToolResult.ok(body, action="reflect", query=query)


def _expand_episode_layers(episode: dict[str, object]) -> list[dict[str, object]]:
    from services.database_service import get_shared_db_service

    try:
        db = get_shared_db_service()
        results: list[dict[str, object]] = []
        _expand_recursive(db, episode, results, depth=0, max_depth=3)
        return results
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Reflect layer expansion failed: {exc}")
        return []


def _expand_recursive(
    db: "DatabaseService", episode: dict[str, object], results: list[dict[str, object]], depth: int, max_depth: int
) -> None:
    if depth >= max_depth:
        return

    consolidated_from = _parse_json_list(episode.get("consolidated_from"))
    transcript_ids = _parse_json_list(episode.get("transcript_ids"))

    if transcript_ids and not consolidated_from:
        results.extend(_fetch_transcript_entries(transcript_ids))
        return

    if consolidated_from:
        children = _fetch_episodes_by_ids(db, consolidated_from)
        for child in children:
            results.append({
                "type": "episode",
                "content": child.get("gist", ""),
                "salience": child.get("salience", 0),
            })
            _expand_recursive(db, child, results, depth + 1, max_depth)


def _parse_json_list(raw: object) -> list[object]:
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(cast("str", raw))
        return cast("list[object]", parsed) if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _fetch_episodes_by_ids(db_service: "DatabaseService", episode_ids: list[object]) -> list[dict[str, object]]:
    if not episode_ids:
        return []
    try:
        with db_service.connection() as conn:
            placeholders = ",".join("?" for _ in episode_ids)
            cursor = conn.execute(
                f"SELECT id, gist, salience, transcript_ids, consolidated_from "
                f"FROM episodes WHERE id IN ({placeholders}) AND deleted_at IS NULL "
                f"ORDER BY salience DESC",
                list(episode_ids),
            )
            return [dict(r) for r in cursor.fetchall()]
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Episode fetch by IDs failed: {exc}")
        return []


def _fetch_transcript_entries(transcript_ids: list[object]) -> list[dict[str, object]]:
    from services.transcript_service import Transcript
    if not transcript_ids:
        return []
    try:
        return [
            {"type": "transcript", "content": r["content"] or "", "salience": None}
            for r in Transcript.by_ids([int(cast(int, tid)) for tid in transcript_ids])
        ]
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Transcript fetch failed: {exc}")
        return []


def _format_reflect(
    query: str,
    top_episode: dict[str, object] | None,
    supporting: list[dict[str, object]],
    dg_hits: list[dict[str, object]],
) -> str:
    lines = []

    lines.append("## Main Memory")
    if top_episode:
        lines.append(f'**The most relevant memory to "{query}"**')
        lines.append(cast("str", top_episode.get("gist", "")))
    else:
        lines.append(f'No episode memories found for "{query}"')

    if supporting:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("### Supporting Memories")
        lines.append("__ordered by salience__")
        lines.append("")
        for entry in supporting:
            content = entry.get("content", "")
            salience = entry.get("salience")
            if salience is not None:
                lines.append(f"* {content} [salience: {salience}]")
            else:
                lines.append(f"* {content}")

    if dg_hits:
        lines.append("")
        lines.append("### Supporting facts:")
        for hit in dg_hits:
            key = hit.get("id", "")
            value = hit.get("text", "")
            lines.append(f"[{key}] {value}")

    return "\n".join(lines)


def _relevance_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _render_behavioral_pattern(raw_value: str) -> str:
    """Parse a behavioral_pattern JSON value into a compact human-readable line."""
    try:
        c = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        name = c.get("name", "unknown")
        freq = c.get("frequency", "?")
        anchor = c.get("time_anchor") or ""
        summary = c.get("summary", "")
        confidence = c.get("confidence", 0)
        anchor_part = f" @ {anchor}" if anchor else ""
        return f"{name} ({freq}{anchor_part}): {summary} [confidence={confidence}]"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return str(raw_value) if raw_value is not None else ""


def _search_data_graph(query: str, limit: int) -> tuple[list[dict[str, object]], str]:
    try:
        from services.data_graph_service import (
            get_data_graph_service,
            KIND_BEHAVIORAL_PATTERN,
            KIND_PLACE,
            KIND_USER_SPECIFIC,
            KIND_SYSTEM,
            KIND_MISC,
            KIND_DISCOVERY,
        )

        # Moments are deliberately absent here: they live in the dedicated
        # ``moments`` table (services/moments_service.py), surfaced as their own
        # labeled explicit-recall lane via ``_search_moments`` — not as part of
        # generic data_graph recall.
        dgs = get_data_graph_service()
        rows = dgs.recall(
            query=query,
            kinds=[KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC,
                   KIND_BEHAVIORAL_PATTERN, KIND_PLACE, KIND_DISCOVERY],
            limit=limit,
        )

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            cos = cast("float", row.get("cos_score", 0.0))
            kind = cast("str", row.get("kind", ""))
            text = cast("str", row.get("value", ""))
            if kind == KIND_BEHAVIORAL_PATTERN:
                text = _render_behavioral_pattern(text)
            hits.append({
                "id": row.get("key", ""),
                "kind": kind,
                "text": text,
                "relevance": _relevance_label(cos),
                "confidence": cos,
                "created_at": None,
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Data graph search failed: {e}")
        return [], f"error: {e}"


def _search_moments(query: str, limit: int) -> tuple[list[dict[str, object]], str]:
    """Search the user-curated moments lane (FTS + vec) for an explicit recall.

    Moments live in the dedicated ``moments`` table, outside data_graph, so they
    are a distinct labeled lane: every hit carries ``kind="moment"`` so the model
    can tell a pinned bookmark apart from a generic fact. The status string is the
    same ``N matches`` / ``error: …`` discriminator the other lanes use, so a dead
    moments backend degrades the recall rather than masquerading as "0 results".
    """
    try:
        from services.moments_service import get_moments_service

        rows = cast("list[dict[str, object]]", get_moments_service().search(query, limit=limit))
        hits = [
            {
                "id": f"moment_{row.get('transcript_id')}",
                "kind": _MOMENT_KIND,
                "text": row.get("content", ""),
                "relevance": _MOMENT_RELEVANCE,
                "confidence": _MOMENT_CONFIDENCE,
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]
        return hits, f"{len(hits)} matches"
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Moments search failed: {e}")
        return [], f"error: {e}"


def _embedding_hash(embedding: list[float]) -> str:
    if not embedding:
        return "empty"
    try:
        h = hashlib.md5()
        for x in embedding[:16]:
            h.update(f"{x:.6f}".encode())
        return h.hexdigest()[:16]
    except Exception:
        return "err"


def _write_recall_telemetry(
    db_service: "DatabaseService",
    *,
    turn_uid: str,
    transcript_id: int | None,
    channel: str | None,
    caller: str,
    query: str,
    embedding_hash: str,
    telemetry: dict[str, object],
) -> None:
    """Persist one recall observation into ``memory_recall_log``.

    The schema dropped the radius columns (): the row now records the
    new normalised-ranking signals — corpus size, per-lane candidate counts,
    how many candidates the relative score floor dropped, the final surfaced
    count, and the top vector distances.
    """
    try:
        with db_service.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_recall_log (
                    turn_uid, transcript_id, channel, caller, query,
                    query_embedding_hash, episode_count, floor_cut_count,
                    final_rrf_count, top_distances
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_uid,
                    transcript_id,
                    channel,
                    caller,
                    query,
                    embedding_hash,
                    telemetry.get("episode_count", 0),
                    telemetry.get("floor_cut_count", 0),
                    telemetry.get("final_rrf_count", 0),
                    json.dumps(telemetry.get("top_distances", [])),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to write memory_recall_log row: {e}")


def _turn_context(proc: object) -> tuple[str, object, str | None]:
    """Resolve (turn_uid, transcript_id, channel) from the bound processor.

    The radius drift-history apparatus is gone; only the telemetry-keying
    identifiers are still needed. The channel is the caller's own channel,
    recorded purely for ``memory_recall_log`` provenance — episode recall reads
    cross-channel regardless (). Returns ephemeral defaults when no
    processor is bound (e.g. the REST recall path).
    """
    if proc is None:
        return "ephemeral", None, None
    _cfg = getattr(proc, "config", None)
    _channel = getattr(_cfg, "channel", None)
    turn_uid = str(getattr(proc, "_uid", None) or _channel or "unbound")
    transcript_id = getattr(proc, "_uid", None)
    return turn_uid, transcript_id, cast("str | None", _channel)


def recall_episodes(
    mp: "MessageProcessor | None",
    query: str,
    *,
    caller: str = "llm_recall",
    limit: int = 10,
    return_raw: bool = False,
) -> tuple[list[dict[str, object]], str]:
    """Public entry point for episode recall.

    Used by both the `memory` skill's recall action (``caller='llm_recall'``)
    and the pre-turn seed path (``caller='seed'``). Ranking and the relative
    score floor live in ``episodic_retrieval_service.retrieve``; this function
    only embeds the query, routes it, records telemetry, and projects results.

    Episode recall is cross-channel by design (, Decision 1): an episode
    encoded on any episode-producing channel (user, dmn, external-agent:*) is
    recallable from any turn, so the read path never filters by the caller's own
    channel. Muted channels write no episodes, so the channel-agnostic read
    naturally scopes to the set that actually has memories. The caller's channel
    is still recorded in ``memory_recall_log`` for provenance via ``_turn_context``.
    """
    try:
        from services import episodic_retrieval_service
        from services.database_service import get_shared_db_service
        from services.embedding_service import get_embedding_service
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} Episode recall imports failed: {exc}")
        return [], f"error: {exc}"

    try:
        db = get_shared_db_service()
        emb_svc = get_embedding_service()

        q_embedding = emb_svc.generate_embedding(query, mp=mp)
        turn_uid, transcript_id, provenance_channel = _turn_context(mp)

        episodes, telemetry = cast(
            "tuple[list[dict[str, object]], dict[str, object]]",
            episodic_retrieval_service.retrieve(
                query_text=query,
                query_embedding=q_embedding,
                channel=None,
                k=limit,
                return_telemetry=True,
            )
        )

        _write_recall_telemetry(
            db,
            turn_uid=turn_uid,
            transcript_id=cast("int | None", transcript_id),
            channel=provenance_channel,
            caller=caller,
            query=query,
            embedding_hash=_embedding_hash(q_embedding),
            telemetry=telemetry,
        )

        if not episodes:
            candidates = _count_episode_candidates(db)
            status = f"0 matches ({candidates} candidates evaluated)"
            return [], status

        if return_raw:
            return episodes[:limit], f"{len(episodes[:limit])} matches"

        hits = []
        for ep in episodes[:limit]:
            gist = ep.get("gist", "")
            conf = min(1.0, cast("float", ep.get("composite_score", 0)) / 100.0)
            hits.append({
                "id": str(ep.get("id", "")),
                # Full gist verbatim — NO truncation (). The recall block
                # is bounded by the result limit + the request-level cap, not by
                # clipping the text the model reads mid-sentence.
                "text": gist,
                "relevance": _relevance_label(conf),
                "confidence": conf,
                "location": ep.get("location_name"),
                "kind": "episode",
                "created_at": ep.get("created_at"),
            })

        return hits, f"{len(hits)} matches"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Episode search failed: {e}", exc_info=True)
        return [], f"error: {e}"


def _search_episodes(
    mp: "MessageProcessor | None", query: str, limit: int, caller: str = "llm_recall"
) -> tuple[list[dict[str, object]], str]:
    return recall_episodes(
        mp,
        query=query,
        caller=caller,
        limit=limit,
    )


def _count_episode_candidates(db_service: "DatabaseService") -> int:
    """Count recall-eligible episodes across every channel.

    Episode recall is cross-channel (), so the "0 matches (N candidates
    evaluated)" status counts the whole episode corpus — not just one channel's
    slice — to honestly report how many candidates the empty recall searched.
    """
    try:
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL",
            )
            row = cursor.fetchone()
            count = cast("int", row[0]) if row else 0
            cursor.close()
        return count
    except Exception:
        return 0


def _recall_payload(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project recall hits into structured rows the model and the transcript
    back-reference both read.

    Each row is ``{id, content, score, kind, created_at}`` (+ ``location`` when an
    episode hit carries one). ``score`` is the relevance label (high/medium/low);
    the raw confidence stays internal. The structured shape replaces the old
    ``[id:X,relevance:Y] text`` prose: it is machine-parseable for the model AND
    is what ``Transcript._fetch_referenced_episodes`` keys its episode
    back-reference on (the ``id`` field), so the format is load-bearing.
    """
    rows: list[dict[str, object]] = []
    for hit in results:
        row: dict[str, object] = {
            "id": hit.get("id", ""),
            "content": hit.get("text", ""),
            "score": hit.get("relevance", "low"),
            "kind": hit.get("kind", "") or "",
            "created_at": hit.get("created_at"),
        }
        location = hit.get("location")
        if location:
            row["location"] = location
        rows.append(row)
    return rows


_LOCATION_SEARCH_CONFIDENCE = 0.9
_LOCATION_SEARCH_RELEVANCE = "high"


def _search_episodes_by_location(
    location: str, limit: int
) -> tuple[list[dict[str, object]], str]:
    """Search episodes whose location_name contains the given text.

    Cross-channel by design (): a location recall surfaces episodes from
    every episode-producing channel, mirroring the channel-agnostic semantic
    recall path. Also resolves saved place labels (e.g. 'home') via data_graph
    kind='place' to pick up alternate location_name strings stored at save time.
    """
    try:
        from services.data_graph_service import KIND_PLACE, get_data_graph_service
        from services.database_service import get_shared_db_service

        # Build the list of strings to LIKE-match against location_name.
        # Start with the raw input and add any resolved name from saved places.
        location_names = [location]
        try:
            dgs = get_data_graph_service()
            places = cast("list[dict[str, object]]", [p for p in dgs.fetch(kinds=[KIND_PLACE]) if p is not None])
            for place in places:
                raw_value = cast("str | None", place.get("value") or "{}")
                val = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
                if isinstance(val, dict) and cast("str", place.get("key", "")).lower() == location.lower():
                    place_name = cast("str | None", cast("dict[str, object]", val).get("name"))
                    if place_name and place_name.lower() != location.lower():
                        location_names.append(place_name)
        except Exception as _resolve_exc:
            logger.debug("%s Place label resolution failed: %s", LOG_PREFIX, _resolve_exc)

        db = get_shared_db_service()
        like_clauses = " OR ".join(["e.location_name LIKE ?"] * len(location_names))
        like_params = [f"%{name}%" for name in location_names]

        with db.connection() as conn:
            sql = (
                "SELECT e.id, e.gist, e.location_name, e.created_at "
                "FROM episodes e "
                f"WHERE e.deleted_at IS NULL AND ({like_clauses}) "
                "AND e.location_name IS NOT NULL "
                "ORDER BY e.created_at DESC LIMIT ?"
            )
            db_params = like_params + [limit]
            rows = conn.execute(sql, db_params).fetchall()

        if not rows:
            return [], "0 matches"

        hits = []
        for row in rows:
            ep_id, gist, loc_name, created_at = row
            hits.append({
                "id": str(ep_id),
                # Full gist verbatim — NO truncation ().
                "text": gist or "",
                "relevance": _LOCATION_SEARCH_RELEVANCE,
                "confidence": _LOCATION_SEARCH_CONFIDENCE,
                "location": loc_name,
                "kind": "episode",
                "created_at": created_at,
            })

        return hits, f"{len(hits)} matches"

    except Exception as exc:
        logger.warning("%s Location episode search failed: %s", LOG_PREFIX, exc)
        return [], f"error: {exc}"


def _store_fok_signal(channel: str, partial_match_count: int) -> None:
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        store.setex(f"fok:{channel}", 300, str(partial_match_count))
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to store FOK signal: {e}")
