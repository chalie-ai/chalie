"""Episodic Service — episode storage, retrieval, extraction, and clustering.

Thin orchestration over :class:`models.episode.Episode` (the sole home of the
``episodes`` / ``episodes_fts`` / ``episodes_vec`` SQL). The service owns the
multi-write cycles the model can't: store/update (embedding + transaction),
hybrid FTS+vector ``retrieve``, the turn-end extraction pipeline, and the
hierarchy roll-up clustering.

Retrieval is a pure read: it never mutates the episodes it returns. Relevance
(``last_relevant_at``, which anchors both the recency rerank term and the decay
engine) advances only on write-relevant events — episode creation and
consolidation — never on access.
"""

from __future__ import annotations

import json
import logging
import math
import struct
import threading
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, cast

import numpy as np

from models.episode import Episode
from models.tool_call import ToolCall
from services.database import Database
from services.embedding_utils import pack_embedding
from services.episodic_constants import (
    EXTRACTION_THRESHOLD,
    EXTRACTION_WINDOW,
    MAX_EPISODE_LEVEL,
    SALIENCE_NOVELTY_WEIGHT,
    SALIENCE_OPEN_LOOP_WEIGHT,
    SEED_CLUSTER_MAX_MATCHES,
    SEED_CLUSTER_MIN_SIZE,
    SEED_CLUSTER_RADIUS,
)
from services.time_utils import PARSE_SENTINEL, parse_utc, utc_now
from configs.channels import (
    EpisodeEncoderConfig,
    SuperEpisodeConfig,
    _collect_transcript_ids,
    _safe_json_load_object,
)

if TYPE_CHECKING:
    from services.embedding_service import EmbeddingService
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EPISODIC]"

# ── Retrieval constants ────────────────────────────────────────────────────────

# Vector KNN depth pulled per query (collapsed-tree KNN k=50).
_VECTOR_KNN_K = 50

# Composite scores are scaled into a 0-100-ish band for the memory skill's
# confidence-label bucketing (composite_score / 100).
_COMPOSITE_DISPLAY_SCALE = 10.0

# Recency half-life: exp(-_RECENCY_DECAY_PER_HOUR × hours) ≈ 0.5 at ~14 days.
_RECENCY_DECAY_PER_HOUR = 0.002

# Recency fallback when the relevance anchor cannot be parsed (sentinel/legacy).
_RECENCY_FALLBACK = 0.5

# Salience normalisation denominator (salience is stored on a 0-10 scale).
_SALIENCE_SCALE = 10.0

# Relative score floor: a candidate must score at least this fraction of the
# top candidate's score to survive. Below it, it is DROPPED (never padded to k).
_RELATIVE_SCORE_FLOOR = 0.5


class EpisodicService:
    """Episode storage/CRUD, hybrid retrieval, and turn-end extraction."""

    # ── Storage / CRUD ───────────────────────────────────────────────

    def store_episode(self, episode_data: dict[str, object], *, embedding: object = None) -> str:
        """Required fields: gist, salience, channel. ``embedding`` kwarg
        supersedes ``episode_data['embedding']`` when both are supplied.
        Raises ``ValueError`` when a required field is missing. Returns the
        new episode UUID."""
        required_fields = ['gist', 'salience', 'channel']
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            # Kwarg takes precedence; fall back to the dict key for callers
            # that embedded the vector inside episode_data (old path).
            if embedding is None:
                embedding = episode_data.get('embedding')

            raw_level = episode_data.get('level', 0)
            if isinstance(raw_level, int):
                level = raw_level
            else:
                # A non-int level would silently misclassify hierarchy depth and
                # mis-gate the consolidation cascade; coerce to leaf and say so.
                logging.warning(
                    f"store_episode: non-int level {raw_level!r} coerced to 0 "
                    f"(channel={episode_data.get('channel')!r})"
                )
                level = 0

            with Database.transaction():
                # Creation is a write-relevant event: seed the relevance clock
                # (last_relevant_at) so absolute decay measures Δt from now.
                episode = Episode(
                    gist=episode_data['gist'],
                    salience=episode_data['salience'],
                    channel=episode_data['channel'],
                    # Hierarchy depth: 0=leaf, 1=super-episode, 2+=era digest.
                    level=level,
                    transcript_ids=json.dumps(episode_data.get('transcript_ids', [])),
                    transcript_id_start=episode_data.get('transcript_id_start'),
                    transcript_id_end=episode_data.get('transcript_id_end'),
                    consolidated_from=json.dumps(episode_data.get('consolidated_from', [])),
                    retrieval_weight=episode_data.get('retrieval_weight', 1.0),
                    location_lat=episode_data.get('location_lat'),
                    location_lon=episode_data.get('location_lon'),
                    location_name=episode_data.get('location_name'),
                    last_relevant_at=utc_now().isoformat(),
                )
                episode.save()
                episode_id = cast(str, episode.id)

                if embedding is not None:
                    self._store_embedding(episode_id, embedding)

                logging.info(f"Stored episode {episode_id} for channel '{episode_data['channel']}'")

        except Exception as e:
            logging.error(f"Failed to store episode: {e}")
            raise

        # Fire consolidation off-thread AFTER the transaction commits so the
        # seed row is durable before the daemon reads it.
        self._check_and_consolidate(episode_id, cast(str, episode_data['channel']), level, embedding)

        return episode_id

    @staticmethod
    def _store_embedding(episode_id: str, embedding: object) -> None:
        """Store an embedding blob in the ``episodes_vec`` virtual table."""
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return
            Episode.set_embedding(episode_id, blob)
        except Exception as e:
            logging.warning(f"Failed to store episode embedding: {e}")

    def update_episode(self, episode_id: str, fields: dict[str, object], embedding: object = None) -> None:
        """Empty ``fields`` + no ``embedding`` is a no-op."""
        if not fields and embedding is None:
            return

        # List columns (transcript_ids, consolidated_from) are stored as JSON
        # text — same encoding store_episode applies on INSERT.
        fields = {
            key: json.dumps(value) if isinstance(value, list) else value
            for key, value in fields.items()
        }

        with Database.transaction():
            if fields:
                Episode.update_fields(episode_id, fields)

            if embedding is not None:
                self._store_embedding(episode_id, embedding)

    def _check_and_consolidate(self, seed_id: str, channel: str, level: int, embedding: object) -> None:
        """Fire-on-creation gate: when ``embedding`` is present and the new
        episode would roll up to a level within ``MAX_EPISODE_LEVEL``, spawn a
        daemon to run seed-cluster search. Gated by the per-source profile
        (only ``extract_episodes`` channels consolidate); runs off-thread so the
        turn never blocks. Never raises."""
        if embedding is None:
            return
        if level + 1 > MAX_EPISODE_LEVEL:
            return
        try:
            # profile_for is inside the try so a malformed channel degrades to a
            # skipped (logged) consolidation instead of raising out of the caller
            # — which, on the recursive parent-write path, runs after the DB
            # transaction has already committed.
            from services.source_profiles import profile_for
            if not profile_for(channel).extract_episodes:
                return
            blob = pack_embedding(embedding)
            if blob is None:
                return
            threading.Thread(
                target=self._consolidate_seed, args=(seed_id, channel, level, blob),
                name=f"consolidate-seed-{channel}", daemon=True,
            ).start()
        except Exception as exc:  # noqa: BLE001 — consolidation must never break a turn
            logger.warning("%s _check_and_consolidate failed (channel=%s): %s",
                           LOG_PREFIX, channel, exc)

    def _consolidate_seed(self, seed_id: str, channel: str, level: int, blob: bytes) -> None:
        """Daemon body: find a cluster around the seed, then encode a parent
        super-episode at ``level + 1``. Mirrors the defensive style of
        ``_extract_window``; never raises."""
        try:
            # Serialize consolidation per channel: hold the lock across cluster
            # discovery AND the parent write so no other daemon can claim the
            # same children in the gap (the encode window spans seconds). This
            # restores the single-writer-per-channel invariant the batch path had
            # — one tick, one thread — which fire-on-creation daemons would
            # otherwise break, producing duplicate parents and dangling lineage.
            with _consolidation_lock(channel):
                cluster = find_seed_cluster(seed_id, channel, level, blob)
                if not cluster:
                    return
                from services.embedding_service import get_embedding_service
                emb_svc = get_embedding_service()
                try:
                    prior_embeddings = Episode.novelty_comparison_blobs(channel)
                except Exception:
                    prior_embeddings = []
                consolidate_cluster(channel, cluster, level + 1, emb_svc, self, prior_embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s _consolidate_seed failed (channel=%s): %s", LOG_PREFIX, channel, exc)

    def decay(self) -> int:
        """Off-turn episodic maintenance (the Decayable contract): tombstone
        stale fossils, recompute retrieval weights by absolute exponential
        decay, then hard-delete expired episodes — in that order, so a fossil
        tombstoned this pass collapses onto the fast τ before its deletion
        eligibility is checked. Each step delegates to the :class:`Episode`
        model (the sole home of the SQL); the model's decay methods run single
        autocommit statements, so the service groups them in one transaction
        (its contract). Returns the total rows maintained (tombstoned +
        reweighted + deleted)."""
        from services.source_profiles import janitor_protected_sql
        with Database.transaction():
            tombstoned = Episode.tombstone_fossils(janitor_protected_sql())
            reweighted = Episode.decay_weights()
            deleted = Episode.delete_expired()
        return tombstoned + reweighted + deleted

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        *,
        query_embedding: list[float] | None = None,
        channel: str | None = None,
        k: int = 10,
        return_telemetry: bool = False,
    ) -> list[Episode] | tuple[list[Episode], dict[str, object]]:
        """Hybrid FTS + vector retrieval over the collapsed episode tree.

        Both lanes pull their full candidate sets (no vector-distance ceiling);
        episodes at all levels (leaf / super / era) compete in one pool, are
        scored by the normalised composite rerank, and the weak tail is dropped
        by the relative score floor (never padded back up to *k*). When
        ``return_telemetry`` is True, returns ``(episodes, telemetry)``.
        """
        telemetry: dict[str, object] = {
            'episode_count': 0,
            'vector_candidates': 0,
            'fts_candidates': 0,
            'floor_cut_count': 0,
            'final_rrf_count': 0,
            'top_distances': [],
        }

        try:
            if query_embedding is None and query_text:
                query_embedding = _generate_embedding(query_text)

            telemetry['episode_count'] = _count_episodes(channel)

            fts_hits = _fts_search(query_text or '', channel, k=k * 2)
            vector_hits = _vector_search(query_embedding, channel, k=_VECTOR_KNN_K)
            telemetry['vector_candidates'] = len(vector_hits)
            telemetry['fts_candidates'] = len(fts_hits)

            candidates = _dedup_by_id(fts_hits + vector_hits)
            if not candidates:
                if return_telemetry:
                    return [], telemetry
                return []

            scored = _rerank_composite(candidates)
            ranked = scored[:k]
            # Candidates dropped by the relative score floor (before top-k trunc).
            telemetry['floor_cut_count'] = len(candidates) - len(scored)
            telemetry['final_rrf_count'] = len(ranked)
            telemetry['top_distances'] = _collect_top_distances(ranked)

            if return_telemetry:
                return ranked, telemetry
            return ranked

        except Exception as exc:
            logger.error(f"[RETRIEVAL] retrieve failed: {exc}")
            if return_telemetry:
                return [], telemetry
            return []

    # ── Window extraction (turn-end episode encoding) ─────────────────

    def check_and_store(self, config: ProcessorConfig) -> None:
        """Turn-end entry point: when ``config.channel`` has accumulated
        ``EXTRACTION_THRESHOLD`` transcript rows past its episode watermark, spawn
        a daemon to encode the latest window into episodes. Gated by the per-source
        profile (only ``extract_episodes`` channels produce episodes); encoding runs
        off-thread so the turn never blocks. Never raises."""
        from services.source_profiles import profile_for
        if not profile_for(config.channel).extract_episodes:
            return
        try:
            from models.transcript import Transcript
            untriggered = (
                Transcript.filter("channel", config.channel)
                .filter("id", Episode.watermark(config.channel), ">")
                .count()
            )
            if untriggered >= EXTRACTION_THRESHOLD:
                threading.Thread(
                    target=self._extract_window, args=(config.channel,), daemon=True
                ).start()
        except Exception as exc:  # noqa: BLE001 — extraction must never break a turn
            logger.warning("%s check_and_store failed (channel=%s): %s",
                           LOG_PREFIX, config.channel, exc)

    def _extract_window(self, channel: str) -> None:
        """Encode the channel's latest ``EXTRACTION_WINDOW`` transcript rows into
        episodes via the episode-encoder channel, then store each snapshot. Runs on
        a daemon thread (see :meth:`check_and_store`); never raises."""
        try:
            from controllers.message_processor import MessageProcessor
            from models.transcript import Transcript

            # ── 1. Fetch the latest window ───────────────────────────────────
            rows = (
                Transcript.filter("channel", channel)
                .order_by("id DESC")
                .limit(EXTRACTION_WINDOW)
                .get()
            )
            if not rows:
                return

            entries: list[dict[str, object]] = []
            for row in reversed(rows):
                data = row.to_dict()
                entries.append({
                    'id': data['id'], 'role': data['role'], 'content': data['content'],
                    'tool_name': data['tool_name'], 'created_at': data['created_at'],
                    'location_lat': data['location_lat'], 'location_lon': data['location_lon'],
                    'location_name': data['location_name'], 'channel': channel,
                })

            # ── 2. Encode via the flat episode-encoder channel ───────────────
            # The window + referenced-episode text ride the turn on `metadata`
            # (PromptService._episode_encoder_prompt reads them, §2.4); a
            # synchronous text turn, so `.process(...).result()`.
            emp = MessageProcessor.process(
                EpisodeEncoderConfig(),
                metadata={
                    "window": self._format_window_entries(entries),
                    "referenced": self._format_episodes_for_prompt(
                        self._fetch_referenced_episodes(entries)),
                },
            )
            snapshots = cast("list[dict[str, object]]", self._safe_json_load(emp.result()))
            if not snapshots:
                return

            # ── 3. Store each snapshot ───────────────────────────────────────
            valid_ids = {e['id'] for e in entries}
            dominant_location = self._aggregate_dominant_location(entries)
            prior_embeddings = Episode.novelty_comparison_blobs(channel)
            for ep in snapshots:
                try:
                    self._store_snapshot(ep, channel, valid_ids, dominant_location, prior_embeddings)
                except Exception as ep_err:  # noqa: BLE001 — one bad snapshot must not abort the rest
                    logger.warning("%s episode store failed: %s", LOG_PREFIX, ep_err)

        except Exception as exc:  # noqa: BLE001
            logger.warning("%s _extract_window failed (channel=%s): %s", LOG_PREFIX, channel, exc)

    def _store_snapshot(self, ep: dict[str, object], channel: str, valid_ids: set[object],
                        dominant_location: dict[str, object], prior_embeddings: list[bytes]) -> None:
        """Persist one encoder snapshot — soft-delete-only, update, or fresh store.
        Filters ``transcript_ids`` to the window, stamps location + salience, then
        appends the new embedding to ``prior_embeddings`` so later snapshots in the
        same run compare against it (not just the stale pre-run set)."""
        from services.embedding_service import get_embedding_service

        if self._is_delete_only(ep):
            delete_id = ep.get('delete_id')
            if delete_id:
                Episode.soft_delete(cast("str", delete_id))
            return

        raw_ids = cast("list[object]", ep.get('transcript_ids') or [])
        ep['transcript_ids'] = [i for i in raw_ids if i in valid_ids]
        if not ep['transcript_ids']:
            return

        ep['transcript_id_start'] = min(cast("list[int]", ep['transcript_ids']))
        ep['transcript_id_end'] = max(cast("list[int]", ep['transcript_ids']))
        ep['channel'] = channel

        if dominant_location.get('lat') is not None:
            ep['location_lat'] = dominant_location['lat']
            ep['location_lon'] = dominant_location['lon']
            ep['location_name'] = dominant_location.get('name')

        gist = cast("str", ep.get('gist', '') or '')
        embedding = get_embedding_service().generate_embedding(gist) if gist else None

        novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
        ep['salience'] = compute_salience(
            has_open_loop=bool(ep.get('has_open_loop', False)),
            novelty=novelty,
        )

        ep.pop('has_open_loop', None)
        update_id = ep.pop('update_id', None)
        ep.pop('delete_id', None)  # defensive — should be None here

        if update_id:
            self.update_episode(cast("str", update_id), ep, embedding=embedding)
        else:
            self.store_episode(ep, embedding=embedding)

        if embedding is not None:
            blob = pack_embedding(embedding)
            if blob is not None:
                prior_embeddings.append(blob)

    # ── Window extraction helpers ─────────────────────────────────────

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _episode_ids_from_envelope(text: str) -> set[str]:
        """Extract episode IDs from one rendered memory recall envelope.

        Only rows with ``kind == "episode"`` are collected — data-graph rows are
        keyed by data-graph key (not an episode id) and are intentionally skipped.
        A malformed / non-recall envelope yields an empty set silently.
        """
        if not text or "[end:memory]" not in text:
            return set()
        nl = text.find("]\n")
        end = text.find("\n[end:memory]")
        if nl == -1 or end == -1 or end <= nl:
            return set()
        body = text[nl + 2:end]
        try:
            parsed = json.loads(body)
        except (ValueError, TypeError):
            return set()
        rows = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(rows, list):
            return set()
        return {
            eid
            for row in rows
            if isinstance(row, dict) and row.get("kind") == "episode"
            if (eid := str(row.get("id", "")).strip())
        }

    @staticmethod
    def _parse_episode_ids_from_results(result_texts: list[str]) -> set[str]:
        """Extract episode IDs from rendered memory recall envelopes."""
        episode_ids: set[str] = set()
        for text in result_texts:
            episode_ids |= EpisodicService._episode_ids_from_envelope(text)
        return episode_ids

    def _fetch_referenced_episodes(self, entries: list[dict[str, object]]) -> list[Episode]:
        """Episodes the window already cited via the memory tool — fed back to the
        encoder so it can update/dedup rather than re-create. ``tool_name='memory'``
        covers both auto-seed and LLM-invoked recall."""
        t_ids = [cast("int", e['id']) for e in entries if e.get('id')]
        if not t_ids:
            return []

        try:
            result_texts = ToolCall.memory_recall_results(t_ids)
        except Exception as exc:
            logger.warning("%s _fetch_referenced_episodes query failed: %s", LOG_PREFIX, exc)
            return []

        episode_ids = self._parse_episode_ids_from_results([text or '' for text in result_texts])
        if not episode_ids:
            return []

        try:
            episodes = []
            for eid in episode_ids:
                ep = Episode.by_id(eid)
                if ep:
                    episodes.append(ep)
            return episodes
        except Exception as exc:
            logger.warning("%s _fetch_referenced_episodes fetch failed: %s", LOG_PREFIX, exc)
            return []

    @staticmethod
    def _format_episodes_for_prompt(episodes: list[Episode]) -> str:
        if not episodes:
            return ''
        lines = []
        for ep in episodes:
            lines.append(f"id: {ep.id} | gist: {ep.gist} | created: {ep.created_at}")
        return "\n".join(lines)

    @staticmethod
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

    @staticmethod
    def _safe_json_load(text: str) -> list[object]:
        if not text:
            return []
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = EpisodicService._strip_code_fence(text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            logger.warning("%s EpisodeEncoder returned non-list JSON", LOG_PREFIX)
            return []
        except ValueError:
            logger.warning("%s EpisodeEncoder returned unparseable JSON", LOG_PREFIX)
            return []

    @staticmethod
    def _is_delete_only(ep: dict[str, object]) -> bool:
        if not ep.get('delete_id'):
            return False
        # All other meaningful fields must be absent or null/empty
        meaningful = ('gist', 'transcript_ids', 'update_id')
        return not any(ep.get(f) for f in meaningful)


# ── Retrieval helpers (module-level) ───────────────────────────────────────────


def _generate_embedding(text: str) -> list[float] | None:
    """Resolve an embedding via the shared embedding service."""
    try:
        from services.embedding_service import get_embedding_service
        return get_embedding_service().generate_embedding(text)
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _generate_embedding failed: {exc}")
        return None


def _pack_embedding(embedding: object) -> bytes | None:
    """Pack a list of floats into a sqlite-vec binary blob."""
    if embedding is None:
        return None
    try:
        return pack_embedding(embedding)
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] _pack_embedding failed: {exc}")
        return None


def _count_episodes(channel: str | None = None) -> int:
    """Return the live episode count, optionally scoped to a channel."""
    try:
        query = Episode.live()
        if channel is not None:
            query = query.filter("channel", channel)
        return query.count()
    except Exception:
        return 0


def _fts_search(query_text: str, channel: str | None, k: int) -> list[Episode]:
    """FTS lane over the gist column. The model sanitises + runs the query and
    raises on DB failure — the warn-guard lives here."""
    try:
        return Episode.fts_search(query_text, channel, k=k)
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] FTS search failed: {exc}")
        return []


def _vector_search(query_embedding: object, channel: str | None, k: int) -> list[Episode]:
    """Cosine KNN lane via sqlite-vec.

    No distance ceiling is applied: the relative score floor in
    ``_rerank_composite`` decides what survives (kills the radius-0 bug where a
    hard ceiling muted the entire vector lane).
    """
    blob = _pack_embedding(query_embedding)
    if blob is None:
        return []
    try:
        return Episode.nearest(blob, channel, k=k)
    except Exception as exc:
        logger.warning(f"[RETRIEVAL] Vector search failed: {exc}")
        return []


def _dedup_by_id(hits: list[Episode]) -> list[Episode]:
    """Deduplicate hits preserving first-seen order. Merges text_rank and
    vector_distance from duplicate entries onto the first occurrence."""
    seen: dict[str, Episode] = {}
    ordered: list[Episode] = []
    for hit in hits:
        eid = str(hit.id)
        if eid not in seen:
            seen[eid] = hit
            ordered.append(hit)
        else:
            existing = seen[eid]
            if existing.text_rank is None and hit.text_rank is not None:
                existing.text_rank = hit.text_rank
            if existing.vector_distance is None and hit.vector_distance is not None:
                existing.vector_distance = hit.vector_distance
    return ordered


def _vector_sim(ep: Episode) -> float:
    """Raw vector-lane similarity (1 - distance), 0 when the lane did not hit."""
    vd = ep.vector_distance
    if vd is None:
        return 0.0
    return max(0.0, 1.0 - float(vd))


def _fts_strength(ep: Episode) -> float:
    """Raw FTS-lane strength from the (negative) FTS5 rank, 0 when no hit.

    FTS5 ``rank`` is negative and more negative = better; the magnitude is the
    raw strength fed into per-lane min-max normalisation.
    """
    tr = ep.text_rank
    if tr is None:
        return 0.0
    return abs(float(tr))


def _recency(ep: Episode, now: datetime) -> float:
    """Exponential recency term anchored on ``last_relevant_at`` (half-life 14d).

    The clock is the relevance anchor (last write-relevant event), falling back
    to creation time — reads never advance it. ``parse_utc`` never raises; a
    sentinel/legacy value yields a flat fallback rather than a crash.
    """
    anchor = ep.last_relevant_at or ep.created_at
    if not anchor:
        return _RECENCY_FALLBACK
    ref_time = parse_utc(anchor)
    if ref_time == PARSE_SENTINEL:
        logger.warning(
            "[RETRIEVAL] unparseable relevance anchor for id=%s: %r",
            ep.id, anchor,
        )
        return _RECENCY_FALLBACK
    hours = (now - ref_time).total_seconds() / 3600.0
    return math.exp(-_RECENCY_DECAY_PER_HOUR * hours)


def _importance(ep: Episode) -> float:
    """Importance term: salience/10 × retrieval_weight."""
    salience_norm = float(ep.salience or 5) / _SALIENCE_SCALE
    retrieval_w = float(ep.retrieval_weight or 1.0)
    return salience_norm * retrieval_w


def _min_max_normalise(values: list[float]) -> list[float]:
    """Min-max normalise a lane's raw signals into [0, 1].

    A degenerate spread (all equal, including all-zero) maps every entry to 0.0
    so a lane that fired uniformly adds no discrimination rather than inflating
    every candidate to 1.0.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def _rerank_composite(episodes: list[Episode]) -> list[Episode]:
    """Collapsed-tree composite rerank with a relative score floor.

    Episodes at all hierarchy levels (leaf, super, era) compete in one pool.
    The two retrieval lanes are min-max normalised WITHIN the pool, then

        score = relevance + recency + importance

    where ``relevance`` is the stronger of the two normalised lane signals,
    ``recency`` is an exp half-life ≈ 14d term on ``last_relevant_at``, and
    ``importance`` is ``salience/10 × retrieval_weight``.

    A RELATIVE floor then drops every candidate scoring below
    ``_RELATIVE_SCORE_FLOOR × top_score`` — survivors only, never padded to *k*.
    """
    if not episodes:
        return []

    now = utc_now()

    vector_norm = _min_max_normalise([_vector_sim(ep) for ep in episodes])
    fts_norm = _min_max_normalise([_fts_strength(ep) for ep in episodes])

    for ep, v_norm, f_norm in zip(episodes, vector_norm, fts_norm):
        relevance = max(v_norm, f_norm)
        score = relevance + _recency(ep, now) + _importance(ep)
        # Scale into the 0-100-ish space the memory skill buckets confidence in.
        ep.composite_score = score * _COMPOSITE_DISPLAY_SCALE

    episodes.sort(key=lambda e: e.composite_score, reverse=True)

    top_score = episodes[0].composite_score
    if top_score <= 0.0:
        return episodes
    floor = top_score * _RELATIVE_SCORE_FLOOR
    return [ep for ep in episodes if ep.composite_score >= floor]


def _collect_top_distances(ranked: list[Episode]) -> list[float]:
    """Return rounded vector distances for the top-5 ranked episodes."""
    top_dists: list[float] = []
    for ep in ranked[:5]:
        vd = ep.vector_distance
        if vd is not None:
            try:
                top_dists.append(round(float(vd), 4))
            except (TypeError, ValueError):
                pass
    return top_dists


# ── Salience / novelty / clustering (module-level) ────────────────────────────


def compute_salience(has_open_loop: bool, novelty: float) -> int:
    """raw = SALIENCE_NOVELTY_WEIGHT × novelty + SALIENCE_OPEN_LOOP_WEIGHT × open_boost,
    then clamped to an integer 1..10."""
    open_boost = 1.0 if has_open_loop else 0.0
    raw = SALIENCE_NOVELTY_WEIGHT * float(novelty) + SALIENCE_OPEN_LOOP_WEIGHT * open_boost
    return max(1, min(10, int(round(raw * 10))))


def _unpack_blob(blob: bytes) -> list[float]:
    """Unpack a sqlite-vec binary blob into a list of floats."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f'{n}f', blob))


def compute_novelty(new_embedding: object, prior_embeddings: list[bytes]) -> float:
    """novelty = 1.0 - max(cosine_sim(new, prior) for prior in
    prior_embeddings). Clamped to [0.0, 1.0]. Returns 1.0 (fully
    novel) if prior_embeddings is empty."""
    if not prior_embeddings:
        return 1.0

    try:
        if isinstance(new_embedding, bytes):
            new_vec = np.array(_unpack_blob(new_embedding), dtype=np.float32)
        else:
            new_vec = np.array(new_embedding, dtype=np.float32)

        # L2-normalize (embeddings from EmbeddingService are already normalized,
        # but defend against callers who pass raw vectors).
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm

        max_sim = 0.0
        for blob in prior_embeddings:
            try:
                prior_vec = np.array(_unpack_blob(blob), dtype=np.float32)
                if prior_vec.shape != new_vec.shape:
                    continue
                sim = float(np.dot(new_vec, prior_vec))
                if sim > max_sim:
                    max_sim = sim
            except Exception:
                continue
        return max(0.0, min(1.0, 1.0 - max_sim))

    except Exception as exc:
        logging.warning(f"[NOVELTY] compute_novelty failed: {exc}")
        return 1.0


# Per-channel consolidation locks. Fire-on-creation consolidation runs in
# daemon threads; two near-simultaneous same-channel creations would otherwise
# race — both discovering the same apex neighbours and both claiming them across
# the multi-second encode window, leaving duplicate parents and children whose
# back-pointers disagree with their parent's lineage. Serializing per channel
# keeps each memory stream's roll-up single-writer while letting distinct
# channels consolidate concurrently. Created lazily under a guard because
# threads may request the same channel's lock at once.
_CONSOLIDATION_LOCKS: dict[str, threading.Lock] = {}
_CONSOLIDATION_LOCKS_GUARD = threading.Lock()


def _consolidation_lock(channel: str) -> threading.Lock:
    """Return the process-wide lock serializing consolidation for ``channel``."""
    with _CONSOLIDATION_LOCKS_GUARD:
        lock = _CONSOLIDATION_LOCKS.get(channel)
        if lock is None:
            lock = threading.Lock()
            _CONSOLIDATION_LOCKS[channel] = lock
        return lock


def find_seed_cluster(
    seed_id: str, channel: str, level: int, embedding: bytes
) -> list[str] | None:
    """Seed-on-creation candidate discovery for super-episode consolidation.

    One new episode seeds a local KNN neighbourhood instead of reclustering the
    whole apex pool (the old batch HDBSCAN path). ``Episode.nearest`` applies
    the qualifying filters (live, same-channel, same-level, apex-only) inside
    the KNN itself, so the hits are already the nearest qualifying neighbours;
    here we only drop the seed itself and anything beyond the cosine-distance
    radius. If the resulting cluster (seed + qualifying neighbours) meets the
    minimum-size floor, return the sorted id list; otherwise return ``None`` —
    the seed stays a lone apex.

    ``vector_distance`` is cosine distance (0.0 = identical, larger = less
    similar). Neighbours beyond :data:`SEED_CLUSTER_RADIUS` are excluded.
    DB failures log a warning and return ``None`` — never raise.
    """
    try:
        # The seed must still be apex. A concurrent same-channel consolidation
        # may have rolled it up (as a neighbour of its own cluster) between this
        # episode's creation and now; a non-apex seed must not anchor a fresh
        # cluster, or its back-pointer would be overwritten and the first
        # parent's lineage left dangling. Neighbours are apex-filtered inside
        # the KNN; the seed is the one member added unconditionally, so it is
        # checked here. Under the per-channel consolidation lock this check and
        # the eventual write are atomic, so the seed cannot be rolled up between.
        seed = Episode.by_id(seed_id)
        if seed is None or seed.consolidated_into is not None:
            return None
        # k = cap + 1: the seed is itself a qualifying row (distance 0) and is
        # dropped below, leaving at most SEED_CLUSTER_MAX_MATCHES neighbours.
        hits = Episode.nearest(
            embedding, channel, SEED_CLUSTER_MAX_MATCHES + 1,
            level=level, apex_only=True,
        )
    except Exception as exc:
        logger.warning(
            "[SEED_CLUSTER] seed/KNN lookup failed for seed=%s channel=%s: %s",
            seed_id, channel, exc,
        )
        return None

    neighbours: list[str] = []
    for hit in hits:
        # Exclude the seed itself (it will always be nearest to itself).
        if hit.id == seed_id:
            continue
        # Must carry a vector_distance overlay — sqlite-vec always sets it on
        # a hit, but guard against a missing value (defensive).
        vd = hit.vector_distance
        if vd is None:
            continue
        # Cosine-distance cutoff: smaller = more similar; keep <= radius.
        if float(vd) > SEED_CLUSTER_RADIUS:
            continue
        neighbours.append(str(hit.id))
        if len(neighbours) >= SEED_CLUSTER_MAX_MATCHES:
            break

    cluster = [seed_id] + neighbours
    if len(cluster) < SEED_CLUSTER_MIN_SIZE:
        return None
    return sorted(cluster)


def consolidate_cluster(
    channel: str,
    cluster_ids: list[str],
    level: int,
    emb_svc: "EmbeddingService",
    episodic_svc: "EpisodicService",
    prior_embeddings: list[bytes],
) -> bool:
    """Encode + store one parent episode for a cluster. Returns True on write."""
    from services.episodic_constants import SEED_CLUSTER_MIN_SIZE
    from controllers.message_processor import MessageProcessor

    try:
        sources = [
            ep.to_dict() for ep in (
                Episode.by_id(eid) for eid in cluster_ids
            )
            if ep
        ]
        if len(sources) < SEED_CLUSTER_MIN_SIZE:
            return False

        all_t_ids = _collect_transcript_ids(cast(list[object], sources))

        config = SuperEpisodeConfig(channel, cast(list[object], sources))
        response = MessageProcessor.process(config).result()

        if not response:
            logger.warning(
                f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response "
                f"for cluster {cluster_ids}"
            )
            return False

        super_ep = _safe_json_load_object(response)
        if not super_ep or not super_ep.get("gist"):
            logger.warning(
                f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty "
                f"gist for cluster {cluster_ids}"
            )
            return False

        super_ep["channel"] = channel
        super_ep["level"] = level
        unique_t_ids = sorted(all_t_ids)
        super_ep["transcript_ids"] = unique_t_ids
        super_ep["transcript_id_start"] = (
            min(unique_t_ids) if unique_t_ids else None
        )
        super_ep["transcript_id_end"] = (
            max(unique_t_ids) if unique_t_ids else None
        )
        super_ep["consolidated_from"] = [ep["id"] for ep in sources]

        gist = cast(str, super_ep["gist"])
        embedding = emb_svc.generate_embedding(gist)
        novelty = (
            compute_novelty(embedding, prior_embeddings) if embedding else 1.0
        )
        super_ep["salience"] = compute_salience(
            has_open_loop=bool(super_ep.get("has_open_loop", False)),
            novelty=novelty,
        )
        super_ep.pop("has_open_loop", None)

        new_id = episodic_svc.store_episode(super_ep, embedding=embedding)
        for src_id in cluster_ids:
            Episode.set_consolidated_into(src_id, new_id)

        logger.info(
            f"{LOG_PREFIX} level-{level} episode {new_id} created from cluster "
            f"{cluster_ids} (channel={channel})"
        )
        return True

    except Exception as exc:
        logger.warning(
            f"{LOG_PREFIX} level-{level} consolidation failed for cluster "
            f"{cluster_ids} (channel={channel}): {exc}"
        )
        return False
