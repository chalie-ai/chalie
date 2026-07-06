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

# Hierarchy roll-up clustering stack. Bidirectional dependency: declared in
# pyproject.toml ("scikit-learn"/"umap-learn"/"scipy"/"numba"/"llvmlite") and
# consumed only by ``find_super_candidates`` below. UMAP is the only reducer that
# yields usable density clusters at our embedding scale; sklearn ships HDBSCAN.
import umap
from sklearn.cluster import HDBSCAN

from models.episode import Episode
from models.tool_call import ToolCall
from services.database import Database
from services.embedding_utils import pack_embedding
from services.episodic_constants import (
    EXTRACTION_THRESHOLD,
    EXTRACTION_WINDOW,
    SALIENCE_NOVELTY_WEIGHT,
    SALIENCE_OPEN_LOOP_WEIGHT,
)
from services.time_utils import PARSE_SENTINEL, parse_utc, utc_now

if TYPE_CHECKING:
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

            with Database.transaction():
                # Creation is a write-relevant event: seed the relevance clock
                # (last_relevant_at) so absolute decay measures Δt from now.
                episode = Episode(
                    gist=episode_data['gist'],
                    salience=episode_data['salience'],
                    channel=episode_data['channel'],
                    # Hierarchy depth: 0=leaf, 1=super-episode, 2+=era digest.
                    level=episode_data.get('level', 0),
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

                return episode_id

        except Exception as e:
            logging.error(f"Failed to store episode: {e}")
            raise

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
                Transcript.filter("channel = ?", config.channel)
                .filter("id > ?", Episode.watermark(config.channel))
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
            from configs.channels import EpisodeEncoderConfig
            from controllers.message_processor import MessageProcessor
            from models.transcript import Transcript

            # ── 1. Fetch the latest window ───────────────────────────────────
            rows = (
                Transcript.filter("channel = ?", channel)
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
            query = query.filter("channel = ?", channel)
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


def cluster_apex_embeddings(ep_ids: list[str], ep_embs: list[bytes]) -> list[list[str]]:
    """L2-normalise rows → UMAP reduce to a low-dimensional space →
    sklearn HDBSCAN. Episodes sharing an HDBSCAN label form a group;
    the noise label (-1) is dropped so genuine outliers stay leaf apexes
    (never force-assigned). Only groups of at least
    ``HDBSCAN_MIN_CLUSTER_SIZE`` survive.

    Shared by the leaf round (level-0 → level-1) and the era round
    (level-1 → level-2) — both feed the same matrix path. The native
    blob dimension is read as-is (768 prod / 256 test); UMAP reduces
    either to ``UMAP_N_COMPONENTS``.

    Returns ID-lists, one per surviving cluster, deterministic (sorted
    IDs within each list; outer list sorted by first ID). Empty when
    the input is too small to cluster.
    """
    from services.episodic_constants import (
        HDBSCAN_MIN_CLUSTER_SIZE,
        UMAP_DISCONNECTION_DISTANCE,
        UMAP_MIN_DIST,
        UMAP_N_COMPONENTS,
        UMAP_N_NEIGHBORS,
        UMAP_RANDOM_SEED,
    )

    n = len(ep_ids)
    # UMAP needs n_neighbors < n_samples; below the cluster floor nothing can form.
    if n < HDBSCAN_MIN_CLUSTER_SIZE or n < 2:
        return []

    matrix = np.vstack([
        np.asarray(_unpack_blob(blob), dtype=np.float32) for blob in ep_embs
    ])
    # L2-normalise rows so UMAP's cosine metric sees unit vectors; guard the rare
    # zero-norm row (division would yield NaN and poison the reducer).
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix /= norms

    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        metric="cosine",
        n_neighbors=min(UMAP_N_NEIGHBORS, n - 1),
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_SEED,
        # Detach genuine off-topic isolates instead of force-embedding them into
        # the nearest dense region; they then surface as HDBSCAN noise and stay
        # leaf apexes rather than polluting a topically-pure roll-up.
        disconnection_distance=UMAP_DISCONNECTION_DISTANCE,
    )
    reduced = reducer.fit_transform(matrix)

    labels = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        metric="euclidean",
        copy=True,
    ).fit_predict(reduced)

    groups: dict[int, list[str]] = {}
    for label, ep_id in zip(labels, ep_ids):
        # Any negative label is noise: -1 is HDBSCAN's own outlier flag and -3 is
        # a UMAP-disconnected isolate. Both stay leaf apexes — never force-grouped.
        if label < 0:
            continue
        groups.setdefault(int(label), []).append(ep_id)

    clusters = sorted(
        (sorted(ids) for ids in groups.values() if len(ids) >= HDBSCAN_MIN_CLUSTER_SIZE),
        key=lambda c: c[0],
    )
    return clusters


def find_super_candidates(channel: str) -> list[list[str]]:
    """Count trigger: NO-OP (return ``[]``) until the channel holds at
    least ``APEX_COUNT_TRIGGER`` leaf apexes — a count trigger always
    eventually fires, unlike the old similarity gate that never did.
    At the trigger the leaves are density-clustered via UMAP→HDBSCAN;
    HDBSCAN noise is dropped so outliers stay leaf apexes.

    Returns a list of ID-lists (strings), one per surviving cluster.
    Empty below the count trigger or when no cluster forms.
    """
    from services.episodic_constants import APEX_COUNT_TRIGGER

    ep_ids, ep_embs = Episode.apex_embeddings(channel, level=0)

    if len(ep_ids) < APEX_COUNT_TRIGGER:
        return []

    clusters = cluster_apex_embeddings(ep_ids, ep_embs)
    if clusters:
        logging.info(
            f"[SUPER_CLUSTER] {len(clusters)} cluster(s) found "
            f"(sizes={[len(c) for c in clusters]}, channel={channel})"
        )
    return clusters


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
