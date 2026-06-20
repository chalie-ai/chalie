# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Episodic Service — episode storage + CRUD.

Retrieval lives in ``episodic_retrieval_service``. Super-episode
clustering helpers (``find_super_candidates``, ``cluster_apex_embeddings``,
``compute_novelty``, and their DB helpers) remain module-level in this
file because they operate directly on the ``episodes`` + ``episodes_vec``
tables and are used by ``transcript_service`` during post-extraction
consolidation.
"""

import json
import logging
import struct
import uuid
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from services.database_service import DatabaseService
from services.embedding_utils import pack_embedding
from services.time_utils import utc_now

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    class _Cursor(Protocol):
        def execute(self, sql: str, params: Sequence[object] = ...) -> object: ...
        def fetchone(self) -> object: ...
        def fetchall(self) -> list[object]: ...
        def close(self) -> None: ...

    class _Conn(Protocol):
        def cursor(self) -> _Cursor: ...
        def execute(self, sql: str, params: Sequence[object] = ...) -> object: ...

# Hierarchy roll-up clustering stack. Bidirectional dependency: declared in
# pyproject.toml ("scikit-learn"/"umap-learn"/"scipy"/"numba"/"llvmlite") and
# consumed only by ``find_super_candidates`` below. UMAP is the only reducer that
# yields usable density clusters at our embedding scale; sklearn ships HDBSCAN.
import umap
from sklearn.cluster import HDBSCAN


class EpisodicService:
    """Manages episode storage and CRUD (no retrieval — see
    ``episodic_retrieval_service.retrieve``)."""

    def __init__(self, database_service: DatabaseService, config: Optional[dict[str, object]] = None):
        self.db_service = database_service
        self.config = config or {}

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
            episode_id = str(uuid.uuid4())
            # Kwarg takes precedence; fall back to the dict key for callers
            # that embedded the vector inside episode_data (old path).
            if embedding is None:
                embedding = episode_data.get('embedding')

            transcript_id_start = episode_data.get('transcript_id_start')
            transcript_id_end = episode_data.get('transcript_id_end')

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Creation is a write-relevant event: seed the relevance clock
                # (last_relevant_at) so absolute decay measures Δt from now.
                now_iso = utc_now().isoformat()
                cursor.execute("""
                    INSERT INTO episodes (
                        id, gist, salience, channel, level,
                        transcript_ids, transcript_id_start, transcript_id_end,
                        emotional_valence, emotional_arousal,
                        consolidated_from, storage_strength, retrieval_weight,
                        location_lat, location_lon, location_name,
                        last_relevant_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    episode_data['gist'],
                    episode_data['salience'],
                    episode_data['channel'],
                    # Hierarchy depth: 0=leaf, 1=super-episode, 2+=era digest.
                    # Roll-up writes stamp this so per-level decay tau applies;
                    # untagged callers default to leaf (0).
                    episode_data.get('level', 0),
                    json.dumps(episode_data.get('transcript_ids', [])),
                    transcript_id_start,
                    transcript_id_end,
                    episode_data.get('emotional_valence'),
                    episode_data.get('emotional_arousal'),
                    json.dumps(episode_data.get('consolidated_from', [])),
                    episode_data.get('storage_strength', 1.0),
                    episode_data.get('retrieval_weight', 1.0),
                    episode_data.get('location_lat'),
                    episode_data.get('location_lon'),
                    episode_data.get('location_name'),
                    now_iso,
                ))

                # Insert embedding into vec table if available
                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

                # Sync FTS index (external-content table requires explicit insert)
                rowid = cursor.execute(
                    "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
                ).fetchone()
                if rowid:
                    conn.execute(
                        "INSERT INTO episodes_fts(rowid, gist) VALUES (?, ?)",
                        (rowid[0], episode_data['gist']),
                    )

                cursor.close()

                logging.info(f"Stored episode {episode_id} for channel '{episode_data['channel']}'")

                return episode_id

        except Exception as e:
            logging.error(f"Failed to store episode: {e}")
            raise

    def _store_embedding(self, conn: "_Conn", episode_id: str, embedding: object) -> None:
        """Store an embedding blob in the ``episodes_vec`` virtual table."""
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return

            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                rowid = cast(tuple[object, ...], row)[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )
            cursor.close()
        except Exception as e:
            logging.warning(f"Failed to store episode embedding: {e}")

    def update_episode(self, episode_id: str, fields: dict[str, object], embedding: object = None) -> None:
        """Empty ``fields`` + no ``embedding`` is a no-op."""
        if not fields and embedding is None:
            return

        with self.db_service.connection() as conn:
            if fields:
                set_clauses = [f"{key} = ?" for key in fields]
                set_clauses.append("updated_at = datetime('now')")
                values = list(fields.values()) + [episode_id]
                query = f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?"
                conn.execute(query, values)

            if embedding is not None:
                self._store_embedding(conn, episode_id, embedding)

    def soft_delete(self, episode_id: str) -> None:
        with self.db_service.connection() as conn:
            conn.execute("""
                UPDATE episodes
                SET deleted_at = datetime('now')
                WHERE id = ? AND deleted_at IS NULL
            """, (episode_id,))

    def set_consolidated_into(self, leaf_id: str, super_id: str) -> None:
        """Consolidation is a write-relevant event for the leaf — its content
        has just been rolled into a parent — so the relevance clock advances.
        The child is also tombstoned: its information now lives in the
        parent, so the decay engine collapses it onto the 7-day tombstone
        tau and hard-deletes it once aged past the janitor threshold. Both
        markers are written in one statement so a child can never carry a
        back-pointer without a tombstone."""
        now_iso = utc_now().isoformat()
        with self.db_service.connection() as conn:
            conn.execute(
                "UPDATE episodes "
                "SET consolidated_into = ?, last_relevant_at = ?, tombstoned_at = ? "
                "WHERE id = ?",
                (super_id, now_iso, now_iso, leaf_id),
            )

    def fetch_fact_extraction_backlog(self, limit: int) -> list[dict[str, object]]:
        """Return up to ``limit`` episodes awaiting fact extraction, oldest-first.

        The backlog is every non-deleted episode whose ``facts_extracted_at`` is
        still NULL — both fresh episodes and the pre-feature fossils that drain on
        the per-tick budget (self-healing convergence, no migration script).
        Oldest-first (``created_at ASC``) gives the cheapest clean supersession
        chains; bi-temporal invalidation keeps correctness order-independent.

        Returns a list of ``{id, gist, created_at, channel}`` dicts.
        """
        with self.db_service.connection() as conn:
            rows = conn.execute(
                "SELECT id, gist, created_at, channel FROM episodes "
                "WHERE facts_extracted_at IS NULL AND deleted_at IS NULL "
                "ORDER BY created_at ASC, id ASC "
                "LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {'id': str(r[0]), 'gist': r[1], 'created_at': r[2], 'channel': r[3]}
            for r in rows
        ]

    def set_facts_extracted_at(self, episode_id: str) -> None:
        """Stamping ``facts_extracted_at`` removes the episode from the
        backlog so the next tick advances. This is a durable progress
        marker — a tick interrupted mid-budget re-processes at most the
        un-stamped remainder."""
        now_iso = utc_now().isoformat()
        with self.db_service.connection() as conn:
            conn.execute(
                "UPDATE episodes SET facts_extracted_at = ? WHERE id = ?",
                (now_iso, episode_id),
            )

    def get_episode_by_id(self, episode_id: str) -> Optional[dict[str, object]]:
        """Pure read — fetching never mutates. Relevance advances only on
        write-relevant events (creation, consolidation), not on access."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, gist, salience, channel,
                           created_at, updated_at, last_accessed_at, access_count,
                           transcript_ids, transcript_id_start, transcript_id_end,
                           emotional_valence, emotional_arousal,
                           consolidated_from, consolidated_into,
                           storage_strength, retrieval_weight,
                           location_lat, location_lon, location_name
                    FROM episodes
                    WHERE id = ? AND deleted_at IS NULL
                """, (episode_id,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                episode = {
                    'id': str(row[0]),
                    'gist': row[1],
                    'salience': row[2],
                    'channel': row[3],
                    'created_at': row[4],
                    'updated_at': row[5],
                    'last_accessed_at': row[6],
                    'access_count': row[7],
                    'transcript_ids': row[8] if row[8] is not None else '[]',
                    'transcript_id_start': row[9],
                    'transcript_id_end': row[10],
                    'emotional_valence': row[11],
                    'emotional_arousal': row[12],
                    'consolidated_from': row[13] if row[13] is not None else '[]',
                    'consolidated_into': row[14],
                    'storage_strength': row[15] if row[15] is not None else 1.0,
                    'retrieval_weight': row[16] if row[16] is not None else 1.0,
                    'location_lat': row[17],
                    'location_lon': row[18],
                    'location_name': row[19],
                }

                return episode

        except Exception as e:
            logging.error(f"Failed to get episode by ID: {e}")
            return None

# ── Module-level novelty helpers ─────────────────────────────────────────────


def _fetch_novelty_comparison_set(channel: str) -> list[bytes]:
    """Comparison set = dedup(last-100 by created_at) ∪ (top-100 by
    access_count), apex-only (consolidated_into IS NULL), same channel,
    not deleted. Returns [] on any failure — novelty will be 1.0
    (fully novel)."""
    from services.database_service import get_shared_db_service
    from services.episodic_constants import NOVELTY_RECENT_LIMIT, NOVELTY_ACTIVATION_LIMIT

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id FROM episodes
                WHERE channel = ? AND deleted_at IS NULL AND consolidated_into IS NULL
                ORDER BY created_at DESC LIMIT ?
                """,
                (channel, NOVELTY_RECENT_LIMIT),
            )
            recent_ids = {row[0] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT id FROM episodes
                WHERE channel = ? AND deleted_at IS NULL AND consolidated_into IS NULL
                ORDER BY access_count DESC LIMIT ?
                """,
                (channel, NOVELTY_ACTIVATION_LIMIT),
            )
            top_ids = {row[0] for row in cursor.fetchall()}

            all_ids = list(recent_ids | top_ids)
            if not all_ids:
                cursor.close()
                return []

            placeholders = ','.join('?' * len(all_ids))
            cursor.execute(
                f"""
                SELECT ev.embedding
                FROM episodes_vec ev
                JOIN episodes e ON e.rowid = ev.rowid
                WHERE e.id IN ({placeholders})
                  AND ev.embedding IS NOT NULL
                """,
                all_ids,
            )
            blobs = [row[0] for row in cursor.fetchall() if row[0]]
            cursor.close()

        return blobs

    except Exception as exc:
        logging.warning(f"[NOVELTY] _fetch_novelty_comparison_set failed: {exc}")
        return []


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


def _fetch_apex_embeddings(channel: str, level: int) -> tuple[list[str], list[bytes]]:
    """Return (ids, embedding blobs) for apex episodes at a given hierarchy level.

    Apex = ``consolidated_into IS NULL AND deleted_at IS NULL`` with an embedding
    present, scoped to one channel and one ``level`` (0 = leaves, 1 = supers).
    Returns two index-aligned lists, or two empty lists on any query failure.
    """
    from services.database_service import get_shared_db_service

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.id, ev.embedding
                FROM episodes e
                JOIN episodes_vec ev ON ev.rowid = e.rowid
                WHERE e.channel = ?
                  AND e.level = ?
                  AND e.consolidated_into IS NULL
                  AND e.deleted_at IS NULL
                  AND ev.embedding IS NOT NULL
                ORDER BY e.created_at ASC
                """,
                (channel, level),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        logging.warning(f"[SUPER_CLUSTER] apex fetch failed (channel={channel}, level={level}): {exc}")
        return [], []

    ep_ids = [str(r[0]) for r in rows]
    ep_embs = [r[1] for r in rows]
    return ep_ids, ep_embs


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

    ep_ids, ep_embs = _fetch_apex_embeddings(channel, level=0)

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
        import numpy as np

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
