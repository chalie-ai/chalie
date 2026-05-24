# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Episodic Service — Episode storage + CRUD.

Retrieval lives in ``episodic_retrieval_service``. Super-episode clustering
helpers (``find_super_candidates``, ``compute_novelty``, and their DB helpers)
remain module-level in this file because they operate directly on the
``episodes`` + ``episodes_vec`` tables and are used by ``transcript_service``
during post-extraction consolidation.
"""

import json
import logging
import struct
import uuid
from typing import Optional

from services.database_service import DatabaseService
from services.embedding_utils import pack_embedding


class EpisodicService:
    """Manages episode storage and CRUD (no retrieval — see
    ``episodic_retrieval_service.retrieve``)."""

    def __init__(self, database_service: DatabaseService, config: dict = None):
        """Initialize the episodic service.

        Args:
            database_service: DatabaseService instance for connection management.
            config: Optional config dict (currently unused — retained for
                back-compat with existing constructor call sites).
        """
        self.db_service = database_service
        self.config = config or {}

    # ── Storage / CRUD ───────────────────────────────────────────────

    def store_episode(self, episode_data: dict, *, embedding=None) -> str:
        """Store a new episode in the database.

        Required fields: gist, salience, channel.

        Args:
            episode_data: Episode fields dict. May include an 'embedding' key
                for backwards compatibility, but the ``embedding`` kwarg takes
                precedence when both are supplied.
            embedding: Optional embedding vector (list of floats). When provided,
                supersedes episode_data.get('embedding').

        Returns:
            UUID of the created episode.

        Raises:
            ValueError: If any required field is missing.
        """
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

                cursor.execute("""
                    INSERT INTO episodes (
                        id, gist, salience, channel,
                        transcript_ids, transcript_id_start, transcript_id_end,
                        emotional_valence, emotional_arousal,
                        consolidated_from, storage_strength, retrieval_weight,
                        location_lat, location_lon, location_name
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    episode_data['gist'],
                    episode_data['salience'],
                    episode_data['channel'],
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

    def _store_embedding(self, conn, episode_id: str, embedding):
        """Store an embedding blob in the ``episodes_vec`` virtual table."""
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return

            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )
            cursor.close()
        except Exception as e:
            logging.warning(f"Failed to store episode embedding: {e}")

    def update_episode(self, episode_id: str, fields: dict, embedding=None) -> None:
        """Update an existing episode with the provided field values.

        Args:
            episode_id: The episode UUID.
            fields: Column→value mapping to apply. Empty dict is a no-op.
            embedding: Optional new embedding vector to store in episodes_vec.

        Raises:
            Exception: Propagates any database error to the caller.
        """
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
        """Soft-delete an episode by setting its ``deleted_at`` timestamp.

        Raises:
            Exception: Propagates any database error to the caller.
        """
        with self.db_service.connection() as conn:
            conn.execute("""
                UPDATE episodes
                SET deleted_at = datetime('now')
                WHERE id = ? AND deleted_at IS NULL
            """, (episode_id,))

    def set_consolidated_into(self, leaf_id: str, super_id: str) -> None:
        """Set the consolidated_into back-pointer on a leaf episode.

        Args:
            leaf_id: UUID of the leaf episode to update.
            super_id: UUID of the super-episode (episodes.id TEXT).

        Raises:
            Exception: Propagates any database error to the caller.
        """
        with self.db_service.connection() as conn:
            conn.execute(
                "UPDATE episodes SET consolidated_into = ? WHERE id = ?",
                (super_id, leaf_id),
            )

    def get_episode_by_id(self, episode_id: str) -> Optional[dict]:
        """Retrieve a single non-deleted episode by its UUID.

        Also triggers an access count + storage strength boost.
        """
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

                # Update access tracking
                self._update_activation_score(episode_id)

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

    def _update_activation_score(self, episode_id: str):
        """Boost storage_strength and reset retrieval_weight on access."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE episodes
                    SET access_count = access_count + 1,
                        last_accessed_at = datetime('now'),
                        storage_strength = MIN(storage_strength + 0.1, 10.0),
                        retrieval_weight = MIN(retrieval_weight + 0.3, 1.0)
                    WHERE id = ?
                """, (episode_id,))

                cursor.close()

        except Exception as e:
            logging.error(f"Failed to update activation score: {e}")


# ── Module-level novelty helpers ─────────────────────────────────────────────


def _fetch_novelty_comparison_set(channel: str) -> list[bytes]:
    """Return embedding blobs for the (recent ∪ most-activated) apex episodes.

    Comparison set = dedup(last-100 by created_at) ∪ (top-100 by access_count),
    apex-only (consolidated_into IS NULL), same channel, not deleted.

    Returns a list of raw binary blobs suitable for compute_novelty().
    Returns [] on any failure — novelty will be 1.0 (fully novel).
    """
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


def _cosine_sim_blobs(blob_a: bytes, blob_b: bytes) -> float:
    """Compute cosine similarity between two sqlite-vec binary blobs.

    Embeddings from EmbeddingService are L2-normalised, so cosine similarity
    is just the dot product of the unpacked float vectors.  Returns 0.0 if
    either blob is malformed or the lengths differ.
    """
    try:
        import numpy as np
        vec_a = np.array(_unpack_blob(blob_a), dtype=np.float32)
        vec_b = np.array(_unpack_blob(blob_b), dtype=np.float32)
        if vec_a.shape != vec_b.shape or vec_a.shape[0] == 0:
            return 0.0
        # Embeddings are pre-normalised — dot product equals cosine sim.
        return float(np.dot(vec_a, vec_b))
    except Exception:
        return 0.0


def _build_adjacency(ep_embs: list[bytes], threshold: float) -> list[list[int]]:
    """Return an undirected adjacency list for episodes whose cosine >= threshold."""
    n = len(ep_embs)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_sim_blobs(ep_embs[i], ep_embs[j]) >= threshold:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _bfs_components(adj: list[list[int]], n: int, min_cluster: int) -> list[list[int]]:
    """Return connected components of size >= min_cluster via iterative BFS."""
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(n):
        if start in visited:
            continue
        component: list[int] = []
        queue: list[int] = [start]
        visited.add(start)
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbour in adj[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        if len(component) >= min_cluster:
            components.append(component)
    return components


def find_super_candidates(channel: str) -> list[list[str]]:
    """Return lists of episode IDs that form semantic clusters via connected components.

    A cluster qualifies when:
      - it contains at least SUPER_EPISODE_MIN_CLUSTER episodes,
      - every member is connected (directly or transitively) to every other
        member via edges where cosine >= SUPER_EPISODE_THRESHOLD.

    Algorithm: build an undirected graph where nodes are apex episodes and
    edges are pairs whose cosine >= SUPER_EPISODE_THRESHOLD. Emit each
    connected component of size >= SUPER_EPISODE_MIN_CLUSTER as a cluster.
    Clique-tightness is NOT required — a chain of related episodes counts as
    one cluster even if the endpoints aren't direct neighbours. This matches
    how humans group related memories and prevents the pair-threshold bar
    from compounding against itself when min_cluster > 2.

    Args:
        channel: The episode channel to cluster.

    Returns:
        List of ID-lists (strings), each list being one connected component.
        Components are non-overlapping and deterministic (sorted IDs within
        each list; outer list sorted by first ID).
    """
    from services.database_service import get_shared_db_service
    from services.episodic_constants import SUPER_EPISODE_THRESHOLD, SUPER_EPISODE_MIN_CLUSTER

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
                  AND e.consolidated_into IS NULL
                  AND e.deleted_at IS NULL
                  AND ev.embedding IS NOT NULL
                ORDER BY e.created_at ASC
                """,
                (channel,),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        logging.warning(f"[SUPER_CLUSTER] find_super_candidates query failed: {exc}")
        return []

    if not rows:
        return []

    ep_ids: list[str] = [str(r[0]) for r in rows]
    ep_embs: list[bytes] = [r[1] for r in rows]

    adj = _build_adjacency(ep_embs, SUPER_EPISODE_THRESHOLD)
    raw_components = _bfs_components(adj, len(ep_ids), SUPER_EPISODE_MIN_CLUSTER)

    clusters = sorted(
        [sorted(ep_ids[m] for m in comp) for comp in raw_components],
        key=lambda c: c[0],
    )
    if clusters:
        logging.info(
            f"[SUPER_CLUSTER] {len(clusters)} component(s) found "
            f"(sizes={[len(c) for c in clusters]}, channel={channel})"
        )
    return clusters


def compute_novelty(new_embedding, prior_embeddings: list[bytes]) -> float:
    """Return semantic novelty of new_embedding vs the prior comparison set.

    novelty = 1.0 - max(cosine_sim(new, prior) for prior in prior_embeddings)
    Clamped to [0.0, 1.0]. Returns 1.0 (fully novel) if prior_embeddings is empty.

    Args:
        new_embedding: The new embedding as a list/array of floats (L2-normalized).
        prior_embeddings: List of raw binary blobs from _fetch_novelty_comparison_set().

    Returns:
        Float in [0.0, 1.0]. Higher = more novel.
    """
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
