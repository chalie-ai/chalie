"""Tests for find_seed_cluster — seed-on-creation candidate discovery.

All tests use the real db fixture with zero mocks. Episodes are seeded via
EpisodicService.store_episode with real embeddings (256-dim, matching the test
DB vec tables). Cover:

  * A seed plus >= SEED_CLUSTER_MIN_SIZE near-duplicate apex episodes at the
    same level/channel → returns a sorted id list containing the seed and its
    neighbours.
  * A seed with only a handful of near neighbours (below the floor) → None.
  * Near neighbours that are NOT apex (consolidated_into set) are excluded —
    an otherwise-large neighbourhood that is mostly consolidated yields None.
  * Near neighbours at a DIFFERENT level are excluded.
  * Neighbours beyond SEED_CLUSTER_RADIUS are excluded.
  * The seed itself is never in the returned cluster twice and the count never
    exceeds SEED_CLUSTER_MAX_MATCHES + 1 (seed).
"""

import sqlite3

import pytest

from services.episodic_service import EpisodicService, find_seed_cluster
from services.episodic_constants import (
    SEED_CLUSTER_MAX_MATCHES,
    SEED_CLUSTER_MIN_SIZE,
)
from services.embedding_utils import pack_embedding

pytestmark = pytest.mark.unit

_DIM = 256


def _unit(index: int, dim: int = _DIM) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _seed(es: EpisodicService, gist: str, *, embedding: list[float], salience: int = 6,
          channel: str = "chat", level: int = 0) -> str:
    return es.store_episode(
        {"gist": gist, "salience": salience, "channel": channel, "level": level},
        embedding=embedding,
    )


def _query_embedding() -> bytes:
    blob = pack_embedding(_unit(1))
    assert blob is not None
    return blob


def test_cluster_forms_with_enough_near_neighbours(db: sqlite3.Connection) -> None:
    """A seed plus >= SEED_CLUSTER_MIN_SIZE near-duplicate apex episodes at the
    same level/channel → returns a sorted id list containing the seed and its
    neighbours."""
    es = EpisodicService()
    seed_id = _seed(es, "seed episode about gozo", embedding=_unit(1))

    # Seed 9 more near-duplicates (total 10 = SEED_CLUSTER_MIN_SIZE).
    for i in range(1, SEED_CLUSTER_MIN_SIZE):
        _seed(es, f"near-duplicate {i} about gozo", embedding=_unit(1))

    result = find_seed_cluster(seed_id, "chat", 0, _query_embedding())

    assert result is not None
    assert seed_id in result
    assert len(result) == SEED_CLUSTER_MIN_SIZE
    assert result == sorted(result), "Result must be sorted (deterministic)"


def test_cluster_below_floor_returns_none(db: sqlite3.Connection) -> None:
    """A seed with only a handful of near neighbours (below the floor) → None."""
    es = EpisodicService()
    seed_id = _seed(es, "seed episode about gozo", embedding=_unit(1))

    # Seed 5 near-duplicates (total 6 < SEED_CLUSTER_MIN_SIZE).
    for i in range(1, 6):
        _seed(es, f"near-duplicate {i} about gozo", embedding=_unit(1))

    result = find_seed_cluster(seed_id, "chat", 0, _query_embedding())

    assert result is None


def test_non_apex_neighbours_excluded(db: sqlite3.Connection) -> None:
    """Near neighbours that are NOT apex (consolidated_into set) are excluded —
    an otherwise-large neighbourhood that is mostly consolidated yields None."""
    es = EpisodicService()
    seed_id = _seed(es, "seed episode about gozo", embedding=_unit(1))

    # Seed 9 near-duplicates.
    neighbour_ids = []
    for i in range(1, SEED_CLUSTER_MIN_SIZE):
        nid = _seed(es, f"near-duplicate {i} about gozo", embedding=_unit(1))
        neighbour_ids.append(nid)

    # Consolidate all neighbours (mark them as rolled up).
    for nid in neighbour_ids:
        es.update_episode(nid, {"consolidated_into": "super_ep_id"})

    result = find_seed_cluster(seed_id, "chat", 0, _query_embedding())

    assert result is None


def test_different_level_neighbours_excluded(db: sqlite3.Connection) -> None:
    """Near neighbours at a DIFFERENT level are excluded."""
    es = EpisodicService()
    seed_id = _seed(es, "seed episode about gozo", embedding=_unit(1))

    # Seed 9 near-duplicates at level 1 (different from seed's level 0).
    for i in range(1, SEED_CLUSTER_MIN_SIZE):
        _seed(es, f"near-duplicate {i} about gozo", embedding=_unit(1), level=1)

    result = find_seed_cluster(seed_id, "chat", 0, _query_embedding())

    assert result is None


def test_neighbours_beyond_radius_excluded(db: sqlite3.Connection) -> None:
    """Neighbours beyond SEED_CLUSTER_RADIUS are excluded."""
    es = EpisodicService()
    seed_id = _seed(es, "seed episode about gozo", embedding=_unit(1))

    # Seed 9 near-duplicates but with embeddings that are orthogonal to the seed
    # (cosine distance = 1.0, which is > SEED_CLUSTER_RADIUS = 0.45).
    for i in range(1, SEED_CLUSTER_MIN_SIZE):
        _seed(es, f"near-duplicate {i} about gozo", embedding=_unit(i + 100))

    result = find_seed_cluster(seed_id, "chat", 0, _query_embedding())

    assert result is None


def test_seed_not_duplicated_and_count_within_limit(db: sqlite3.Connection) -> None:
    """The seed itself is never in the returned cluster twice and the count
    never exceeds SEED_CLUSTER_MAX_MATCHES + 1 (seed)."""
    es = EpisodicService()
    seed_id = _seed(es, "seed episode about gozo", embedding=_unit(1))

    # Seed SEED_CLUSTER_MAX_MATCHES near-duplicates (total = SEED_CLUSTER_MAX_MATCHES + 1).
    for i in range(1, SEED_CLUSTER_MAX_MATCHES + 1):
        _seed(es, f"near-duplicate {i} about gozo", embedding=_unit(1))

    result = find_seed_cluster(seed_id, "chat", 0, _query_embedding())

    assert result is not None
    # Seed appears exactly once.
    assert result.count(seed_id) == 1
    # Total count <= SEED_CLUSTER_MAX_MATCHES + 1 (seed + max neighbours).
    assert len(result) <= SEED_CLUSTER_MAX_MATCHES + 1
