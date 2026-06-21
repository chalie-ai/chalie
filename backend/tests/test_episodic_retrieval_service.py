"""Feature tests for collapsed-tree episode retrieval engine.

All lanes run real (zero mocks): FTS, sqlite-vec, per-lane min-max normalisation,
relative score floor, and collapsed-tree walk — via the production retrieve entry point
and real db fixture with episodes seeded against the actual tables.

What's implemented:

  * vector lane stays non-empty on representative queries (radius-0 regression pinned dead);
  * lane signals min-max normalised within candidate pool; RELATIVE floor drops weak tail —
    survivors only, count may be < k, never padded;
  * leaves, super-episodes and era digests compete in ONE pool — a super and its leaf can
    both surface (no apex-walk that discarded the lower levels).

Test DB vec tables are 256-dim (conftest template), so embeddings are seeded to 256 dimensions.
"""

import sqlite3
from typing import cast

import pytest

from services import episodic_retrieval_service as ers
from services.episodic_service import EpisodicService

pytestmark = pytest.mark.unit

_DIM = 256


def _unit(index: int, dim: int = _DIM) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _seed(es: EpisodicService, gist: str, *, embedding: list[float], salience: int = 6,
          channel: str = "chat") -> str:
    return es.store_episode(
        {"gist": gist, "salience": salience, "channel": channel},
        embedding=embedding,
    )


def test_vector_lane_returns_the_semantically_nearest_episode(db: sqlite3.Connection) -> None:
    """A representative query whose embedding matches one stored episode surfaces
    that episode through the vector lane — the lane is NON-EMPTY (the radius-0
    regression that silently emptied it is gone)."""
    from services.database_service import get_shared_db_service

    es = EpisodicService(get_shared_db_service())
    target = _seed(es, "home wifi password is hunter2", embedding=_unit(3))
    _seed(es, "grocery list has bananas and milk", embedding=_unit(120))

    episodes, telemetry = cast(
        "tuple[list[dict[str, object]], dict[str, object]]",
        ers.retrieve(query_text="wifi", query_embedding=_unit(3), channel="chat", k=10, return_telemetry=True),
    )

    ids = [e["id"] for e in episodes]
    assert target in ids, f"vector lane missed the nearest episode: {ids!r}"
    # The lane actually ran and produced candidates (radius-0 bug pinned dead).
    assert cast(int, telemetry["vector_candidates"]) >= 1, telemetry
    # The top hit is the exact match (vector_distance ~ 0).
    top = episodes[0]
    assert top["id"] == target
    assert top.get("vector_distance") is not None
    assert cast(float, top["vector_distance"]) == pytest.approx(0.0, abs=1e-3)


def test_relative_floor_drops_the_weak_tail_count_below_k(db: sqlite3.Connection) -> None:
    """A strong match and a much weaker (orthogonal) one go into the pool; the
    relative floor drops the weak candidate rather than padding the result up to
    k. Count < number of candidates, and the floor cut is reported."""
    from services.database_service import get_shared_db_service

    es = EpisodicService(get_shared_db_service())
    strong = _seed(es, "annual leave booked for the trip to Gozo", salience=9,
                   embedding=_unit(10))
    weak = _seed(es, "random note about a stapler", salience=1,
                 embedding=_unit(200))

    episodes, telemetry = cast(
        "tuple[list[dict[str, object]], dict[str, object]]",
        ers.retrieve(query_text="Gozo", query_embedding=_unit(10), channel="chat", k=10, return_telemetry=True),
    )

    ids = [e["id"] for e in episodes]
    assert strong in ids, f"strong match dropped: {ids!r}"
    # The weak, far candidate is floored out — NOT padded back in to fill k.
    assert weak not in ids, f"relative floor failed to drop the weak tail: {ids!r}"
    assert len(episodes) < 2, f"floor did not trim the pool: {episodes!r}"
    assert cast(int, telemetry["floor_cut_count"]) >= 1, telemetry


def test_collapsed_tree_surfaces_super_and_leaf_in_one_pool(db: sqlite3.Connection) -> None:
    """A leaf consolidated into a super-episode and the super itself BOTH compete
    in one pool. When a query matches both, both surface — the leaf is not walked
    away into its apex (collapsed-tree retrieval, )."""
    from services.database_service import get_shared_db_service

    es = EpisodicService(get_shared_db_service())
    leaf = _seed(es, "monday standup notes about the wifi outage",
                 embedding=_unit(7))
    super_ep = _seed(es, "weekly summary of network and wifi issues",
                     salience=8, embedding=_unit(8))
    # Promote the super to level 1 and hang the leaf off it — the real
    # consolidation columns the hierarchy path writes.
    es.update_episode(super_ep, {"level": 1})
    es.set_consolidated_into(leaf, super_ep)

    # A vector query equidistant-ish to both lands them both as candidates; FTS
    # ("wifi") matches both gists too.
    episodes = cast(
        "list[dict[str, object]]",
        ers.retrieve(
            query_text="wifi",
            query_embedding=_unit(7),
            channel="chat",
            k=10,
        ),
    )

    ids = [e["id"] for e in episodes]
    assert leaf in ids, f"leaf was discarded by an apex-walk: {ids!r}"
    assert super_ep in ids, f"super-episode did not compete in the pool: {ids!r}"


def test_higher_level_super_can_outrank_a_leaf(db: sqlite3.Connection) -> None:
    """A higher-salience super-episode competes on equal footing and can outrank
    a weaker leaf — levels are not muted; importance (salience × weight) lifts the
    super in the normalised composite."""
    from services.database_service import get_shared_db_service

    es = EpisodicService(get_shared_db_service())
    # Both match the query in the vector lane (same basis vector → both distance 0)
    # so lane relevance is equal; salience is close enough that BOTH clear the
    # relative floor and the only discriminator left is importance ordering.
    leaf = _seed(es, "brief chat about the Gozo ferry", salience=7,
                 embedding=_unit(15))
    super_ep = _seed(es, "consolidated memory of the whole Gozo trip",
                     salience=9, embedding=_unit(15))
    es.update_episode(super_ep, {"level": 2})

    episodes = cast(
        "list[dict[str, object]]",
        ers.retrieve(
            query_text="Gozo trip",
            query_embedding=_unit(15),
            channel="chat",
            k=10,
        ),
    )

    ids = [e["id"] for e in episodes]
    assert super_ep in ids, f"era/super digest never surfaced: {ids!r}"
    assert leaf in ids, f"leaf vanished: {ids!r}"
    # Equal lane relevance → the higher-importance super sorts ahead of the leaf.
    assert ids.index(super_ep) < ids.index(leaf), (
        f"higher-importance super did not outrank the weak leaf: {ids!r}"
    )


def test_empty_corpus_returns_nothing_without_padding(db: sqlite3.Connection) -> None:
    """No episodes stored → empty result and a clean telemetry envelope (no
    crash, no phantom rows). The floor never fabricates a result."""
    episodes, telemetry = cast(
        "tuple[list[dict[str, object]], dict[str, object]]",
        ers.retrieve(query_text="anything at all", query_embedding=_unit(5), channel="chat", k=10, return_telemetry=True),
    )
    assert episodes == []
    assert cast(int, telemetry["final_rrf_count"]) == 0
    assert cast(int, telemetry["floor_cut_count"]) == 0
