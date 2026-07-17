# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — seed-on-creation super-episode consolidation, end to end.

Companion to ``test_seed_cluster.py`` (which covers the discovery primitive
``find_seed_cluster``): these tests drive the *write* half of the pipeline —
``consolidate_cluster`` encoding one parent super-episode from a cluster and
wiring every child's ``consolidated_into`` back-pointer — plus the
fire-on-creation gate that ``store_episode`` runs.

Everything runs on the real ``db`` fixture with zero mocks of the code under
test. The only substitutions are the two non-deterministic external ML
boundaries, at exactly the seams the codebase already uses elsewhere:

  * the LLM network transport, patched at ``services.provider_service.build_client``
    with a scripted provider that replays one ``ProviderResponse`` — the same
    seam ``test_message_processor_runaway_loop.py`` drives; and
  * the embedding model, supplied as a deterministic 768-dim embedder — either
    via ``consolidate_cluster``'s injected ``emb_svc`` parameter (pure DI) or,
    for the ``store_episode`` path, patched at ``get_embedding_service`` so the
    written vector matches the test DB's 768-dim vec table.

The consolidation encoder, JSON parse, field stamping, transcript-id union,
salience compute, parent write and lineage wiring all run for real.
"""

import json
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from models.episode import Episode
from models.provider_response import ProviderResponse
from services.episodic_service import (
    EpisodicService,
    _consolidation_lock,
    consolidate_cluster,
    find_seed_cluster,
)
from services.embedding_utils import pack_embedding
from services.episodic_constants import (
    MAX_EPISODE_LEVEL,
    SEED_CLUSTER_MIN_SIZE,
)
from services.source_profiles import CHANNEL_USER

if TYPE_CHECKING:
    from services.embedding_service import EmbeddingService

pytestmark = pytest.mark.unit

_DIM = 768

# ProviderService builds its thin transport client via this factory — the real
# LLM network boundary (see test_message_processor_runaway_loop.py). Patching it
# substitutes the network call and nothing else.
_BUILD_CLIENT = "services.provider_service.build_client"

# The super-episode encoder's contract is a single JSON object; the consolidated
# gist is what gets stored on the parent.
_ENCODER_GIST = "The team settled the Gozo launch timeline and cleared the blocker."
_ENCODER_JSON = json.dumps({"gist": _ENCODER_GIST, "has_open_loop": False})


def _unit(index: int) -> list[float]:
    v = [0.0] * _DIM
    v[index] = 1.0
    return v


def _query_blob() -> bytes:
    blob = pack_embedding(_unit(1))
    assert blob is not None
    return blob


class _FixedEmbedder:
    """A deterministic stand-in for the embedding model: every gist maps to the
    same 768-dim unit vector, matching the test DB's vec-table dimension."""

    def generate_embedding(self, text: str, mp: object = None) -> list[float]:
        return _unit(0)


class _ScriptedClient:
    """Replays scripted ``ProviderResponse`` frames, one per ``send()``. Past the
    end it returns a benign no-tool terminal so an unscripted extra call ends a
    turn cleanly rather than raising (mirrors the runaway-loop test's client)."""

    _TERMINAL = ProviderResponse(text="", model="scripted-terminal", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _seed(
    es: EpisodicService,
    gist: str,
    *,
    embedding: list[float],
    channel: str = "chat",
    level: int = 0,
    salience: int = 6,
) -> str:
    """Store one episode. On the muted 'chat' channel this fires no consolidation
    daemon; at ``level == MAX_EPISODE_LEVEL`` the level gate blocks the daemon on
    any channel — both keep seeding side-effect-free."""
    return es.store_episode(
        {"gist": gist, "salience": salience, "channel": channel, "level": level},
        embedding=embedding,
    )


def _level_ids(db: sqlite3.Connection, level: int) -> list[str]:
    """Ids of live (non-deleted) episodes at exactly *level*."""
    rows = db.execute(
        "SELECT id FROM episodes WHERE level = ? AND deleted_at IS NULL",
        (level,),
    ).fetchall()
    return [row[0] for row in rows]


def _count_episodes(db: sqlite3.Connection) -> int:
    return int(
        db.execute("SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL").fetchone()[0]
    )


def _join_consolidation_daemons(timeout_s: float = 10.0) -> None:
    """Block until every consolidation daemon has exited, so no off-turn thread
    is still running when the caller tears its patches down and the next test
    repoints the DB. ``_check_and_consolidate`` names these threads
    ``consolidate-seed-<channel>``; we join only those, never the process-
    lifetime ``memory-store-reaper`` the encoder turn spins up (which would never
    exit). Re-scanning in a loop also catches the recursive parent-write daemon a
    winning consolidation spawns for the next level up."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name.startswith("consolidate-seed-") and t.is_alive()
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=0.1)


# ── consolidate_cluster: the encode + write + lineage seam ────────────────────


def test_consolidate_cluster_forms_one_parent_with_lineage(db: sqlite3.Connection) -> None:
    """A floor-sized cluster consolidates into exactly one parent super-episode
    carrying the encoder's gist, and every source is stamped as rolled up into
    that parent."""
    es = EpisodicService()
    source_ids = [
        _seed(es, f"gozo launch note {i}", embedding=_unit(1))
        for i in range(SEED_CLUSTER_MIN_SIZE)
    ]

    client = _ScriptedClient(ProviderResponse(text=_ENCODER_JSON, model="scripted"))
    with patch(_BUILD_CLIENT, return_value=client):
        wrote = consolidate_cluster(
            "chat", source_ids, 1, cast("EmbeddingService", _FixedEmbedder()), es, [],
        )

    assert wrote is True
    parents = _level_ids(db, 1)
    assert len(parents) == 1, "a cluster must consolidate into exactly one parent"
    # Exactly one new episode was written: the sources plus the single parent.
    assert _count_episodes(db) == SEED_CLUSTER_MIN_SIZE + 1
    parent = Episode.by_id(parents[0])
    assert parent is not None
    assert parent.gist == _ENCODER_GIST
    assert set(json.loads(parent.consolidated_from or "[]")) == set(source_ids)

    # Every source now points at the parent and is no longer apex.
    for sid in source_ids:
        src = Episode.by_id(sid)
        assert src is not None
        assert src.consolidated_into == parent.id


def test_below_floor_cluster_does_not_consolidate(db: sqlite3.Connection) -> None:
    """Fewer than SEED_CLUSTER_MIN_SIZE sources never form a parent — the one
    minimum-cluster-size floor is enforced in the encoder path too, so the
    scripted provider is never even reached."""
    es = EpisodicService()
    source_ids = [
        _seed(es, f"note {i}", embedding=_unit(1))
        for i in range(SEED_CLUSTER_MIN_SIZE - 1)
    ]

    client = _ScriptedClient(ProviderResponse(text=_ENCODER_JSON, model="scripted"))
    with patch(_BUILD_CLIENT, return_value=client):
        wrote = consolidate_cluster(
            "chat", source_ids, 1, cast("EmbeddingService", _FixedEmbedder()), es, [],
        )

    assert wrote is False
    assert client.sends == 0, "below the floor the encoder must not be called"
    assert _level_ids(db, 1) == []
    for sid in source_ids:
        src = Episode.by_id(sid)
        assert src is not None
        assert src.consolidated_into is None


def test_empty_encoder_response_forms_no_parent(db: sqlite3.Connection) -> None:
    """An empty encoder response leaves the cluster untouched — no parent, and
    every source stays apex — rather than writing a hollow super-episode."""
    es = EpisodicService()
    source_ids = [
        _seed(es, f"note {i}", embedding=_unit(1))
        for i in range(SEED_CLUSTER_MIN_SIZE)
    ]

    client = _ScriptedClient(ProviderResponse(text="", model="scripted"))
    with patch(_BUILD_CLIENT, return_value=client):
        wrote = consolidate_cluster(
            "chat", source_ids, 1, cast("EmbeddingService", _FixedEmbedder()), es, [],
        )

    assert wrote is False
    assert _level_ids(db, 1) == []
    for sid in source_ids:
        src = Episode.by_id(sid)
        assert src is not None
        assert src.consolidated_into is None


# ── Fire-on-creation gate via store_episode ───────────────────────────────────


def test_seed_at_max_level_does_not_consolidate(db: sqlite3.Connection) -> None:
    """A cluster already at MAX_EPISODE_LEVEL never rolls up further: discovery
    would still return a full cluster, but storing the seed spawns no daemon —
    the level gate short-circuits — so no episode above the cap is ever written."""
    es = EpisodicService()
    ids = [
        _seed(es, f"era digest {i}", embedding=_unit(1),
              channel=CHANNEL_USER, level=MAX_EPISODE_LEVEL)
        for i in range(SEED_CLUSTER_MIN_SIZE)
    ]

    # The neighbourhood is genuinely consolidation-worthy...
    cluster = find_seed_cluster(ids[-1], CHANNEL_USER, MAX_EPISODE_LEVEL, _query_blob())
    assert cluster is not None and len(cluster) >= SEED_CLUSTER_MIN_SIZE

    # ...but nothing forms above the cap, and no parent was written at all.
    assert _level_ids(db, MAX_EPISODE_LEVEL + 1) == []
    assert _count_episodes(db) == SEED_CLUSTER_MIN_SIZE


def test_consolidation_fires_off_turn_under_single_writer_lock(db: sqlite3.Connection) -> None:
    """Storing a floor-sized burst of near-duplicates on an episode-extracting
    channel consolidates them — but OFF the turn (every store returns while the
    work is still pending behind the per-channel lock) and under a single writer
    (many concurrent daemons yield exactly one parent, not one each)."""
    es = EpisodicService()

    client = _ScriptedClient(ProviderResponse(text=_ENCODER_JSON, model="scripted"))
    with patch(_BUILD_CLIENT, return_value=client), \
         patch(
             "services.embedding_service.get_embedding_service",
             return_value=_FixedEmbedder(),
         ):
        lock = _consolidation_lock(CHANNEL_USER)
        lock.acquire()
        try:
            source_ids = [
                _seed(es, f"gozo burst {i}", embedding=_unit(1), channel=CHANNEL_USER)
                for i in range(SEED_CLUSTER_MIN_SIZE)
            ]
            # All stores returned; every consolidation daemon they spawned is
            # blocked on the lock we hold — so nothing has consolidated yet.
            # That is the proof consolidation runs off the turn, not on it.
            assert _level_ids(db, 1) == [], "consolidation ran on the turn, not off it"
        finally:
            lock.release()
        # Release lets the blocked daemons proceed; join them (and the recursive
        # next-level daemon a winner spawns) so none is mid-flight against the
        # patched provider when we tear the patch down. Once they have all
        # exited, the winner's parent + lineage are committed for the assertions.
        _join_consolidation_daemons()

    parents = _level_ids(db, 1)
    assert len(parents) == 1, "single-writer lock must yield one parent, not one per daemon"
    parent = Episode.by_id(parents[0])
    assert parent is not None
    assert set(json.loads(parent.consolidated_from or "[]")) == set(source_ids)
    for sid in source_ids:
        src = Episode.by_id(sid)
        assert src is not None
        assert src.consolidated_into == parent.id
