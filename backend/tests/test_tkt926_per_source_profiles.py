# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — TKT-926 per-source memory profiles + provenance fixes (G).

These drive the per-source memory behaviour end-to-end against the real
production stack:

  * Cross-pollination recall — a user-turn flashback surfaces episodes from
    every episode-producing channel (user + dmn + external-agent), not just the
    turn's own channel.
  * Fact-extraction per-source provenance — facts the worker extracts carry a
    channel-tagged ``data_graph.source`` so a dmn / external-agent fact is
    distinguishable from a user one.
  * Decay-janitor HEAVY-channel protection — the fossil janitor must NOT
    tombstone an apex dmn leaf (dmn never consolidates, so its leaves are
    permanently apex), while a muted/legacy leaf in the same state IS tombstoned.
  * Scheduled two-stage — a fired scheduled prompt runs its work loop on the
    muted ``scheduled`` channel (instruction recoverable there) and surfaces the
    result on the ``user`` channel.

ZERO mocks of in-process code. The single sanctioned seam is the LLM network
boundary (Pattern H): ``Providers._resolve`` is swapped via a plain try/finally
context manager — never ``unittest.mock``. Everything else is real: the ``db``
fixture (schema.sql through SchemaConvergenceService at 256-dim + redesign
backfill), real ``EpisodicService.store_episode`` (the production write path),
real ``DataGraphService``, real ``MemoryStore`` (the ``store`` fixture, TKT-922
dual-store isolation), real ``SubconsciousWorker`` / ``DecayEngineService`` /
``scheduler`` entry points.

═══════════════════════════════════════════════════════════════════════════════
RED-FIRST / COORDINATION NOTE (read before "fixing" a failure)
═══════════════════════════════════════════════════════════════════════════════
Written alongside the coder. The cross-pollination and scheduled-two-stage tests
are RED until their anchors land:
  * cross-pollination : ``turn_zero_flashback.py`` recall must pass channel=None
                        (currently ``channel=self._mp.config.channel``).
  * scheduled two-stage: ``scheduler_service._fire_item`` is_prompt branch must
                        run the ScheduledConfig work loop on channel='scheduled'
                        before the user return hop (currently a single
                        ``dispatch_message(source='scheduled')``).
The fact-extraction-provenance and janitor-protection tests are GREEN on arrival
(their anchors landed) and stand as regression locks: each would FAIL if the
provenance tag reverted to the bare default, or if the janitor reverted to
``channel != 'user'`` (which reaps dmn HEAVY memory wholesale — the cycle-39
class of silent loss).
"""

import contextlib
import json
import math
import struct
import threading
from datetime import timedelta

import pytest

from services.episodic_service import EpisodicService
from services.llm_clients.base import ProviderClient
from services.provider_api import ProviderApiResponse
from services.providers import Providers

pytestmark = pytest.mark.unit

_DIM = 256  # conftest vec tables are 256-dim (suite convention)


# ── 256-dim embedding helper (test_episodic_retrieval_service.py precedent) ─────

def _unit(index: int, dim: int = _DIM) -> list:
    v = [0.0] * dim
    v[index] = 1.0
    return v


# ── LLM network-boundary seam (Pattern H — the ONLY sanctioned mock) ───────────

class _FakeLLMService(ProviderClient):
    """Stand-in for the resolved provider client. A real ProviderClient subclass
    so Providers.send still runs prompt assembly, tool building and pre-flight."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, send_fn):
        self._send_fn = send_fn

    def get_context_limit(self) -> int:
        return 200_000

    def estimate_request_tokens(self, dto) -> int:
        return 1

    def send(self, dto) -> ProviderApiResponse:
        return self._send_fn(dto)


@contextlib.contextmanager
def _inject_fake_client(send_fn):
    """Swap Providers._resolve at the class level for the block. No unittest.mock."""
    original = Providers._resolve
    Providers._resolve = lambda self, *_a, **_kw: _FakeLLMService(send_fn)
    try:
        yield
    finally:
        Providers._resolve = original


def _text_response(text: str) -> ProviderApiResponse:
    """A plain assistant reply with no tool calls — ends an ACT loop on the first
    turn (the model decides it is done)."""
    return ProviderApiResponse(
        text=text, model="test-model", provider="mock", tool_calls=None,
    )


def _ops_response(ops: list) -> ProviderApiResponse:
    """A fact-extraction constrained-op batch (configs/channels/fact_extraction.py
    envelope: ``{"ops": [{op, key, value}, ...]}``)."""
    return ProviderApiResponse(
        text=json.dumps({"ops": ops}), model="test-model", provider="mock",
        tool_calls=None,
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _db_service():
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


def _seed_episode(db, gist, *, channel, emb_index=3, salience=8):
    """Seed one episode via the PRODUCTION write path (EpisodicService).

    store_episode writes both the episodes row and the episodes_fts row, and
    leaves facts_extracted_at NULL — exactly as a freshly-encoded episode would.
    """
    es = EpisodicService(_db_service())
    return es.store_episode(
        {"gist": gist, "salience": salience, "channel": channel},
        embedding=_unit(emb_index),
    )


def _worker():
    from services.subconscious_worker import SubconsciousWorker
    return SubconsciousWorker(tick_sec=10, idle_window_sec=60)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-pollination recall (Decision 1 / §3.9)
# ══════════════════════════════════════════════════════════════════════════════

def _new_user_turn(text: str):
    """A real UserConfig MessageProcessor positioned where the turn-0 seed fires —
    input row written (anchors the act-trail FK), config attached, tools seeded.
    Mirrors tests/test_turn0_flashback_continuation_gate.py:_new_turn."""
    from configs.channels import UserConfig
    from services.message_processor import MessageProcessor
    from services.transcript_service import write_input_row

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, text, {})
    mp.config = UserConfig()
    mp.uid = write_input_row("user", "user", text)
    mp.active_tools = list(mp.config.always_available or [])
    assert mp.config.channel == "user"
    return mp


def test_user_turn_recall_cross_pollinates_across_producing_channels(db, store):
    """A flashback recall on a USER turn must surface episodes stored from every
    episode-producing channel — user, dmn AND external-agent — because memory is
    one corpus that cross-pollinates (Decision 1). Before this change a user turn
    only saw its own channel's episodes, so dmn's HEAVY reflections were
    write-only.

    Driven through the real ``TurnZeroFlashback._recall_episodes`` (the exact
    method seed() calls). The FTS lane carries the episodes deterministically
    (the 768/256 vec mismatch no-ops on width); each gist shares the query token
    'harbour' so FTS5 matches all three.
    """
    from services.turn_zero_flashback import TurnZeroFlashback

    _seed_episode(db, "The user jogs along the harbour each morning.",
                  channel="user", emb_index=4)
    _seed_episode(db, "Reflection: the user values the harbour walk for calm.",
                  channel="dmn", emb_index=5)
    _seed_episode(db, "Agent noted the user's harbour cafe preference.",
                  channel="external-agent:bob", emb_index=6)

    mp = _new_user_turn("what do you remember about the harbour")
    episodes = TurnZeroFlashback(mp)._recall_episodes("harbour")

    channels = {ep.get("channel") for ep in episodes}
    assert "user" in channels, "user-channel episode missing from recall"
    assert "dmn" in channels, (
        "dmn HEAVY episode did not cross-pollinate into the user-turn recall — "
        "recall is still scoped to the turn's own channel"
    )
    assert "external-agent:bob" in channels, (
        "external-agent episode did not cross-pollinate into the user-turn recall"
    )


# NOTE — the shouldn't-fire side of muting is a WRITE-side property, not a read
# filter. Recall is deliberately channel-agnostic (recall_episodes hardcodes
# channel=None) because muted channels never call store_episode, so nothing muted
# is ever in the corpus to recall. That write gate
# (transcript_service._trigger_episode_extraction → profile_for(channel)
# .extract_episodes) is covered in tests/test_transcript_service.py
# ::test_muted_channel_never_fires_even_above_threshold. A read-side exclusion
# test would assert a filter the design explicitly rejects.


# ══════════════════════════════════════════════════════════════════════════════
# Fact-extraction per-source provenance (E, F)
# ══════════════════════════════════════════════════════════════════════════════

def _facts_for_value(db, value):
    """All data_graph fact rows carrying ``value`` with their source tag."""
    return db.execute(
        "SELECT key, value, source FROM data_graph "
        "WHERE value=? AND deleted_at IS NULL",
        (value,),
    ).fetchall()


def _episode_gist_from_dto(dto) -> str:
    """Isolate the CURRENT episode's gist from a fact-extraction request so the
    fake routes on the episode being processed — never on a recalled neighbour.

    The backlog is ``created_at ASC, id ASC`` ordered; two episodes seeded in the
    same tick tie on created_at and break by UUID, so call-order is
    non-deterministic — routing on gist content is the only stable seam. But an
    already-stored fact can resurface as a neighbour inside a later prompt, so we
    parse ONLY the ``Episode:\\n<gist>`` segment that FactExtractionConfig
    .get_user_prompt emits (the gist sits before the 'Most similar known facts:'
    header), keeping neighbours out of the routing decision.
    """
    body_parts = []
    for m in dto.messages or []:
        body_parts.append(m.get("content", "") if isinstance(m, dict) else str(m))
    body = "\n".join(body_parts)
    marker = "Episode:\n"
    if marker in body:
        after = body.split(marker, 1)[1]
        return after.split("Most similar known facts:", 1)[0]
    return body


def test_fact_extraction_tags_provenance_with_origin_channel(db, store):
    """Facts the worker extracts must carry channel-tagged provenance in
    ``data_graph.source`` — ``fact_extraction:user`` for a user-origin fact and
    ``fact_extraction:dmn`` for a dmn-origin fact — so a fact's source channel is
    recoverable. Before this, every extracted fact carried one bare
    ``fact_extraction`` tag and a dmn-derived fact was indistinguishable from a
    user-stated one.

    Drives the real ``_step_fact_extraction`` over a real two-channel backlog
    through the sanctioned LLM seam. The fake routes on the gist in the assembled
    prompt (backlog order is non-deterministic for same-tick episodes), so each
    episode's op is matched to its own channel. Regression lock: a revert to a
    single channel-less source string reds one of the assertions.
    """
    _seed_episode(db, "User lives in Valletta.", channel="user", emb_index=3)
    _seed_episode(db, "Reflection notes the user prefers Mdina above all.",
                  channel="dmn", emb_index=8)

    def _send(dto):
        # Route on the current episode's gist only (neighbours excluded). The
        # user gist mentions "Valletta"; the dmn gist mentions "Mdina".
        gist = _episode_gist_from_dto(dto)
        if "Valletta" in gist:
            return _ops_response(
                [{"op": "ADD", "key": "residence", "value": "Valletta"}]
            )
        if "Mdina" in gist:
            return _ops_response(
                [{"op": "ADD", "key": "favourite_city", "value": "Mdina"}]
            )
        return _ops_response([{"op": "NOOP"}])

    with _inject_fake_client(_send):
        _worker()._step_fact_extraction()

    user_fact = _facts_for_value(db, "Valletta")
    assert user_fact, "user episode's fact did not land in data_graph"
    assert user_fact[0][2] == "fact_extraction:user", (
        f"user-origin fact must carry channel-tagged 'fact_extraction:user' "
        f"source, got {user_fact[0][2]!r}"
    )

    dmn_fact = _facts_for_value(db, "Mdina")
    assert dmn_fact, "dmn episode's fact did not land in data_graph"
    assert dmn_fact[0][2] == "fact_extraction:dmn", (
        f"dmn-origin fact must carry channel-tagged 'fact_extraction:dmn' "
        f"source, got {dmn_fact[0][2]!r}"
    )


# NOTE — decay-janitor HEAVY-channel protection (anchor G) is covered in its own
# home, tests/test_decay_engine_service.py::TestFossilJanitor, which TKT-926
# updates from the old "non-user → reaped" contract to "non-episode-producing →
# reaped, HEAVY protected". It is not duplicated here.


# ══════════════════════════════════════════════════════════════════════════════
# Consolidation generalization (anchor D / §3.6 — the +293 highest-risk change)
# ══════════════════════════════════════════════════════════════════════════════

def _sqlite_vec_available() -> bool:
    """True when the episodes_vec virtual table is queryable in this env.

    Clustering joins episodes_vec, so consolidation cannot run without sqlite-vec
    (mirrors tests/test_super_episode_pipeline.py::_sqlite_vec_available)."""
    try:
        with _db_service().connection() as conn:
            conn.execute("SELECT COUNT(*) FROM episodes_vec")
        return True
    except Exception:
        return False


def _super_episodes(db, channel):
    """Super-episodes (consolidated_from non-empty) currently in a channel."""
    return db.execute(
        "SELECT id, gist, consolidated_from, level FROM episodes "
        "WHERE channel=? AND deleted_at IS NULL "
        "  AND consolidated_from IS NOT NULL AND consolidated_from != '[]'",
        (channel,),
    ).fetchall()


# ── Count-trigger apex seeding (mirrors tests/test_super_episode_pipeline.py) ───
# The roll-up now fires on a per-channel apex COUNT (>= APEX_COUNT_TRIGGER), then
# clusters via UMAP→HDBSCAN. To make the trigger fire AND produce real clusters we
# seed apex leaves with directly-inserted 256-dim embeddings (the test vec table
# is 256-d; the production embedder emits 768-d, which the vec table rejects on
# width — so the production store_episode path leaves no usable embedding here).
# This is the same seeding idiom test_super_episode_pipeline.py uses, kept
# consistent across the two files.

def _pack(floats: list) -> bytes:
    """Pack a float list into a sqlite-vec binary blob."""
    return struct.pack(f"{len(floats)}f", *floats)


def _topic_vec(axis: int, *, jitter_axis: int, jitter: float = 0.0,
               dim: int = _DIM) -> list:
    """An L2-normalised vector dominated by one ``axis`` with a small component on
    a second ``jitter_axis``. Identical primary axes are mutually cosine ~1.0 (one
    tight topic); distinct axes are orthogonal (separated topics). The per-leaf
    jitter gives a topic intra-cluster spread so a density clusterer sees real
    points instead of one degenerate stacked point."""
    v = [0.0] * dim
    v[axis] = 1.0
    if jitter:
        v[jitter_axis] = jitter
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _insert_embedding(db, episode_id: str, emb: list) -> None:
    """Insert an embedding blob into episodes_vec for ``episode_id``."""
    row = db.execute(
        "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if row:
        db.execute(
            "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
            (row[0], _pack(emb)),
        )
        db.commit()


def _seed_apex_cluster(db, channel, *, count, axis, jitter_axis, salience=7):
    """Seed ``count`` apex leaves on one tight topic (shared primary ``axis``)
    via the PRODUCTION write path (EpisodicService.store_episode), then attach a
    256-d clusterable embedding directly to episodes_vec. Returns the leaf ids."""
    es = EpisodicService(_db_service())
    ids = []
    for i in range(count):
        eid = es.store_episode(
            {"gist": f"{channel} shard {axis}-{i}", "salience": salience,
             "channel": channel},
        )
        _insert_embedding(
            db, eid,
            _topic_vec(axis, jitter_axis=jitter_axis, jitter=0.05 * ((i % 5) + 1)),
        )
        ids.append(eid)
    return ids


def _apex_leaf_count(db, channel):
    """Count apex level-0 leaves (consolidated_into IS NULL, not deleted)."""
    return db.execute(
        "SELECT COUNT(*) FROM episodes "
        "WHERE channel=? AND consolidated_into IS NULL AND deleted_at IS NULL "
        "  AND level=0",
        (channel,),
    ).fetchone()[0]


def test_consolidation_is_per_channel_and_skips_muted_sources(db, store):
    """``_step_consolidate`` must consolidate EACH episode-producing channel into
    its OWN super-episode(s) (channel-scoped, never pooled across channels), and
    must NEVER consolidate a muted channel even when it carries an apex cluster.

    This is the direct feature lock for the consolidation generalization
    (Decision-1: dmn HEAVY + external-agent first-class consolidate; delegates
    muted). Before the generalization, ``find_super_candidates('user')`` was
    hard-coded, so dmn and external-agent apex clusters were never consolidated;
    a regression to that pooled/single-channel form reds these assertions.

    The roll-up now fires on a per-channel apex COUNT (>= APEX_COUNT_TRIGGER=50),
    not a similarity gate, and clusters via UMAP→HDBSCAN. So each consolidating
    channel — user, dmn AND an external-agent channel — is seeded with >= 50 apex
    leaves spanning two well-separated topics (orthogonal axes) so density
    clustering finds coherent groups. The MUTED delegate channel is seeded with an
    identical >= 50-leaf cluster: it would consolidate too if the muting were
    broken, so the count trigger is NOT the thing keeping it out — the source
    profile is.

    Drives the real ``SubconsciousWorker._step_consolidate()`` end to end — the
    per-cluster SuperEpisode encode runs ``MessageProcessor.process`` through the
    sanctioned ``Providers._resolve`` seam. The per-tick summarization budget is
    bounded, so the real multi-tick drain is exercised. The worker stamps the
    cluster's source channel + level=1 onto each super-episode, so a super-episode
    appearing under the wrong channel fails (a) and a pooled implementation would
    consolidate leaves from another channel into one super.
    """
    if not _sqlite_vec_available():
        pytest.skip("sqlite-vec not available — consolidation requires episodes_vec")

    # Two orthogonal topics per channel (25 + 25 = 50 apex leaves) so the >= 50
    # count trigger fires and HDBSCAN finds coherent groups. Distinct axis pairs
    # per channel keep every channel's geometry independent.
    consolidating = {
        "user": (0, 1),
        "dmn": (2, 3),
        "external-agent:bob": (4, 5),
    }
    muted_channel = "delegate:research"
    seeded_leaves: dict[str, set] = {}

    for channel, (axis_a, axis_b) in consolidating.items():
        a = _seed_apex_cluster(db, channel, count=25, axis=axis_a,
                               jitter_axis=axis_a + 200)
        b = _seed_apex_cluster(db, channel, count=25, axis=axis_b,
                               jitter_axis=axis_b + 200)
        seeded_leaves[channel] = set(a) | set(b)
        assert _apex_leaf_count(db, channel) == 50  # at the literal count trigger

    # The muted channel carries an identical >= 50-leaf apex cluster.
    _seed_apex_cluster(db, muted_channel, count=25, axis=6, jitter_axis=206)
    _seed_apex_cluster(db, muted_channel, count=25, axis=7, jitter_axis=207)
    assert _apex_leaf_count(db, muted_channel) == 50

    def _send(_dto):
        # The SuperEpisode encoder expects a JSON object carrying at least a
        # non-empty 'gist'; the worker overwrites 'channel'/'level' itself.
        return _text_response(json.dumps({
            "gist": "Consolidated reflection of the shards.",
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
            "has_open_loop": False,
        }))

    with _inject_fake_client(_send):
        _worker()._step_consolidate()

    # (a) Every episode-producing channel consolidated its OWN apex cluster into
    # at least one level-1 super-episode (channel-scoped — never pooled across
    # channels). Each super-episode consolidates ONLY that channel's own leaves,
    # and every reparented leaf on the channel is one of that channel's seeds and
    # back-points at one of that channel's supers. A pooled / single-channel
    # implementation (the pre-generalization ``find_super_candidates('user')``)
    # reds this for dmn and external-agent.
    for channel in consolidating:
        supers = _super_episodes(db, channel)
        assert supers, (
            f"channel {channel!r} reached the count trigger but produced no "
            f"super-episode — the count-triggered roll-up did not run for it"
        )

        super_ids = set()
        for sid, _gist, cfrom_json, level in supers:
            super_ids.add(sid)
            assert level == 1, (
                f"channel {channel!r} super-episode must be stamped level=1 "
                f"(decays at the wrong tau otherwise), got {level!r}"
            )
            children = set(json.loads(cfrom_json))
            assert children, "super-episode carries an empty consolidated_from"
            assert children <= seeded_leaves[channel], (
                f"channel {channel!r} super-episode consolidated a leaf from "
                f"another channel — consolidation must be channel-scoped; "
                f"from={children!r}"
            )

        # Every reparented leaf on this channel is one of this channel's seeds and
        # back-points at one of this channel's supers (no cross-channel leakage).
        reparented = db.execute(
            "SELECT id, consolidated_into FROM episodes "
            "WHERE channel=? AND consolidated_into IS NOT NULL",
            (channel,),
        ).fetchall()
        assert reparented, f"channel {channel!r} consolidated no leaves"
        for leaf_id, into in reparented:
            assert leaf_id in seeded_leaves[channel]
            assert into in super_ids, (
                f"leaf {leaf_id} on channel {channel!r} back-points at a super "
                f"that is not one of this channel's — consolidation leaked across "
                f"channels"
            )

    # (b) The muted channel — despite carrying an identical >= 50-leaf apex
    # cluster — is NEVER consolidated. No super-episode, no leaf re-parented. It
    # is the source profile (muted), not the count trigger, that keeps it out:
    # the cluster is above the trigger, so the only thing excluding it is muting.
    assert _super_episodes(db, muted_channel) == [], (
        "a muted channel must never be consolidated, even with an apex cluster "
        "above the count trigger"
    )
    muted_consolidated = db.execute(
        "SELECT COUNT(*) FROM episodes WHERE channel=? AND consolidated_into IS NOT NULL",
        (muted_channel,),
    ).fetchone()[0]
    assert muted_consolidated == 0, (
        "no leaf on a muted channel may be re-parented into a super-episode"
    )
    assert _apex_leaf_count(db, muted_channel) == 50, (
        "all 50 muted-channel leaves must remain apex (untouched)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Scheduled two-stage (Decision 2 / §4)
# ══════════════════════════════════════════════════════════════════════════════

def _join_threads(prefix, timeout=15.0):
    """Join every live background thread whose name starts with ``prefix``.

    Joining by name synchronises on the real production threads — no sleep, no
    poll of a guessed duration. The scheduler fires a prompt on a daemon thread
    named ``scheduled-work-<item_id>`` (scheduler_service.py:353); that thread
    spawns the Stage-2 user turn synchronously on a ``chat-<id>`` thread
    (api/chat.py:314) before it returns, so joining ``scheduled-work-*`` first
    guarantees the ``chat-*`` thread already exists when we enumerate for it."""
    for t in threading.enumerate():
        if t.name.startswith(prefix) and t is not threading.current_thread():
            t.join(timeout)


def _transcript_rows(db, channel, role=None):
    sql = "SELECT content FROM transcript WHERE channel=?"
    params = [channel]
    if role is not None:
        sql += " AND role=?"
        params.append(role)
    sql += " ORDER BY id ASC"
    return [r[0] for r in db.execute(sql, params).fetchall()]


def test_scheduled_poll_claims_item_then_surfaces_two_stage_on_user(db, store):
    """Driving the FULL scheduler poll path ``_poll_and_fire`` over a genuine due
    prompt row must:

      (claim) commit the claim transaction cleanly — the row ends ``status='fired'``
              (the atomicity defect that earlier surfaced because Stage-1 ran SYNC
              inside the open poll transaction; the fix moved the work onto a
              ``scheduled-work-<id>`` daemon thread so ``_fire_item`` returns
              immediately and the poll txn commits the claim before any LLM work).
      (Stage 1) run the work loop on the MUTED ``scheduled`` channel with the
              instruction persisted there, so a fired task is recoverable and its
              own output never leaks onto the user channel as input.
      (Stage 2) surface the worked result to the user as an assistant message on
              the ``user`` channel — the channel whose source profile is
              ``extract_episodes`` (so it is encoded as ordinary user output),
              unlike the muted ``scheduled`` channel — with NO synthetic user
              input row (hidden_input=True).

    This drives the real production entry point ``_poll_and_fire()`` (not
    ``_fire_item`` directly), so the claim/commit transaction the review flagged
    is exercised for real. The LLM seam returns a no-tool reply for both stages,
    so each ACT loop ends on its first turn deterministically. The two real
    daemon threads are joined by name — ``scheduled-work-*`` (Stage 1 + dispatch)
    first, then ``chat-*`` (Stage 2) — for synchronisation.
    """
    from services import scheduler_service
    from services.source_profiles import profile_for
    from services.time_utils import utc_now

    item_id = "sched-poll-1"
    instruction = "summarise the user's harbour cafe notes"
    result_text = "Here is your harbour cafe summary: espresso at dawn."

    # A genuine due prompt row — exactly what the poll SELECT picks up:
    # status='pending', due_at in the past, is_prompt=1, item_type='prompt'.
    due_at = (utc_now() - timedelta(minutes=1)).isoformat()
    db.execute(
        "INSERT INTO scheduled_items (id, item_type, message, due_at, status, is_prompt) "
        "VALUES (?, 'prompt', ?, ?, 'pending', 1)",
        (item_id, instruction, due_at),
    )
    db.commit()

    def _send(_dto):
        # Both stages finish in one turn with a plain reply. Stage-1's text is the
        # scheduled result the Stage-2 user turn relays; Stage-2 echoes it back.
        return _text_response(result_text)

    with _inject_fake_client(_send):
        scheduler_service._poll_and_fire()
        # Stage 1 (work loop + dispatch) runs on scheduled-work-<id>; it spawns
        # the Stage-2 chat-<id> thread synchronously before returning, so join it
        # first, then the user-turn thread.
        _join_threads("scheduled-work")
        _join_threads("chat-")

    # (claim) The poll's claim transaction committed cleanly: the due row is
    # marked 'fired'. A non-atomic poll (Stage-1 flushing the in-progress UPDATE
    # on the shared connection) would leave it 'pending' or 'failed'.
    status = db.execute(
        "SELECT status FROM scheduled_items WHERE id=?", (item_id,)
    ).fetchone()[0]
    assert status == "fired", (
        f"the poll must atomically claim and mark the due prompt 'fired'; got "
        f"{status!r}"
    )

    # (Stage 1) the work loop ran on the muted 'scheduled' channel and persisted
    # the instruction there (recoverable). That channel is muted, so its rows
    # never become episodes.
    assert not profile_for("scheduled").extract_episodes, (
        "test premise: the scheduled work channel must be muted"
    )
    scheduled_rows = _transcript_rows(db, "scheduled")
    assert any(instruction in c for c in scheduled_rows), (
        "the scheduled work loop must persist the instruction on channel="
        f"'scheduled'; found scheduled rows: {scheduled_rows!r}"
    )

    # (Stage 2) the result surfaced to the user as an assistant message on the
    # 'user' channel — the encode-eligible channel — and the return hop wrote no
    # synthetic user input row (hidden_input=True).
    assert profile_for("user").extract_episodes, (
        "test premise: the user channel must be episode-producing (encoded)"
    )
    user_assistant = _transcript_rows(db, "user", role="assistant")
    assert any(result_text in c for c in user_assistant), (
        "the scheduled return hop must surface the worked result as an assistant "
        f"message on channel='user'; found user rows: "
        f"{_transcript_rows(db, 'user')!r}"
    )
    assert not _transcript_rows(db, "user", role="user"), (
        "the return hop must not write a synthetic user input row "
        "(hidden_input=True) — the scheduled result is the model's reply, not a "
        "faked user message"
    )
