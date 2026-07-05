"""Feature tests for DecayEngineService — episodic-memory redesign (ticket B).

No mocks: episodes, episodes_fts and episodes_vec are all real tables.

Ticket B changes:
  * Absolute law ``rw = (salience/10) × exp(-Δt/τ)`` replaces old compounding
    law that floored ~98% of the corpus at 0.01 and was NOT idempotent.
  * ``retrieval_decay_exponent`` removed — constructor test asserts attribute gone.
  * Zero-mock rule: no patch.object call-log or DB-failure patches.
"""

import math
import sqlite3
import struct
from datetime import timedelta
from typing import cast

import pytest

from services.database_service import get_shared_db_service
from services.decay_engine_service import (
    DecayEngineService,
    _LEAF_DELETE_AGE_DAYS,
    _LEAF_DELETE_SALIENCE_MAX,
    _TOMBSTONE_DELETE_AFTER_DAYS,
    _JANITOR_FOSSIL_AGE_DAYS,
)
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


_EMBED_DIM = 256  # matches conftest convergence(embedding_dimensions=256)


# ── seeding helpers (real rows, no mocks) ─────────────────────────────────────


def _insert_episode(db: sqlite3.Connection, episode_id: str, *, salience: int, channel: str = "user", level: int = 0,
                    retrieval_weight: float, last_relevant_at: str, tombstoned_at: str | None = None,
                    consolidated_into: str | None = None, created_at: str | None = None, with_shadows: bool = True) -> int:
    created = created_at or last_relevant_at
    db.execute(
        "INSERT INTO episodes "
        "(id, gist, salience, channel, created_at, retrieval_weight, level, "
        " last_relevant_at, tombstoned_at, consolidated_into) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (episode_id, f"gist {episode_id}", salience, channel, created,
         retrieval_weight, level, last_relevant_at, tombstoned_at,
         consolidated_into),
    )
    rowid = cast(int, db.execute(
        "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()[0])
    if with_shadows:
        db.execute(
            "INSERT INTO episodes_fts(rowid, gist) VALUES (?, ?)",
            (rowid, f"gist {episode_id}"),
        )
        blob = struct.pack(f"{_EMBED_DIM}f", *([0.05] * _EMBED_DIM))
        db.execute(
            "INSERT INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
    db.commit()
    return rowid


def _iso_ago(**kwargs: int) -> str:
    return (utc_now() - timedelta(**kwargs)).isoformat()


def _legacy_ago(**kwargs: int) -> str:
    return (utc_now() - timedelta(**kwargs)).strftime("%Y-%m-%d %H:%M:%S")


def _fresh_conn() -> sqlite3.Connection:
    return get_shared_db_service()._get_connection()


def _rw(conn: sqlite3.Connection, episode_id: str) -> float | None:
    row = conn.execute(
        "SELECT retrieval_weight FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    return row[0] if row else None


def _exists(conn: sqlite3.Connection, episode_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone() is not None


class TestAbsoluteDecay:
    def test_floored_corpus_recovers_on_first_tick(self, db: sqlite3.Connection) -> None:
        """A salience-7 episode whose last_relevant_at is ~1 day old should land near
        0.7 × exp(-24/336) ≈ 0.65 via the absolute law — not stuck at old 0.01 floor.
        """
        _insert_episode(db, "ep-floored", salience=7, retrieval_weight=0.01,
                        last_relevant_at=_iso_ago(hours=24))

        updated = DecayEngineService()._decay_episodic()

        conn = _fresh_conn()
        new_rw = cast(float, _rw(conn, "ep-floored"))
        assert updated == 1
        assert new_rw > 0.5, f"floored weight should recover, got {new_rw}"
        # Sanity: it matches the absolute formula, not a multiply-down.
        expected = 0.7 * math.exp(-24.0 / (14 * 24))
        assert new_rw == pytest.approx(expected, abs=0.02)

    def test_legacy_space_separated_anchor_is_parsed(self, db: sqlite3.Connection) -> None:
        """Backfilled 'YYYY-MM-DD HH:MM:SS' anchor must parse correctly; an old
        salience-8 leaf should decay below its ceiling but stay positive.
        """
        _insert_episode(db, "ep-legacy", salience=8, retrieval_weight=0.01,
                        last_relevant_at=_legacy_ago(days=60))

        DecayEngineService()._decay_episodic()

        conn = _fresh_conn()
        rw = _rw(conn, "ep-legacy")
        # 60 days vs 14-day τ → 0.8 × exp(-1440/336) ≈ 0.011, still > 0 (parsed OK).
        assert rw is not None and 0.0 < rw < 0.1

    def test_tick_is_idempotent(self, db: sqlite3.Connection) -> None:
        """A settled corpus produces zero further change on a repeat tick.
        """
        _insert_episode(db, "ep-idem", salience=6, retrieval_weight=0.01,
                        last_relevant_at=_iso_ago(hours=10))

        DecayEngineService()._decay_episodic()
        after_first = _rw(_fresh_conn(), "ep-idem")

        second_updated = DecayEngineService()._decay_episodic()
        after_second = _rw(_fresh_conn(), "ep-idem")

        assert second_updated == 0, "settled corpus must not be re-written"
        assert after_second == after_first

    def test_tombstone_tau_decays_faster_than_leaf(self, db: sqlite3.Connection) -> None:
        """At equal age/salience, a tombstoned row decays below a live leaf.

        τ_tombstoned (7d) << τ_leaf (14d), so the tombstoned row's recomputed
        weight must be strictly lower.
        """
        anchor = _iso_ago(hours=48)
        _insert_episode(db, "ep-leaf", salience=6, retrieval_weight=0.01,
                        last_relevant_at=anchor)
        _insert_episode(db, "ep-tomb", salience=6, retrieval_weight=0.01,
                        last_relevant_at=anchor, tombstoned_at=_iso_ago(hours=1))

        DecayEngineService()._decay_episodic()

        conn = _fresh_conn()
        assert cast(float, _rw(conn, "ep-tomb")) < cast(float, _rw(conn, "ep-leaf"))


class TestHardDeletion:
    """Tombstone-window and weak-leaf deletion, with vec/fts cleanup."""

    def _shadow_counts(self, conn: sqlite3.Connection, rowid: int) -> tuple[int, int]:
        fts = conn.execute(
            "SELECT COUNT(*) FROM episodes_fts WHERE rowid = ?", (rowid,)
        ).fetchone()[0]
        vec = conn.execute(
            "SELECT COUNT(*) FROM episodes_vec WHERE rowid = ?", (rowid,)
        ).fetchone()[0]
        return fts, vec

    def test_tombstoned_beyond_window_is_deleted_with_shadows(self, db: sqlite3.Connection) -> None:
        """A row tombstoned > 30 days ago is hard-deleted; its fts+vec rows go too."""
        rowid = _insert_episode(
            db, "ep-old-tomb", salience=5, retrieval_weight=0.2,
            last_relevant_at=_iso_ago(days=40),
            tombstoned_at=_iso_ago(days=_TOMBSTONE_DELETE_AFTER_DAYS + 5),
        )
        # Precondition: shadows exist.
        assert self._shadow_counts(db, rowid) == (1, 1)

        deleted = DecayEngineService()._delete_expired_episodes()

        conn = _fresh_conn()
        assert deleted == 1
        assert not _exists(conn, "ep-old-tomb")
        assert self._shadow_counts(conn, rowid) == (0, 0), \
            "fts/vec shadow rows must be cleaned up on hard-delete"

    def test_recently_tombstoned_survives(self, db: sqlite3.Connection) -> None:
        """A row tombstoned only 5 days ago is inside the 30-day window — kept."""
        _insert_episode(db, "ep-young-tomb", salience=5, retrieval_weight=0.2,
                        last_relevant_at=_iso_ago(days=10),
                        tombstoned_at=_iso_ago(days=5))

        deleted = DecayEngineService()._delete_expired_episodes()

        assert deleted == 0
        assert _exists(_fresh_conn(), "ep-young-tomb")

    def test_weak_old_junk_leaf_deleted_but_near_misses_survive(self, db: sqlite3.Connection) -> None:
        """Weak+old+junk leaf is deleted; one-axis near-misses each survive.

        Delete branch requires ALL of: rw < 0.05, age > 90d, salience <= 3,
        level 0, not consolidated. Three near-miss rows each violate exactly
        one condition and must be kept — this pins both the should-fire and
        shouldn't-fire sides.
        """
        old = _iso_ago(days=_LEAF_DELETE_AGE_DAYS + 10)
        recent = _iso_ago(days=_LEAF_DELETE_AGE_DAYS - 10)
        # Target: satisfies every condition.
        _insert_episode(db, "junk", salience=2, retrieval_weight=0.02,
                        last_relevant_at=old)
        # Near-miss A: salience 4 (> junk threshold of 3).
        _insert_episode(db, "nm-salience", salience=_LEAF_DELETE_SALIENCE_MAX + 1,
                        retrieval_weight=0.02, last_relevant_at=old)
        # Near-miss B: too young (age 80d < 90d).
        _insert_episode(db, "nm-young", salience=2, retrieval_weight=0.02,
                        last_relevant_at=recent)
        # Near-miss C: weight above the floor.
        _insert_episode(db, "nm-strong", salience=2, retrieval_weight=0.20,
                        last_relevant_at=old)

        deleted = DecayEngineService()._delete_expired_episodes()

        conn = _fresh_conn()
        assert deleted == 1
        assert not _exists(conn, "junk")
        assert _exists(conn, "nm-salience")
        assert _exists(conn, "nm-young")
        assert _exists(conn, "nm-strong")

    def test_consolidated_leaf_is_never_junk_deleted(self, db: sqlite3.Connection) -> None:
        """A weak old low-salience leaf that was rolled up is preserved.

        consolidated_into IS NOT NULL means the content lives on in a parent —
        the junk branch explicitly excludes it.
        """
        _insert_episode(db, "ep-rolled", salience=2, retrieval_weight=0.02,
                        last_relevant_at=_iso_ago(days=_LEAF_DELETE_AGE_DAYS + 10),
                        consolidated_into="super-1")

        deleted = DecayEngineService()._delete_expired_episodes()

        assert deleted == 0
        assert _exists(_fresh_conn(), "ep-rolled")


class TestFossilJanitor:
    """Stranded apex leaves on NON-episode-producing channels get tombstoned to
    enter deletion — but HEAVY (episode-producing) channels are protected.

     spec change: the janitor's protection moved from a single
    ``channel != 'user'`` filter to the per-source allowlist
    (``janitor_protected_sql()`` = every channel whose profile sets
    ``extract_episodes``). dmn is HEAVY and never consolidates, so its leaves are
    permanently apex; reaping them after the fossil window destroyed durable
    background memory. The reaping behaviour itself is unchanged — it just now
    targets the genuinely muted / absent-profile channels instead of dmn.
    """

    def test_old_muted_channel_leaf_is_tombstoned(self, db: sqlite3.Connection) -> None:
        """An apex leaf older than the fossil window on a MUTED channel
        (skills_building) and one on an ABSENT-profile channel (discord, defaults
        to muted) both get tombstoned_at set. Replaces the old dmn fixture, which
         now protects."""
        _insert_episode(db, "ep-muted", salience=5, retrieval_weight=0.3,
                        channel="skills_building",
                        last_relevant_at=_iso_ago(days=_JANITOR_FOSSIL_AGE_DAYS + 3))
        _insert_episode(db, "ep-legacy", salience=5, retrieval_weight=0.3,
                        channel="discord",
                        last_relevant_at=_iso_ago(days=_JANITOR_FOSSIL_AGE_DAYS + 3))

        tombstoned = DecayEngineService()._janitor_fossil_episodes()

        conn = _fresh_conn()
        assert tombstoned == 2
        assert conn.execute(
            "SELECT tombstoned_at FROM episodes WHERE id = 'ep-muted'"
        ).fetchone()[0] is not None
        assert conn.execute(
            "SELECT tombstoned_at FROM episodes WHERE id = 'ep-legacy'"
        ).fetchone()[0] is not None

    def test_heavy_channel_and_young_fossils_untouched(self, db: sqlite3.Connection) -> None:
        """The shouldn't-fire side. The janitor leaves untouched:
          * every HEAVY (episode-producing) channel's old apex leaf — user, dmn
            AND external-agent:* — because  protects the full allowlist
            (dmn would otherwise be reaped wholesale: it never consolidates), and
          * any young leaf regardless of channel.
        """
        _insert_episode(db, "ep-user-old", salience=5, retrieval_weight=0.3,
                        channel="user",
                        last_relevant_at=_iso_ago(days=_JANITOR_FOSSIL_AGE_DAYS + 3))
        _insert_episode(db, "ep-dmn-old", salience=5, retrieval_weight=0.3,
                        channel="dmn",
                        last_relevant_at=_iso_ago(days=_JANITOR_FOSSIL_AGE_DAYS + 3))
        _insert_episode(db, "ep-ext-old", salience=5, retrieval_weight=0.3,
                        channel="external-agent:bob",
                        last_relevant_at=_iso_ago(days=_JANITOR_FOSSIL_AGE_DAYS + 3))
        _insert_episode(db, "ep-muted-young", salience=5, retrieval_weight=0.3,
                        channel="skills_building",
                        last_relevant_at=_iso_ago(days=1))

        tombstoned = DecayEngineService()._janitor_fossil_episodes()

        conn = _fresh_conn()
        assert tombstoned == 0
        for ep_id in ("ep-user-old", "ep-dmn-old", "ep-ext-old", "ep-muted-young"):
            assert conn.execute(
                "SELECT tombstoned_at FROM episodes WHERE id = ?", (ep_id,)
            ).fetchone()[0] is None, f"{ep_id} must not be tombstoned"


class TestPureRead:
    """Retrieval must change zero rows (relevance advances only on writes)."""

    def test_get_episode_by_id_does_not_mutate_the_row(self, db: sqlite3.Connection) -> None:
        """EpisodicService.get_episode_by_id is a pure read.

        The old _update_activation_score bumped access_count / last_accessed_at /
        storage_strength / retrieval_weight on every fetch. That is deleted —
        the full episodes table must be byte-identical before and after a fetch.
        """
        from services.episodic_service import EpisodicService

        _insert_episode(db, "ep-read", salience=5, retrieval_weight=0.42,
                        last_relevant_at=_iso_ago(hours=5))
        before = db.execute(
            "SELECT id, retrieval_weight, access_count, last_accessed_at, "
            "last_relevant_at FROM episodes ORDER BY id"
        ).fetchall()

        ep = EpisodicService().get_episode_by_id("ep-read")
        assert ep is not None and ep["id"] == "ep-read"

        after = db.execute(
            "SELECT id, retrieval_weight, access_count, last_accessed_at, "
            "last_relevant_at FROM episodes ORDER BY id"
        ).fetchall()
        assert after == before, "a read must not mutate any episode column"
