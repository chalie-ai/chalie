import sqlite3
from pathlib import Path

import pytest

from services.database_service import DatabaseService
from services.schema_convergence_service import SchemaConvergenceService
from services.time_utils import utc_now


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path, name: str = "redesign.db") -> DatabaseService:
    return DatabaseService(str(tmp_path / name))


def _converge_and_backfill(db: DatabaseService) -> None:
    svc = SchemaConvergenceService(db, embedding_dimensions=256)
    svc.converge()
    svc.backfill_redesign_columns()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


# Legacy data_graph DDL: pre-redesign — carries the CHECK constraint and lacks
# valid_from / valid_to. Rebuilding the table this way forces the convergence
# constraint-strip path (and its by-name column copy) to run on next boot.
_LEGACY_DATA_GRAPH_DDL = """
CREATE TABLE data_graph (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT NOT NULL CHECK(kind IN ('user_specific','system','misc','moment')),
    key               TEXT NOT NULL,
    value             TEXT,
    storage_strength  REAL NOT NULL DEFAULT 0.5,
    retrieval_weight  REAL NOT NULL DEFAULT 1.0,
    salience_score    REAL NOT NULL DEFAULT 0.0,
    evidence_count    INTEGER NOT NULL DEFAULT 1,
    first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at  TEXT,
    source            TEXT,
    deleted_at        TEXT,
    active            INTEGER NOT NULL DEFAULT 1,
    search_queries    TEXT DEFAULT NULL
)
"""

_RADIUS_COLUMNS = {
    "input_radius", "narrow_factor", "expand_factor", "adaptive_shrink_divisor",
    "effective_radius", "vector_candidates", "fts_candidates", "survivors_after_radius",
}


def _build_legacy_db(tmp_path: Path) -> tuple[DatabaseService, dict[str, str]]:
    db = _make_db(tmp_path)
    # First converge to get the full, correct schema for every OTHER table; then
    # we mutate only the three tables the redesign touches back to legacy shape.
    SchemaConvergenceService(db, embedding_dimensions=256).converge()

    # Distinct, fixed timestamps so derived-value assertions are exact.
    now = utc_now()
    ep_created = now.replace(microsecond=0).isoformat()
    # A distinct, fixed access time so the COALESCE(last_accessed_at, created_at)
    # assertion is unambiguous (clearly different from created_at).
    ep_accessed = "2026-04-20T09:15:00+00:00"
    old_first_seen = "2026-01-01T00:00:00+00:00"
    new_first_seen = "2026-03-15T12:30:00+00:00"
    live_first_seen = "2026-02-02T08:00:00+00:00"

    seeded = {
        "old_first_seen": old_first_seen,
        "new_first_seen": new_first_seen,
        "live_first_seen": live_first_seen,
    }

    with db.connection() as conn:
        # ── episodes → legacy shape (drop the 4 redesign columns) ──────────────
        conn.execute("ALTER TABLE episodes RENAME TO episodes_legacy_tmp")
        conn.execute(
            """
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                gist TEXT NOT NULL,
                salience INTEGER NOT NULL CHECK (salience BETWEEN 1 AND 10),
                channel TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                deleted_at TEXT,
                transcript_ids TEXT DEFAULT '[]',
                transcript_id_start INTEGER,
                transcript_id_end INTEGER,
                emotional_valence REAL,
                emotional_arousal REAL,
                consolidated_from TEXT DEFAULT '[]',
                consolidated_into TEXT,
                storage_strength REAL DEFAULT 1.0,
                retrieval_weight REAL DEFAULT 1.0,
                location_lat REAL,
                location_lon REAL,
                location_name TEXT
            )
            """
        )
        conn.execute("DROP TABLE episodes_legacy_tmp")

        # ep_with_access: has last_accessed_at → last_relevant_at must COALESCE to it.
        conn.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_accessed_at) "
            "VALUES ('ep-access', 'user mentioned moving to Valletta', 7, 'web', ?, ?)",
            (ep_created, ep_accessed),
        )
        # ep_no_access: last_accessed_at NULL → last_relevant_at must COALESCE to created_at.
        conn.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_accessed_at) "
            "VALUES ('ep-noaccess', 'discussed weekend hiking plans', 4, 'web', ?, NULL)",
            (ep_created,),
        )
        seeded["ep_created"] = ep_created
        seeded["ep_accessed"] = ep_accessed

        # ── data_graph → legacy shape (CHECK constraint, no valid_from/valid_to) ─
        # The fresh converge already stripped CHECK, so rebuild WITH it to make the
        # constraint-strip + by-name-copy path run on the next boot.
        conn.execute("DROP TABLE IF EXISTS data_graph")
        conn.execute(_LEGACY_DATA_GRAPH_DDL)

        # Superseded fact (old), its superseder (new/live), and an unrelated live fact.
        conn.execute(
            "INSERT INTO data_graph (id, kind, key, value, first_seen_at, active) "
            "VALUES (1, 'user_specific', 'home_city', 'Swieqi', ?, 0)",
            (old_first_seen,),
        )
        conn.execute(
            "INSERT INTO data_graph (id, kind, key, value, first_seen_at, active) "
            "VALUES (2, 'user_specific', 'home_city', 'Valletta', ?, 1)",
            (new_first_seen,),
        )
        conn.execute(
            "INSERT INTO data_graph (id, kind, key, value, first_seen_at, active) "
            "VALUES (3, 'user_specific', 'favourite_food', 'pastizzi', ?, 1)",
            (live_first_seen,),
        )
        # superseded_by edge: old (1) → new (2). Backfill must set 1.valid_to = 2.first_seen_at.
        conn.execute(
            "INSERT INTO data_graph_edges (from_id, to_id, edge_type) VALUES (1, 2, 'superseded_by')"
        )

    return db, seeded


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSchemaRedesignConvergence:

    def test_old_shape_db_self_heals_to_redesign_shape(self, tmp_path: Path) -> None:
        """A pre-redesign DB must self-heal to redesign shape after converge() + backfill()."""
        db, seeded = _build_legacy_db(tmp_path)

        # Pre-condition snapshot: legacy shape + row data, to prove no loss later.
        with db.connection() as conn:
            assert "level" not in _columns(conn, "episodes")
            assert "valid_from" not in _columns(conn, "data_graph")
            dg_ddl_before = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='data_graph'"
            ).fetchone()[0]
            assert "CHECK" in dg_ddl_before.upper()
            episodes_before = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            dg_before = conn.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0]

        # ── The production boot path ──────────────────────────────────────────
        _converge_and_backfill(db)

        with db.connection() as conn:
            # 1. All 6 new columns exist.
            ep_cols = _columns(conn, "episodes")
            for c in ("level", "last_relevant_at", "tombstoned_at", "facts_extracted_at"):
                assert c in ep_cols, f"episodes.{c} missing after converge"
            dg_cols = _columns(conn, "data_graph")
            assert "valid_from" in dg_cols and "valid_to" in dg_cols

            # 2. episodes.level backfilled to leaf (0); both rows.
            levels = {r[0] for r in conn.execute("SELECT level FROM episodes")}
            assert levels == {0}, f"expected all level=0, got {levels}"

            # 3. last_relevant_at == COALESCE(last_accessed_at, created_at) per-row.
            row_access = conn.execute(
                "SELECT last_relevant_at, last_accessed_at, created_at FROM episodes WHERE id='ep-access'"
            ).fetchone()
            assert row_access[0] == seeded["ep_accessed"], (
                "ep-access last_relevant_at should equal its last_accessed_at"
            )
            row_noaccess = conn.execute(
                "SELECT last_relevant_at, last_accessed_at, created_at FROM episodes WHERE id='ep-noaccess'"
            ).fetchone()
            assert row_noaccess[1] is None
            assert row_noaccess[0] == seeded["ep_created"], (
                "ep-noaccess last_relevant_at should fall back to created_at"
            )

            # 4. valid_from == first_seen_at for every data_graph row.
            for fact_id, expected in (
                (1, seeded["old_first_seen"]),
                (2, seeded["new_first_seen"]),
                (3, seeded["live_first_seen"]),
            ):
                vf = conn.execute(
                    "SELECT valid_from FROM data_graph WHERE id=?", (fact_id,)
                ).fetchone()[0]
                assert vf == expected, f"data_graph[{fact_id}].valid_from mismatch"

            # 5. Bi-temporal invariant: superseded row's valid_to == superseder's valid_from.
            old_valid_to = conn.execute(
                "SELECT valid_to FROM data_graph WHERE id=1"
            ).fetchone()[0]
            assert old_valid_to == seeded["new_first_seen"], (
                "old.valid_to must equal new.first_seen_at (bi-temporal supersession)"
            )

            # 6. Live rows keep valid_to NULL.
            for fact_id in (2, 3):
                assert conn.execute(
                    "SELECT valid_to FROM data_graph WHERE id=?", (fact_id,)
                ).fetchone()[0] is None, f"live fact {fact_id} valid_to should be NULL"

            # 7. CHECK constraint gone.
            dg_ddl_after = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='data_graph'"
            ).fetchone()[0]
            assert "CHECK" not in dg_ddl_after.upper(), "legacy kind CHECK constraint survived"

            # 8. No data loss — counts and spot-check values survive the rebuild.
            assert conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == episodes_before
            assert conn.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0] == dg_before
            assert conn.execute(
                "SELECT value FROM data_graph WHERE id=1"
            ).fetchone()[0] == "Swieqi"
            assert conn.execute(
                "SELECT value FROM data_graph WHERE id=2"
            ).fetchone()[0] == "Valletta"
            assert conn.execute(
                "SELECT gist FROM episodes WHERE id='ep-access'"
            ).fetchone()[0] == "user mentioned moving to Valletta"

    def test_backfill_is_idempotent(self, tmp_path: Path) -> None:
        """Running converge() + backfill a SECOND time changes no derived value —
        the backfill only touches still-NULL targets, so it must no-op on rerun."""
        db, _ = _build_legacy_db(tmp_path)
        _converge_and_backfill(db)

        def _snapshot(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
            eps = conn.execute(
                "SELECT id, level, last_relevant_at, tombstoned_at, facts_extracted_at "
                "FROM episodes ORDER BY id"
            ).fetchall()
            dg = conn.execute(
                "SELECT id, valid_from, valid_to FROM data_graph ORDER BY id"
            ).fetchall()
            return [tuple(r) for r in eps], [tuple(r) for r in dg]

        with db.connection() as conn:
            first = _snapshot(conn)

        # Second full boot pass.
        _converge_and_backfill(db)

        with db.connection() as conn:
            second = _snapshot(conn)

        assert first == second, "second converge+backfill churned redesign column values"

    def test_fresh_db_gets_redesign_shape_and_backfill_no_ops(self, tmp_path: Path) -> None:
        """A fresh DB converges directly to the new shape; backfill runs cleanly
        and seeds level=0 with no rows present (no error, no churn)."""
        db = _make_db(tmp_path, name="fresh.db")
        _converge_and_backfill(db)

        with db.connection() as conn:
            ep_cols = _columns(conn, "episodes")
            assert {"level", "last_relevant_at", "tombstoned_at", "facts_extracted_at"} <= ep_cols
            dg_cols = _columns(conn, "data_graph")
            assert {"valid_from", "valid_to"} <= dg_cols
            mrl_cols = _columns(conn, "memory_recall_log")
            assert "floor_cut_count" in mrl_cols
            assert not (_RADIUS_COLUMNS & mrl_cols)
            # The live-fact partial index from the redesign exists.
            idx = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )}
            assert "idx_data_graph_live" in idx
            # No episodes seeded on a fresh DB → backfill touched nothing.
            assert conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
