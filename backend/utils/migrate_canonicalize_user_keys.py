"""
Backfill migration: embed each active user_specific key, look it up against concept_lut.sqlite,
and canonicalize keys that differ. Rule-aware merge when two raw keys map to the same target:
  temporal   → newer row stays active=1; older demoted to active=0 with halved weight + supersedes edges.
  coexist    → same value merges evidence and hard-deletes; different value rewrites key, both stay active.
  immutable  → same value merges evidence and hard-deletes; different value logs and skips.
Idempotent / safe to re-run.

Run from backend/:
    python -m utils.migrate_canonicalize_user_keys [--dry-run]
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import cast
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.data_graph_service import (
    KIND_USER_SPECIFIC,
    _CONCEPT_LUT_PATH,
    _CONCEPT_LUT_THRESHOLD,
    _l2_dist_to_cosine,
)
from services.database_service import DatabaseService, get_shared_db_service
from services.embedding_service import EmbeddingService
from services.embedding_utils import pack_embedding

_BATCH_SIZE = 32
_TEMPORAL_DEMOTION_FACTOR = 0.5


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension('vec0')


def _open_lut() -> sqlite3.Connection:
    if not Path(_CONCEPT_LUT_PATH).exists():
        raise FileNotFoundError(f"concept_lut.sqlite not found at {_CONCEPT_LUT_PATH}")
    conn = sqlite3.connect(_CONCEPT_LUT_PATH)
    _load_sqlite_vec(conn)
    return conn


def _lut_lookup(lut_conn: sqlite3.Connection, blob: bytes) -> tuple[str, str] | None:
    hits = lut_conn.execute(
        "SELECT rowid, distance FROM lut_embeddings WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
        (blob,),
    ).fetchall()
    if not hits:
        return None
    rowid, distance = hits[0]
    cos = _l2_dist_to_cosine(distance)
    if cos < _CONCEPT_LUT_THRESHOLD:
        return None
    row = lut_conn.execute(
        "SELECT canonical_key, rule FROM lut_concepts WHERE id = ?", (rowid,)
    ).fetchone()
    return (row[0], row[1]) if row else None


def _update_fts(
    conn: sqlite3.Connection,
    rowid: int,
    old_key: str,
    old_value: str,
    old_kind: str,
    old_sq: str,
    new_key: str,
) -> None:
    try:
        conn.execute(
            "INSERT INTO data_graph_fts(data_graph_fts, rowid, key, value, kind, search_queries) "
            "VALUES('delete', ?, ?, ?, ?, ?)",
            (rowid, old_key, old_value or '', old_kind, old_sq or ''),
        )
    except Exception:
        try:
            conn.execute("DELETE FROM data_graph_fts WHERE rowid = ?", (rowid,))
        except Exception:
            pass
    try:
        conn.execute(
            "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) VALUES (?, ?, ?, ?, ?)",
            (rowid, new_key, old_value or '', old_kind, old_sq or ''),
        )
    except Exception as e:
        print(f"  WARN: FTS insert failed for rowid={rowid}: {e}")


def _hard_delete_row(conn: sqlite3.Connection, row_id: int) -> None:
    try:
        conn.execute(
            "INSERT INTO data_graph_fts(data_graph_fts, rowid, key, value, kind, search_queries) "
            "VALUES('delete', ?, '', '', '', '')",
            (row_id,),
        )
    except Exception:
        try:
            conn.execute("DELETE FROM data_graph_fts WHERE rowid = ?", (row_id,))
        except Exception:
            pass
    conn.execute("DELETE FROM data_graph WHERE rowid = ?", (row_id,))
    conn.execute("DELETE FROM data_graph_edges WHERE from_id=? OR to_id=?", (row_id, row_id))
    try:
        conn.execute("DELETE FROM data_graph_key_vec WHERE rowid = ?", (row_id,))
    except Exception:
        pass
    try:
        conn.execute("DELETE FROM data_graph_value_vec WHERE rowid = ?", (row_id,))
    except Exception:
        pass


def _fetch_active_row(conn: sqlite3.Connection, canonical_key: str) -> dict[str, object] | None:
    row = conn.execute(
        "SELECT id, key, value, evidence_count, retrieval_weight, "
        "first_seen_at, last_confirmed_at "
        "FROM data_graph "
        "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
        "LIMIT 1",
        (KIND_USER_SPECIFIC, canonical_key),
    ).fetchone()
    if row is None:
        return None
    return {
        'id': row[0],
        'key': row[1],
        'value': row[2],
        'evidence_count': row[3] or 1,
        'retrieval_weight': row[4] or 1.0,
        'first_seen_at': row[5],
        'last_confirmed_at': row[6],
    }


def _effective_date(row: dict[str, object]) -> str:
    return cast(str, row.get('last_confirmed_at') or row.get('first_seen_at') or '')


def _add_supersession_edges(conn: sqlite3.Connection, winner_id: int, loser_id: int) -> None:
    from services.time_utils import utc_now
    now_iso = utc_now().isoformat()
    for from_id, to_id, edge_type in (
        (winner_id, loser_id, 'supersedes'),
        (loser_id, winner_id, 'superseded_by'),
    ):
        try:
            conn.execute(
                "INSERT OR IGNORE INTO data_graph_edges "
                "(from_id, to_id, edge_type, strength, created_at) "
                "VALUES (?, ?, ?, 1.0, ?)",
                (from_id, to_id, edge_type, now_iso),
            )
        except Exception as e:
            print(f"  WARN: edge insert failed ({from_id}→{to_id} {edge_type}): {e}")


def _apply_temporal_collision(
    conn: sqlite3.Connection,
    migrating: dict[str, object],
    existing: dict[str, object],
    canonical_key: str,
    dry_run: bool,
) -> None:
    migrating_date = _effective_date(migrating)
    existing_date = _effective_date(existing)
    if migrating_date >= existing_date:
        winner_id, loser_id = cast(int, migrating['id']), cast(int, existing['id'])
        winner_key = canonical_key
        loser_rw = cast(float, existing['retrieval_weight'])
        action = f"  TEMPORAL collision on '{canonical_key}': migrating id={migrating['id']} wins over existing id={existing['id']}"
    else:
        winner_id, loser_id = cast(int, existing['id']), cast(int, migrating['id'])
        winner_key = canonical_key
        loser_rw = cast(float, migrating['retrieval_weight'])
        action = f"  TEMPORAL collision on '{canonical_key}': existing id={existing['id']} wins over migrating id={migrating['id']}"

    print(action)
    if dry_run:
        return

    conn.execute(
        "UPDATE data_graph SET active=0, retrieval_weight=? WHERE rowid=?",
        (loser_rw * _TEMPORAL_DEMOTION_FACTOR, loser_id),
    )
    # Ensure the winner carries the canonical key (handles case where winner is the migrating row)
    conn.execute(
        "UPDATE data_graph SET key=? WHERE rowid=?",
        (winner_key, winner_id),
    )
    _add_supersession_edges(conn, winner_id, loser_id)


def _apply_coexist_collision(
    conn: sqlite3.Connection,
    migrating: dict[str, object],
    existing: dict[str, object],
    canonical_key: str,
    key: str,
    value: str,
    search_queries: str,
    canonical_blob: bytes | None,
    emb_service: EmbeddingService,
    dry_run: bool,
) -> str:
    if (cast(str, migrating['value'] or '')).strip().lower() == (cast(str, existing['value'] or '')).strip().lower():
        print(f"  COEXIST same-value merge on '{canonical_key}': hard-deleting id={migrating['id']}, incrementing id={existing['id']} evidence_count")
        if dry_run:
            return 'merged-coexist'
        merged_count = cast(int, existing['evidence_count'] or 1) + cast(int, migrating['evidence_count'] or 1)
        conn.execute(
            "UPDATE data_graph SET evidence_count=? WHERE rowid=?",
            (merged_count, existing['id']),
        )
        _hard_delete_row(conn, cast(int, migrating['id']))
        return 'merged-coexist'

    # Different values — rewrite migrating row's key so both coexist under canonical_key
    print(f"  COEXIST diff-value rewrite on '{canonical_key}': id={migrating['id']} '{key}' → '{canonical_key}'")
    if dry_run:
        return 'updated'
    _do_rewrite(conn, cast(int, migrating['id']), key, value, search_queries, canonical_key, canonical_blob, emb_service)
    return 'updated'


def _apply_immutable_collision(
    conn: sqlite3.Connection,
    migrating: dict[str, object],
    existing: dict[str, object],
    canonical_key: str,
    key: str,
    dry_run: bool,
) -> str:
    if (cast(str, migrating['value'] or '')).strip().lower() == (cast(str, existing['value'] or '')).strip().lower():
        print(f"  IMMUTABLE same-value merge on '{canonical_key}': hard-deleting id={migrating['id']}, incrementing id={existing['id']} evidence_count")
        if dry_run:
            return 'merged-immutable'
        merged_count = cast(int, existing['evidence_count'] or 1) + cast(int, migrating['evidence_count'] or 1)
        conn.execute(
            "UPDATE data_graph SET evidence_count=? WHERE rowid=?",
            (merged_count, existing['id']),
        )
        _hard_delete_row(conn, cast(int, migrating['id']))
        return 'merged-immutable'

    print(
        f"  IMMUTABLE conflict on '{canonical_key}': "
        f"existing='{existing['value']}' migrating='{migrating['value']}' "
        f"— leaving migrated row at original key '{key}'"
    )
    return 'skipped'


def _do_rewrite(
    conn: sqlite3.Connection,
    row_id: int,
    old_key: str,
    old_value: str,
    search_queries: str,
    canonical_key: str,
    canonical_blob: bytes | None,
    emb_service: EmbeddingService,
) -> None:
    """Rewrite key, key_vec, and FTS for a single data_graph row."""
    conn.execute(
        "UPDATE data_graph SET key = ? WHERE id = ?",
        (canonical_key, row_id),
    )
    conn.execute("DELETE FROM data_graph_key_vec WHERE rowid = ?", (row_id,))
    if canonical_blob is None:
        canonical_emb = emb_service.generate_embedding(canonical_key)
        canonical_blob = pack_embedding(canonical_emb)
    if canonical_blob:
        conn.execute(
            "INSERT INTO data_graph_key_vec(rowid, embedding) VALUES (?, ?)",
            (row_id, canonical_blob),
        )
    _update_fts(conn, row_id, old_key, old_value, KIND_USER_SPECIFIC, search_queries, canonical_key)


def _build_canonical_blob(
    canonical_key: str, emb_service: EmbeddingService
) -> bytes | None:
    """Generate and pack the embedding for canonical_key."""
    return pack_embedding(emb_service.generate_embedding(canonical_key))


def _process_row(
    conn: sqlite3.Connection,
    lut_conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    row_id: int,
    key: str,
    value: str,
    search_queries: str,
    blob: bytes,
    dry_run: bool,
) -> str:
    """Dispatch a single row through the full canonicalization pipeline.

    Returns one of: 'updated', 'skipped', 'merged-temporal', 'merged-coexist',
    'merged-immutable'.
    """
    hit = _lut_lookup(lut_conn, blob)
    if hit is None or hit[0] == key:
        return 'skipped'

    canonical_key, rule = hit
    existing = _fetch_active_row(conn, canonical_key)

    if existing is None or existing['id'] == row_id:
        # No collision — straight rewrite
        print(f"  Canonicalized id={row_id}: '{key}' → '{canonical_key}'")
        if dry_run:
            return 'updated'
        canonical_blob = _build_canonical_blob(canonical_key, emb_service)
        _do_rewrite(conn, row_id, key, value, search_queries, canonical_key, canonical_blob, emb_service)
        return 'updated'

    # Collision detected — read migrating row's full data for merge logic
    migrating_row = conn.execute(
        "SELECT id, value, evidence_count, retrieval_weight, first_seen_at, last_confirmed_at "
        "FROM data_graph WHERE id=? LIMIT 1",
        (row_id,),
    ).fetchone()
    if migrating_row is None:
        return 'skipped'

    migrating = {
        'id': migrating_row[0],
        'value': migrating_row[1],
        'evidence_count': migrating_row[2] or 1,
        'retrieval_weight': migrating_row[3] or 1.0,
        'first_seen_at': migrating_row[4],
        'last_confirmed_at': migrating_row[5],
    }

    if rule == 'temporal':
        _apply_temporal_collision(conn, migrating, existing, canonical_key, dry_run)
        return 'merged-temporal'

    if rule == 'coexist':
        canonical_blob = None if dry_run else _build_canonical_blob(canonical_key, emb_service)
        return _apply_coexist_collision(
            conn, migrating, existing, canonical_key,
            key, value, search_queries, canonical_blob, emb_service, dry_run,
        )

    if rule == 'immutable':
        return _apply_immutable_collision(conn, migrating, existing, canonical_key, key, dry_run)

    # Unknown rule — fall back to temporal semantics (safe default)
    _apply_temporal_collision(conn, migrating, existing, canonical_key, dry_run)
    return 'merged-temporal'


_OUTCOME_COUNTERS = {
    'updated': 'updated',
    'merged-temporal': 'tx',
    'merged-coexist': 'cx',
    'merged-immutable': 'ix',
}


def _run_row(
    db: DatabaseService,
    lut_conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    row: tuple[int, str, str, str],
    emb: np.ndarray,
    dry_run: bool,
) -> str:
    """Embed, validate, and process a single row. Returns an outcome string."""
    row_id, key, value, search_queries = row[0], row[1], row[2], row[3]
    blob = pack_embedding(emb)
    if blob is None:
        return 'skipped'
    try:
        with db.connection() as conn:
            return _process_row(
                conn, lut_conn, emb_service,
                row_id, key, value or '', search_queries or '',
                blob, dry_run,
            )
    except Exception as e:
        print(f"  ERROR: row id={row_id} key='{key}' failed: {e}")
        return 'skipped'


def main() -> None:
    """Entry point for the canonicalization migration."""
    parser = argparse.ArgumentParser(
        description="Canonicalize user_specific data_graph keys via the concept LUT."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Print planned actions without writing any changes.",
    )
    args = parser.parse_args()
    dry_run: bool = args.dry_run

    if dry_run:
        print("DRY RUN — no changes will be written.\n")

    db = get_shared_db_service()
    emb_service = EmbeddingService()
    lut_conn = _open_lut()

    print(f"Opened LUT: {_CONCEPT_LUT_PATH}")

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, key, value, search_queries FROM data_graph "
            "WHERE kind=? AND active=1 AND deleted_at IS NULL",
            (KIND_USER_SPECIFIC,),
        ).fetchall()

    print(f"Found {len(rows)} active user_specific rows to inspect")

    counts: dict[str, int] = {'updated': 0, 'skipped': 0, 'tx': 0, 'cx': 0, 'ix': 0}

    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        embeddings = emb_service.generate_embeddings_batch([r[1] for r in batch])

        for row, emb in zip(batch, embeddings):
            outcome = _run_row(db, lut_conn, emb_service, row, emb, dry_run)
            counts[_OUTCOME_COUNTERS.get(outcome, 'skipped')] += 1

        print(f"  Processed {min(i + _BATCH_SIZE, len(rows))}/{len(rows)}...")

    lut_conn.close()
    print(
        f"\nMigration complete. "
        f"Updated: {counts['updated']}, Skipped: {counts['skipped']}, "
        f"Merged-temporal: {counts['tx']}, Merged-coexist: {counts['cx']}, "
        f"Conflict-immutable: {counts['ix']}"
    )


if __name__ == "__main__":
    main()
