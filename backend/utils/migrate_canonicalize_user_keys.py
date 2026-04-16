"""
Backfill migration: canonicalize existing user_specific data_graph keys via the concept LUT.

Walks every active, non-deleted user_specific row in data_graph. For each row,
embeds the key and performs a KNN lookup against concept_lut.sqlite. If the LUT
returns a canonical key that differs from the stored key, the row is updated
in-place:
    - data_graph.key     → canonical_key
    - data_graph_key_vec → new embedding blob for the canonical key
    - FTS index          → delete old entry, insert new entry

Idempotent: rows already at their canonical key are skipped. Safe to re-run.

Run from backend/:
    python -m utils.migrate_canonicalize_user_keys
"""

import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.data_graph_service import (
    KIND_USER_SPECIFIC,
    _CONCEPT_LUT_PATH,
    _CONCEPT_LUT_THRESHOLD,
    _l2_dist_to_cosine,
)
from services.database_service import get_shared_db_service
from services.embedding_service import EmbeddingService
from services.embedding_utils import pack_embedding

_BATCH_SIZE = 32


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension('vec0')


def _open_lut() -> sqlite3.Connection:
    """Open the concept LUT read-only."""
    if not Path(_CONCEPT_LUT_PATH).exists():
        raise FileNotFoundError(f"concept_lut.sqlite not found at {_CONCEPT_LUT_PATH}")
    conn = sqlite3.connect(_CONCEPT_LUT_PATH)
    _load_sqlite_vec(conn)
    return conn


def _lut_lookup(lut_conn: sqlite3.Connection, blob: bytes) -> tuple[str, str] | None:
    """KNN lookup against lut_embeddings; return (canonical_key, rule) or None."""
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


def _update_fts(conn, rowid: int, old_key: str, old_value: str, old_kind: str,
                old_sq: str, new_key: str) -> None:
    """Delete old FTS entry and insert updated entry with new key."""
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
    except Exception:
        pass


def main() -> None:
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

    updated = 0
    skipped = 0

    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        keys = [r[1] for r in batch]
        embeddings = emb_service.generate_embeddings_batch(keys)

        for row, emb in zip(batch, embeddings):
            row_id, key, value, search_queries = row[0], row[1], row[2], row[3]

            blob = pack_embedding(emb)
            if blob is None:
                skipped += 1
                continue

            hit = _lut_lookup(lut_conn, blob)
            if hit is None or hit[0] == key:
                skipped += 1
                continue

            canonical_key = hit[0]

            with db.connection() as conn:
                conn.execute(
                    "UPDATE data_graph SET key = ? WHERE id = ?",
                    (canonical_key, row_id),
                )
                conn.execute(
                    "DELETE FROM data_graph_key_vec WHERE rowid = ?", (row_id,)
                )
                # Embed the canonical key for the updated key_vec entry
                canonical_emb = emb_service.generate_embedding(canonical_key)
                canonical_blob = pack_embedding(canonical_emb)
                if canonical_blob:
                    conn.execute(
                        "INSERT INTO data_graph_key_vec(rowid, embedding) VALUES (?, ?)",
                        (row_id, canonical_blob),
                    )
                _update_fts(conn, row_id, key, value, KIND_USER_SPECIFIC, search_queries, canonical_key)

            print(f"  Canonicalized id={row_id}: '{key}' → '{canonical_key}'")
            updated += 1

        print(f"  Processed {min(i + _BATCH_SIZE, len(rows))}/{len(rows)}...")

    lut_conn.close()
    print(f"\nMigration complete. Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
