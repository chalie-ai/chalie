"""
Generate concept LUT embeddings into concept_lut.sqlite.

Reads canonical keys + rules from concept_lut.yaml, embeds each canonical_key
via gte-modernbert (sha256-pinned, same encoder as production), and writes the
result into a sqlite + sqlite-vec virtual table at:
    backend/services/data_graph/assets/concept_lut.sqlite

Tables produced:
    lut_concepts   — (id, canonical_key, rule)  regular table
    lut_embeddings — vec0(embedding float[768])  rowid mirrors lut_concepts.id

Run from repo root:
    python -m utils.generate_concept_lut
"""

import sqlite3
import sys
from pathlib import Path

import yaml

# backend/ must be on sys.path so services.* imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.embedding_service import EmbeddingService
from services.embedding_utils import pack_embedding

_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "services" / "data_graph" / "assets" / "concept_lut.yaml"
)
_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "services" / "data_graph" / "assets" / "concept_lut.sqlite"
)

_BATCH_SIZE = 32


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension; falls back to the bare SO name."""
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension('vec0')


def _read_concepts(yaml_path: Path) -> list[dict]:
    """Parse YAML and return [{canonical_key, rule}, ...]."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    concepts = data.get("concepts", [])
    if not concepts:
        raise ValueError(f"No concepts found in {yaml_path}")
    return [{"canonical_key": c["canonical_key"], "rule": c["rule"]} for c in concepts]


def _rebuild_tables(conn: sqlite3.Connection) -> None:
    """Drop and recreate both tables for an idempotent rebuild."""
    conn.execute("DROP TABLE IF EXISTS lut_embeddings")
    conn.commit()
    conn.execute("CREATE VIRTUAL TABLE lut_embeddings USING vec0(embedding float[768])")
    conn.commit()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lut_concepts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_key TEXT NOT NULL UNIQUE,
            rule          TEXT NOT NULL
        )
    """)
    conn.execute("DELETE FROM lut_concepts")
    conn.commit()


def _insert_concepts(conn: sqlite3.Connection, concepts: list[dict]) -> list[int]:
    """Insert concept metadata rows and return their assigned IDs."""
    ids = []
    for c in concepts:
        conn.execute(
            "INSERT INTO lut_concepts(canonical_key, rule) VALUES (?, ?)",
            (c["canonical_key"], c["rule"]),
        )
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        ids.append(row[0])
    conn.commit()
    return ids


def _embed_and_insert(
    conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    concepts: list[dict],
    ids: list[int],
) -> int:
    """Embed canonical keys in batches and insert into lut_embeddings."""
    inserted = 0
    texts = [c["canonical_key"] for c in concepts]

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + _BATCH_SIZE]
        batch_ids = ids[batch_start:batch_start + _BATCH_SIZE]

        embeddings = emb_service.generate_embeddings_batch(batch_texts)

        for concept_id, embedding in zip(batch_ids, embeddings):
            blob = pack_embedding(embedding)
            conn.execute(
                "INSERT INTO lut_embeddings(rowid, embedding) VALUES (?, ?)",
                (concept_id, blob),
            )
            inserted += 1

        conn.commit()
        print(f"  {inserted}/{len(texts)} embedded...")

    return inserted


def main() -> None:
    if not _YAML_PATH.exists():
        print(f"ERROR: {_YAML_PATH} not found")
        sys.exit(1)

    print(f"Reading concepts from {_YAML_PATH}")
    concepts = _read_concepts(_YAML_PATH)
    print(f"Found {len(concepts)} concepts — loading embedding model...")

    emb_service = EmbeddingService()

    conn = sqlite3.connect(str(_DB_PATH))
    try:
        _load_sqlite_vec(conn)
        _rebuild_tables(conn)
        ids = _insert_concepts(conn, concepts)
        inserted = _embed_and_insert(conn, emb_service, concepts, ids)
    finally:
        conn.close()

    size_kb = _DB_PATH.stat().st_size / 1024
    print(f"\nDone. {inserted} concepts embedded into {_DB_PATH.name}")
    print(f"Database size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
