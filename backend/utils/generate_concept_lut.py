"""Generate concept LUT embeddings into concept_lut.sqlite.

Each alias is stored as its own row in lut_concepts pointing back to the same
canonical_key + rule, because cosine similarity between an alias and its canonical_key
under gte-modernbert falls below the 0.80 LUT threshold, causing canonicalization
misses at runtime.
"""

import sqlite3
import sys
from pathlib import Path

import yaml

# backend/ must be on sys.path so services.* imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.embedding_service import EmbeddingService
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

_YAML_PATH = FileMapperService.get_concept_lut_yaml_path()
_DB_PATH = FileMapperService.get_concept_lut_db_path()

_BATCH_SIZE = 32


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension('vec0')


def _read_concepts(yaml_path: Path) -> list[dict[str, str]]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    raw_concepts = data.get("concepts", [])
    if not raw_concepts:
        raise ValueError(f"No concepts found in {yaml_path}")

    rows = []
    for c in raw_concepts:
        canonical_key = c["canonical_key"]
        rule = c["rule"]
        # Always embed the canonical_key itself first
        rows.append({"canonical_key": canonical_key, "rule": rule, "label": canonical_key})
        # Embed every alias as an additional lookup label
        for alias in c.get("aliases", []):
            rows.append({"canonical_key": canonical_key, "rule": rule, "label": alias})
    return rows


def _rebuild_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS lut_embeddings")
    conn.commit()
    conn.execute("CREATE VIRTUAL TABLE lut_embeddings USING vec0(embedding float[768])")
    conn.commit()
    conn.execute("DROP TABLE IF EXISTS lut_concepts")
    conn.execute("""
        CREATE TABLE lut_concepts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_key TEXT NOT NULL,
            rule          TEXT NOT NULL,
            label         TEXT NOT NULL UNIQUE
        )
    """)
    conn.commit()


def _insert_concepts(conn: sqlite3.Connection, concepts: list[dict[str, str]]) -> list[int]:
    ids = []
    for c in concepts:
        conn.execute(
            "INSERT INTO lut_concepts(canonical_key, rule, label) VALUES (?, ?, ?)",
            (c["canonical_key"], c["rule"], c["label"]),
        )
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        ids.append(row[0])
    conn.commit()
    return ids


def _embed_and_insert(
    conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    concepts: list[dict[str, str]],
    ids: list[int],
) -> int:
    inserted = 0
    texts = [c["label"] for c in concepts]

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
    print(f"Found {len(concepts)} labels (canonical_keys + aliases) — loading embedding model...")

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
    print(f"\nDone. {inserted} labels (canonical_keys + aliases) embedded into {_DB_PATH.name}")
    print(f"Database size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
