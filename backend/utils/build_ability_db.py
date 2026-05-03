"""
Build or drift-check the ability search database.

Walks backend/abilities/ for concrete Ability subclasses, embeds SUMMARY +
EXAMPLES for each, and writes:
  backend/abilities/assets/abilities.sqlite  — vector + FTS5 search index
  resources/pre-trained/abilities_sha.json   — drift sidecar

Run from backend/:
    python -m utils.build_ability_db           # build (default)
    python -m utils.build_ability_db --check   # drift check
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
from abilities._registry import AbilityRegistry  # noqa: E402
from services.embedding_service import EmbeddingService  # noqa: E402
from services.embedding_utils import pack_embedding  # noqa: E402

_DB_PATH = Path(__file__).resolve().parent.parent / "abilities" / "assets" / "abilities.sqlite"
_SHA_PATH = paths.PRETRAINED_DIR / "abilities_sha.json"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension("vec0")


def _rebuild_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS ability_search_fts")
    conn.execute("DROP TABLE IF EXISTS ability_search_vec")
    conn.execute("DROP INDEX IF EXISTS idx_search_entries_ability")
    conn.execute("DROP TABLE IF EXISTS ability_search_entries")
    conn.execute("DROP TABLE IF EXISTS abilities")
    conn.commit()

    conn.execute("""
        CREATE TABLE abilities (
            id      INTEGER PRIMARY KEY,
            name    TEXT    UNIQUE NOT NULL,
            summary TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE ability_search_entries (
            id         INTEGER PRIMARY KEY,
            ability_id INTEGER NOT NULL REFERENCES abilities(id) ON DELETE CASCADE,
            text       TEXT    NOT NULL,
            kind       TEXT    NOT NULL CHECK(kind IN ('summary', 'example'))
        )
    """)
    conn.execute("CREATE INDEX idx_search_entries_ability ON ability_search_entries(ability_id)")
    conn.execute("CREATE VIRTUAL TABLE ability_search_vec USING vec0(embedding float[768])")
    conn.execute("""
        CREATE VIRTUAL TABLE ability_search_fts USING fts5(
            text,
            content='ability_search_entries',
            content_rowid='id'
        )
    """)
    conn.commit()


def _compute_sha(ability) -> str:
    raw = json.dumps([ability.SUMMARY, *ability.EXAMPLES], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _dedup_entries(
    entries: list[tuple[str, str]],
    embeddings: list[np.ndarray],
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    """Drop entry j when cos(emb[i], emb[j]) > 0.95 for any i < j.

    Embeddings are L2-normalised, so dot-product == cosine similarity.
    Entry order determines precedence: summary first, then examples in order.
    """
    kept_entries: list[tuple[str, str]] = []
    kept_embs: list[np.ndarray] = []

    for entry, emb_j in zip(entries, embeddings):
        duplicate = any(float(np.dot(emb_i, emb_j)) > 0.95 for emb_i in kept_embs)
        if not duplicate:
            kept_entries.append(entry)
            kept_embs.append(emb_j)

    return kept_entries, kept_embs


def _insert_ability(conn: sqlite3.Connection, emb_service: EmbeddingService, ability) -> int:
    """Insert one ability and its search entries; return count of entries inserted."""
    conn.execute(
        "INSERT INTO abilities(name, summary) VALUES (?, ?)",
        (ability.NAME, ability.SUMMARY),
    )
    ability_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    raw_entries: list[tuple[str, str]] = [
        (ability.SUMMARY, "summary"),
        *((ex, "example") for ex in ability.EXAMPLES),
    ]
    texts = [e[0] for e in raw_entries]
    embeddings = list(emb_service.generate_embeddings_batch(texts))

    entries, embeddings = _dedup_entries(raw_entries, embeddings)

    for (text, kind), emb in zip(entries, embeddings):
        conn.execute(
            "INSERT INTO ability_search_entries(ability_id, text, kind) VALUES (?, ?, ?)",
            (ability_id, text, kind),
        )
        entry_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_vec(rowid, embedding) VALUES (?, ?)",
            (entry_id, pack_embedding(emb)),
        )
        conn.execute(
            "INSERT INTO ability_search_fts(rowid, text) VALUES (?, ?)",
            (entry_id, text),
        )

    conn.commit()
    return len(entries)


def _build_sha_map() -> dict[str, str]:
    """SHA map covers every ability indexed in the search DB."""
    return {a.NAME: _compute_sha(a) for a in AbilityRegistry.all()}


def _build(db_path: Path, sha_path: Path) -> None:
    # Index every ability — the search DB is the single source of truth for
    # find_tools. Per-processor scoping (which abilities a given processor
    # may discover) is gated at find_tools query time via the calling
    # processor's DISCOVERABLE list.
    abilities = AbilityRegistry.all()
    print(f"Found {len(abilities)} abilities — building {db_path.name}...")

    emb_service = EmbeddingService() if abilities else None

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        _load_sqlite_vec(conn)
        _rebuild_schema(conn)

        total_entries = 0
        for ability in abilities:
            n = _insert_ability(conn, emb_service, ability)
            print(f"  {ability.NAME}: {n} entries")
            total_entries += n
    finally:
        conn.close()

    sha_map = _build_sha_map()
    sha_path.write_text(json.dumps(sha_map, indent=2, sort_keys=True) + "\n")

    print(f"\nDone. {len(abilities)} abilities, {total_entries} entries total.")
    print(f"SHA sidecar written to {sha_path.name}")


def _check(sha_path: Path) -> None:
    if not sha_path.exists():
        print(f"ERROR: sidecar not found at {sha_path}")
        sys.exit(1)

    existing = json.loads(sha_path.read_text())
    current = _build_sha_map()

    if existing == current:
        print("OK")
        sys.exit(0)

    classified: list[str] = []
    for name in sorted(set(existing) | set(current)):
        if name not in existing:
            classified.append(f"{name} [added]")
        elif name not in current:
            classified.append(f"{name} [removed]")
        elif existing[name] != current[name]:
            classified.append(f"{name} [changed]")

    print("DRIFT DETECTED:")
    for line in classified:
        print(f"  {line}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or check the ability search database")
    parser.add_argument("--check", action="store_true", help="Check for drift instead of building")
    args = parser.parse_args()

    if args.check:
        _check(_SHA_PATH)
    else:
        _build(_DB_PATH, _SHA_PATH)


if __name__ == "__main__":
    main()
