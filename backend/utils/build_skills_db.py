
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import cast

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.embedding_service import EmbeddingService, get_embedding_service  # noqa: E402
from services.embedding_utils import pack_embedding  # noqa: E402
from services.file_mapper_service import FileMapperService  # noqa: E402

_SKILLS_DIR = FileMapperService.get_abilities_skills_path()
_DB_PATH = FileMapperService.get_skills_db_path()
_SHA_PATH = FileMapperService.get_skills_sha_path()

_DEDUP_THRESHOLD = 0.95
_SOURCE_CURATED = "curated"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension("vec0")


def _rebuild_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS skill_search_fts")
    conn.execute("DROP TABLE IF EXISTS skill_search_vec")
    conn.execute("DROP INDEX IF EXISTS idx_skill_search_entries_skill")
    conn.execute("DROP TABLE IF EXISTS skill_search_entries")
    conn.execute("DROP TABLE IF EXISTS skill_associations")
    conn.execute("DROP TABLE IF EXISTS skills")
    conn.commit()

    conn.execute("""
        CREATE TABLE skills (
            id               INTEGER PRIMARY KEY,
            title            TEXT    NOT NULL,
            use_for          TEXT    NOT NULL,
            content          TEXT    NOT NULL,
            tags             TEXT,
            version          INTEGER DEFAULT 1,
            source           TEXT    NOT NULL DEFAULT 'curated'
                                 CHECK(source IN ('curated', 'user')),
            enabled          INTEGER NOT NULL DEFAULT 1,
            based_on         INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE skill_associations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id     INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            pattern_name TEXT    NOT NULL,
            rule         TEXT    NOT NULL,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(skill_id, pattern_name)
        )
    """)
    conn.execute("""
        CREATE TABLE skill_search_entries (
            id       INTEGER PRIMARY KEY,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            text     TEXT    NOT NULL,
            kind     TEXT    NOT NULL CHECK(kind IN ('title', 'use_for', 'tag'))
        )
    """)
    conn.execute(
        "CREATE INDEX idx_skill_search_entries_skill ON skill_search_entries(skill_id)"
    )
    conn.execute("CREATE VIRTUAL TABLE skill_search_vec USING vec0(embedding float[768])")
    # Trigram tokenizer over the skill TITLE only — the FTS branch of the discovery
    # cascade is bm25-on-title (see _search.SearchableAbility). use_for rows feed
    # the vec index alone, so they are NOT inserted into FTS (see index_skill).
    conn.execute("""
        CREATE VIRTUAL TABLE skill_search_fts USING fts5(
            text,
            content='skill_search_entries',
            content_rowid='id',
            tokenize='trigram'
        )
    """)
    conn.commit()


def _parse_skill_file(path: Path) -> dict[str, object]:
    text = path.read_text()
    if not text.startswith('---'):
        raise ValueError(f"Skill file {path} missing YAML frontmatter")
    _, fm_raw, body = text.split('---', 2)
    meta = cast("dict[str, object]", yaml.safe_load(fm_raw))
    meta['content'] = body.strip()
    return meta


def _compute_sha(meta: dict[str, object]) -> str:
    raw = json.dumps(
        [meta.get('title', ''), meta.get('use_for', ''), meta.get('tags', '')],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _dedup_entries(
    entries: list[tuple[str, str]],
    embeddings: list[np.ndarray],
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    """Entry order determines precedence: title first, then use_for."""
    kept_entries: list[tuple[str, str]] = []
    kept_embs: list[np.ndarray] = []

    for entry, emb_j in zip(entries, embeddings):
        duplicate = any(
            float(np.dot(emb_i, emb_j)) > _DEDUP_THRESHOLD for emb_i in kept_embs
        )
        if not duplicate:
            kept_entries.append(entry)
            kept_embs.append(emb_j)

    return kept_entries, kept_embs


def index_skill(
    conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    skill_id: int,
    title: str,
    use_for: str,
    tags_str: str,
) -> int:
    combined = f"{title}. {use_for}"
    raw_entries: list[tuple[str, str]] = [
        (title, 'title'),
        (use_for, 'use_for'),
        (combined, 'use_for'),
    ]
    texts = [e[0] for e in raw_entries]
    embeddings = list(emb_service.generate_embeddings_batch(texts))

    entries, embeddings = _dedup_entries(raw_entries, embeddings)

    for (text, kind), emb in zip(entries, embeddings):
        conn.execute(
            "INSERT INTO skill_search_entries(skill_id, text, kind) VALUES (?, ?, ?)",
            (skill_id, text, kind),
        )
        entry_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO skill_search_vec(rowid, embedding) VALUES (?, ?)",
            (entry_id, pack_embedding(emb)),
        )
        # Title only into FTS — the cascade's bm25 rung is title-only; use_for
        # rows are reachable via the vec index alone.
        if kind == "title":
            conn.execute(
                "INSERT INTO skill_search_fts(rowid, text) VALUES (?, ?)",
                (entry_id, text),
            )

    conn.commit()
    return len(entries)


def _insert_skill(
    conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    meta: dict[str, object],
    source: str = _SOURCE_CURATED,
) -> int:
    title = cast("str", meta.get('title', ''))
    use_for = cast("str", meta.get('use_for', ''))
    tags_raw = meta.get('tags', '')
    tags_str = tags_raw if isinstance(tags_raw, str) else ', '.join(cast("list[str]", tags_raw))

    conn.execute(
        "INSERT INTO skills(title, use_for, content, tags, version, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            title,
            use_for,
            meta.get('content', ''),
            tags_str,
            meta.get('version', 1),
            source,
        ),
    )
    skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    return index_skill(conn, emb_service, skill_id, title, use_for, tags_str)


def _load_skills() -> list[dict[str, object]]:
    if not _SKILLS_DIR.exists():
        return []
    skills = []
    for path in sorted(_SKILLS_DIR.glob("*.yaml")):
        try:
            meta = _parse_skill_file(path)
            meta['_path'] = path.name
            skills.append(meta)
        except Exception as exc:
            print(f"  WARNING: skipping {path.name}: {exc}")
    return skills


def _build_sha_map(skills: list[dict[str, object]]) -> dict[str, str]:
    return {cast("str", m.get('title', m['_path'])): _compute_sha(m) for m in skills}


def _build(db_path: Path, sha_path: Path) -> None:
    """Build the shipped, committed skills index — curated skills ONLY.

    User skills (``data/skills/user/*.yaml``) are per-user runtime state, never
    baked into the committed artifact. They are indexed individually into the
    local DB via ``index_skill`` from ``skill_builder``/``api.skills`` on
    create/edit. Do NOT re-add ``_load_user_skills`` here — it leaks the
    machine's personal skills into the shipped ``skills.sqlite``.
    """
    curated_skills = _load_skills()
    print(f"Found {len(curated_skills)} curated skills — building {db_path.name}...")

    emb_service = get_embedding_service() if curated_skills else None

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        _load_sqlite_vec(conn)
        _rebuild_schema(conn)

        total_entries = 0
        for meta in curated_skills:
            n = _insert_skill(conn, cast("EmbeddingService", emb_service), meta, source=_SOURCE_CURATED)
            print(f"  [curated] {meta.get('title', meta['_path'])}: {n} entries")
            total_entries += n
    finally:
        conn.close()

    sha_map = _build_sha_map(curated_skills)
    sha_path.write_text(json.dumps(sha_map, indent=2, sort_keys=True) + "\n")

    print(f"\nDone. {len(curated_skills)} curated skills, {total_entries} entries total.")
    print(f"SHA sidecar written to {sha_path.name}")


def _check(sha_path: Path) -> None:
    if not sha_path.exists():
        print(f"ERROR: sidecar not found at {sha_path}")
        sys.exit(1)

    existing = json.loads(sha_path.read_text())
    current = _build_sha_map(_load_skills())

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
    parser = argparse.ArgumentParser(description="Build or check the skill search database")
    parser.add_argument("--check", action="store_true", help="Check for drift instead of building")
    args = parser.parse_args()

    if args.check:
        _check(_SHA_PATH)
    else:
        _build(_DB_PATH, _SHA_PATH)


if __name__ == "__main__":
    main()
