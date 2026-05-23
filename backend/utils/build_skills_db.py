"""
Build or drift-check the skill search database.

Walks backend/abilities/skills/ for .yaml skill files (curated) and
data/skills/user/ for user-created skills, embeds title + use_for
text for each, and writes:
  backend/abilities/assets/skills.sqlite  — vector + FTS5 search index
  backend/pre-trained/skills_sha.json   — drift sidecar

Run from backend/:
    python -m utils.build_skills_db           # build (default)
    python -m utils.build_skills_db --check   # drift check
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
from services.embedding_service import EmbeddingService  # noqa: E402
from services.embedding_utils import pack_embedding  # noqa: E402

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "abilities" / "skills"
_USER_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "skills" / "user"
_DB_PATH = Path(__file__).resolve().parent.parent / "abilities" / "assets" / "skills.sqlite"
_SHA_PATH = paths.PRETRAINED_DIR / "skills_sha.json"

_DEDUP_THRESHOLD = 0.95
_SOURCE_CURATED = "curated"
_SOURCE_USER = "user"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension("vec0")


def _rebuild_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate all skill search tables."""
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
            related_abilities TEXT,
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
    conn.execute("""
        CREATE VIRTUAL TABLE skill_search_fts USING fts5(
            text,
            content='skill_search_entries',
            content_rowid='id'
        )
    """)
    conn.commit()


def _parse_skill_file(path: Path) -> dict:
    """Parse a YAML-frontmatter skill file into a metadata dict."""
    text = path.read_text()
    if not text.startswith('---'):
        raise ValueError(f"Skill file {path} missing YAML frontmatter")
    _, fm_raw, body = text.split('---', 2)
    meta = yaml.safe_load(fm_raw)
    meta['content'] = body.strip()
    return meta


def _compute_sha(meta: dict) -> str:
    """SHA of the fields that drive the search index (title + use_for + tags)."""
    raw = json.dumps(
        [meta.get('title', ''), meta.get('use_for', ''), meta.get('tags', '')],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _dedup_entries(
    entries: list[tuple[str, str]],
    embeddings: list[np.ndarray],
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    """Drop entry j when cos(emb[i], emb[j]) > threshold for any i < j.

    Embeddings are L2-normalised so dot-product == cosine similarity.
    Entry order determines precedence: title first, then use_for.
    """
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
    """Embed and insert search entries for an already-inserted skill row.

    Writes to skill_search_entries, skill_search_vec, and skill_search_fts.
    Returns the number of search entries inserted.
    """
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
        conn.execute(
            "INSERT INTO skill_search_fts(rowid, text) VALUES (?, ?)",
            (entry_id, text),
        )

    conn.commit()
    return len(entries)


def remove_skill(conn: sqlite3.Connection, skill_id: int) -> None:
    """Delete a skill row and its dependent search entries (via CASCADE)."""
    conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
    conn.commit()


def _insert_skill(
    conn: sqlite3.Connection,
    emb_service: EmbeddingService,
    meta: dict,
    source: str = _SOURCE_CURATED,
) -> int:
    """Insert one skill row and its search entries. Returns count of entries inserted."""
    title = meta.get('title', '')
    use_for = meta.get('use_for', '')
    tags_raw = meta.get('tags', '')
    tags_str = tags_raw if isinstance(tags_raw, str) else ', '.join(tags_raw)

    conn.execute(
        "INSERT INTO skills(title, use_for, content, tags, version, related_abilities, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            title,
            use_for,
            meta.get('content', ''),
            tags_str,
            meta.get('version', 1),
            meta.get('related_abilities', ''),
            source,
        ),
    )
    skill_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    return index_skill(conn, emb_service, skill_id, title, use_for, tags_str)


def _load_skills() -> list[dict]:
    """Walk skills/ directory and parse all .yaml skill files."""
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


def _load_user_skills() -> list[dict]:
    """Walk data/skills/user/ directory and parse all .yaml skill files."""
    if not _USER_SKILLS_DIR.exists():
        return []
    skills = []
    for path in sorted(_USER_SKILLS_DIR.glob("*.yaml")):
        try:
            meta = _parse_skill_file(path)
            meta['_path'] = path.name
            skills.append(meta)
        except Exception as exc:
            print(f"  WARNING: skipping user skill {path.name}: {exc}")
    return skills


def _build_sha_map(skills: list[dict]) -> dict[str, str]:
    """SHA map covering every skill in the search DB."""
    return {m.get('title', m['_path']): _compute_sha(m) for m in skills}


def _build(db_path: Path, sha_path: Path) -> None:
    curated_skills = _load_skills()
    user_skills = _load_user_skills()
    all_skills = curated_skills + user_skills
    print(f"Found {len(curated_skills)} curated + {len(user_skills)} user skills — building {db_path.name}...")

    emb_service = EmbeddingService() if all_skills else None

    db_path.parent.mkdir(parents=True, exist_ok=True)
    _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        _load_sqlite_vec(conn)
        _rebuild_schema(conn)

        total_entries = 0
        for meta in curated_skills:
            n = _insert_skill(conn, emb_service, meta, source=_SOURCE_CURATED)
            print(f"  [curated] {meta.get('title', meta['_path'])}: {n} entries")
            total_entries += n

        for meta in user_skills:
            n = _insert_skill(conn, emb_service, meta, source=_SOURCE_USER)
            print(f"  [user] {meta.get('title', meta['_path'])}: {n} entries")
            total_entries += n
    finally:
        conn.close()

    sha_map = _build_sha_map(all_skills)
    sha_path.write_text(json.dumps(sha_map, indent=2, sort_keys=True) + "\n")

    print(f"\nDone. {len(curated_skills)} curated + {len(user_skills)} user skills, {total_entries} entries total.")
    print(f"SHA sidecar written to {sha_path.name}")


def _check(sha_path: Path) -> None:
    if not sha_path.exists():
        print(f"ERROR: sidecar not found at {sha_path}")
        sys.exit(1)

    existing = json.loads(sha_path.read_text())
    all_skills = _load_skills() + _load_user_skills()
    current = _build_sha_map(all_skills)

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
