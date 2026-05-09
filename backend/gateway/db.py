"""
Dashboard SQLite database — manages installed interfaces, scopes, and config.

Separate from Chalie's database. Auto-creates on first boot.
"""

import json
import logging
import sqlite3
import threading

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS installed_interfaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    status TEXT DEFAULT 'online',
    version TEXT,
    description TEXT,
    author TEXT,
    requested_scopes TEXT DEFAULT '{}',
    approved_scopes TEXT DEFAULT '{}',
    chalie_interface_id TEXT,
    paired_at TEXT,
    last_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS interface_meta (
    interface_id TEXT PRIMARY KEY,
    data TEXT DEFAULT '{}',
    FOREIGN KEY (interface_id) REFERENCES installed_interfaces(id) ON DELETE CASCADE
);
"""

_local = threading.local()
_db_path: str = ""


def init_db(path: str) -> None:
    """Initialise the database path and create tables if needed."""
    global _db_path
    _db_path = path
    conn = _get_conn()
    conn.executescript(_SCHEMA)
    # Normalize any non-loopback hosts from previous registrations
    conn.execute(
        "UPDATE installed_interfaces SET host = '127.0.0.1' "
        "WHERE host != '127.0.0.1'"
    )
    conn.commit()
    logger.info("[Dashboard DB] Initialised at %s", path)


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(_db_path)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


# ── Config helpers ─────────────────────────────────────────────────────────

def get_config(key: str, default: str | None = None) -> str | None:
    """Read a config value."""
    row = _get_conn().execute(
        "SELECT value FROM config WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else default


def set_config(key: str, value: str) -> None:
    """Write a config value."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


# ── Interface CRUD ─────────────────────────────────────────────────────────

def list_interfaces() -> list[dict]:
    """Return all installed interfaces."""
    rows = _get_conn().execute(
        """SELECT id, name, host, port, status, version, description, author,
                  requested_scopes, approved_scopes, chalie_interface_id,
                  paired_at, last_seen_at
           FROM installed_interfaces ORDER BY paired_at ASC"""
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_interface(interface_id: str) -> dict | None:
    """Return a single interface by ID."""
    row = _get_conn().execute(
        """SELECT id, name, host, port, status, version, description, author,
                  requested_scopes, approved_scopes, chalie_interface_id,
                  paired_at, last_seen_at
           FROM installed_interfaces WHERE id = ?""",
        (interface_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def insert_interface(data: dict) -> None:
    """Insert a new interface record."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO installed_interfaces
           (id, name, host, port, status, version, description, author,
            requested_scopes, approved_scopes, chalie_interface_id, paired_at, last_seen_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["id"], data["name"], data["host"], data["port"],
            data.get("status", "online"),
            data.get("version"), data.get("description"), data.get("author"),
            json.dumps(data.get("requested_scopes", {})),
            json.dumps(data.get("approved_scopes", {})),
            data.get("chalie_interface_id"),
            data.get("paired_at"), data.get("last_seen_at"),
        ),
    )
    conn.commit()


def update_interface(interface_id: str, **fields) -> None:
    """Update specific fields on an interface record."""
    conn = _get_conn()
    sets = []
    vals = []
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        vals.append(json.dumps(v) if isinstance(v, dict) else v)
    vals.append(interface_id)
    conn.execute(
        f"UPDATE installed_interfaces SET {', '.join(sets)} WHERE id = ?",
        vals,
    )
    conn.commit()


def delete_interface(interface_id: str) -> None:
    """Delete an interface record."""
    conn = _get_conn()
    conn.execute("DELETE FROM installed_interfaces WHERE id = ?", (interface_id,))
    conn.commit()


# ── Interface metadata ─────────────────────────────────────────────────

def get_meta(interface_id: str) -> dict:
    """Return the daemon's metadata object."""
    row = _get_conn().execute(
        "SELECT data FROM interface_meta WHERE interface_id = ?",
        (interface_id,),
    ).fetchone()
    if row:
        return json.loads(row[0]) if row[0] else {}
    return {}


def set_meta(interface_id: str, data: dict) -> None:
    """Replace the daemon's entire metadata object."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO interface_meta (interface_id, data) VALUES (?, ?)",
        (interface_id, json.dumps(data)),
    )
    conn.commit()


def patch_meta(interface_id: str, updates: dict) -> dict:
    """Merge updates into the daemon's metadata object. Returns new state."""
    current = get_meta(interface_id)
    current.update(updates)
    set_meta(interface_id, current)
    return current


def delete_meta(interface_id: str) -> None:
    """Delete the daemon's metadata row."""
    conn = _get_conn()
    conn.execute("DELETE FROM interface_meta WHERE interface_id = ?", (interface_id,))
    conn.commit()


def _row_to_dict(row) -> dict:
    """Convert a database row to a dict."""
    return {
        "id": row[0],
        "name": row[1],
        "host": row[2],
        "port": row[3],
        "status": row[4],
        "version": row[5],
        "description": row[6],
        "author": row[7],
        "requested_scopes": json.loads(row[8]) if row[8] else {},
        "approved_scopes": json.loads(row[9]) if row[9] else {},
        "chalie_interface_id": row[10],
        "paired_at": row[11],
        "last_seen_at": row[12],
    }
