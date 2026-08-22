"""Migration 021 — remove the inbound MCP server's bearer token.

The MCP server no longer authenticates at the transport: what an external
agent may do is decided per tool by the external-agent policy channel. The
credential it used to mint is therefore residue — two ``settings`` rows
(``mcp_server_token`` held the raw token in clear text,
``mcp_server_token_wrapper_id`` named its ``wrapper_tokens`` record) and the
wrapper rows themselves, minted under the ``__mcp_server`` id prefix. Those
wrapper rows still pass as a bearer credential on the REST API, so they are
revoked rather than left behind.

Idempotent — ``needed()`` is False once the rows are gone and the wrappers
revoked, so fresh installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_021_drop_mcp_server_token.py``
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from migrations.connection import connect  # noqa: E402
from services.file_mapper_service import FileMapperService  # noqa: E402
from services.time_utils import utc_now  # noqa: E402

_SETTING_KEYS = ("mcp_server_token", "mcp_server_token_wrapper_id")


def needed(conn: sqlite3.Connection) -> bool:
    """True while a token setting row or an unrevoked MCP wrapper token remains."""
    if conn.execute(
        "SELECT 1 FROM settings WHERE key IN (?, ?) LIMIT 1", _SETTING_KEYS
    ).fetchone() is not None:
        return True
    return conn.execute(
        "SELECT 1 FROM wrapper_tokens "
        "WHERE wrapper_id GLOB '__mcp_server*' AND revoked_at IS NULL LIMIT 1"
    ).fetchone() is not None


def apply(db_path: str) -> None:
    """Delete the token setting rows and revoke the MCP wrapper tokens. Idempotent."""
    conn = connect(db_path)
    try:
        settings_removed = conn.execute(
            "DELETE FROM settings WHERE key IN (?, ?)", _SETTING_KEYS
        ).rowcount
        revoked = conn.execute(
            "UPDATE wrapper_tokens SET revoked_at = ? "
            "WHERE wrapper_id GLOB '__mcp_server*' AND revoked_at IS NULL",
            (utc_now().isoformat(),),
        ).rowcount
        conn.commit()
        print(
            f"[migration_021] removed {settings_removed} token setting row(s), "
            f"revoked {revoked} MCP wrapper token(s) ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
