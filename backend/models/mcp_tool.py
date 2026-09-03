"""McpTool — one ``mcp_tools`` row: a synced remote MCP tool description.

Active-record row-model (Rule 5 / §4.1). ``id`` is the DDL's own
``INTEGER PRIMARY KEY`` (auto-assigned), so the base's ``save``/``get``/
``delete`` id-centric verbs apply unmodified; no id-generation override is
needed here (contrast :class:`~models.mcp_client_server.McpClientServer`'s
TEXT-PK). The table lives in the dedicated ``mcp_tools.sqlite`` runtime
file (schema owned by :mod:`services.mcp_tools_db`, not chalie.db's
schema.sql), so :meth:`_bound_connection` is overridden to reach that file
through :func:`~services.mcp_tools_db.get_tools_connection` — the same
sanctioned separate-db-file exception as :class:`~models.skill.Skill`.
This model is the SOLE home of ``mcp_tools`` row SQL; :class:`~services.mcp_client_service.McpClientService`
reads and writes exclusively through it. Holds no mp, calls no business
service (Rule-3 depth).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, ClassVar

from models.model import Model
from services.mcp_tools_db import get_tools_connection

if TYPE_CHECKING:
    from models.mcp_client_server import McpClientServer


class McpTool(Model):
    """One ``mcp_tools`` row: field storage + CRUD + the server relationship."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "server_id", "tool_name", "summary", "raw_schema",
    )

    @classmethod
    def get_table(cls) -> str:
        return "mcp_tools"

    @classmethod
    def _bound_connection(cls) -> sqlite3.Connection:
        """Reach mcp_tools.sqlite directly rather than the chalie.db-bound
        getter ``Database.bind()`` installs on the base — the sanctioned
        exception for this table's dedicated db file (owner ruling).

        The gateway opens the DB and ensures the schema exists, so reads
        on a fresh install never crash with ``no such table: mcp_tools``.
        """
        return get_tools_connection()

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    id: int | None
    server_id: str
    tool_name: str
    summary: str
    raw_schema: str

    # ── Relationship ─────────────────────────────────────────────────────────

    def server(self) -> McpClientServer | None:
        """The parent server row whose id matches this tool's ``server_id``,
        or ``None`` if no such server exists."""
        from models.mcp_client_server import McpClientServer  # late import to avoid circular
        return McpClientServer.get(self.server_id)
