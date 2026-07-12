"""McpClientServer — one ``mcp_client_servers`` row: a remote MCP endpoint.

Active-record row-model (Rule 5 / §4.1). The table's primary key is a TEXT
id (UUID4, generated the same way ``McpClientService.add_server`` always has —
via ``uuid.uuid4``), so :meth:`save` overrides the base's autoincrement-
``lastrowid`` INSERT the same way :class:`~models.list.List.save` does. This
model is the SOLE home of ``mcp_client_servers`` row SQL;
:class:`~services.mcp_client_service.McpClientService` reads and writes
exclusively through it. Holds no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, ClassVar, Self

from models.model import Model

if TYPE_CHECKING:
    from models.mcp_tool import McpTool


class McpClientServer(Model):
    """One ``mcp_client_servers`` row: field storage + CRUD + the tool
    relationship."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "name", "host", "headers", "enabled", "status",
        "last_pinged_at", "created_at", "updated_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "mcp_client_servers"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    id: str | None
    name: str
    host: str
    headers: str
    enabled: int
    status: str
    last_pinged_at: str | None
    created_at: str
    updated_at: str

    def save(self) -> Self:
        """INSERT a new server (TEXT-UUID4 PK generated here), or UPDATE the
        existing row — mirrors ``List.save``'s id-generation override."""
        if self.id is not None:
            return super().save()
        self.id = str(uuid.uuid4())
        connection = self._bound_connection()
        columns = self._fields()
        values = [getattr(self, name) for name in columns]
        column_list = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO {self.get_table()} ({column_list}) VALUES ({placeholders})",
            values,
        )
        return self

    # ── Relationship ─────────────────────────────────────────────────────────

    def tool(self) -> list[McpTool]:
        """All tool rows whose ``server_id`` matches this server's id."""
        from models.mcp_tool import McpTool  # late import to avoid circular
        return McpTool.filter("server_id", self.id).get()
