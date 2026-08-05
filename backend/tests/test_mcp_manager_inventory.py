"""Feature tests — ``mcp_manager``'s description carries a live server inventory.

The tool description the model actually receives is assembled by
``AbilityRegistry.build_tools(mp)`` → ``Ability.get_input_schema()`` →
``get_summary()``. These tests assert against that rendered descriptor, not the
getter in isolation, so a break anywhere in the assembly chain is caught.

Real ``McpClientService`` writes against the real temp SQLite ``db`` fixture,
real ``MessageProcessor`` carrying the production ``UserConfig``, zero mocks.
"""

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from abilities._registry import AbilityRegistry
from abilities.mcp_manager import McpManagerAbility
from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor
from services.file_mapper_service import FileMapperService
from services.mcp_client_service import McpClientService

pytestmark = pytest.mark.unit


def _rendered_description() -> str:
    """The mcp_manager description as the model receives it — off the real
    build_tools path on a real MessageProcessor, not a bare getter call."""
    mp = MessageProcessor(UserConfig({"hidden_input": True}), raw_input="mcp inventory test")
    mp.active_tools = list(mp.config.always_available or [])
    tools = {cast("str", t["name"]): t for t in AbilityRegistry.build_tools(mp)}
    assert McpManagerAbility.NAME in tools, (
        f"mcp_manager must be on the default surface. tools={sorted(tools)}"
    )
    return cast("str", tools[McpManagerAbility.NAME]["description"])


def _add(name: str, host: str, *, enabled: bool = True) -> str:
    return cast("str", McpClientService().add_server(
        name=name, host=host, headers={}, enabled=enabled,
    )["id"])


def _stamp_order(db: sqlite3.Connection, ids: list[str]) -> None:
    """Pin created_at explicitly so the ``ORDER BY created_at`` the inventory
    inherits from ``list_servers`` is deterministic by construction rather than
    by insert-clock luck."""
    for position, server_id in enumerate(ids):
        db.execute(
            "UPDATE mcp_client_servers SET created_at=? WHERE id=?",
            (f"2026-07-30T10:0{position}:00+00:00", server_id),
        )
    db.commit()


def test_inventory_names_connected_servers_in_creation_order(db: sqlite3.Connection) -> None:
    """The description carries the connected servers as one plain name list, in
    the same order the ``list`` action reports them."""
    ids = [
        _add("notes", "https://mcp.example.com/notes"),
        _add("weather", "https://mcp.example.com/weather"),
        _add("home-assistant", "https://mcp.example.com/home"),
    ]
    _stamp_order(db, ids)

    description = _rendered_description()

    assert "Connected MCP Servers: notes, weather, home-assistant" in description, (
        f"inventory line missing or misordered. description={description!r}"
    )


def test_disabled_server_is_absent_from_the_inventory(db: sqlite3.Connection) -> None:
    """``disable`` hides a server's tools from discovery, so it is registered but
    NOT connected — it must not appear in an inventory headed 'Connected'."""
    ids = [
        _add("live-one", "https://mcp.example.com/live"),
        _add("switched-off", "https://mcp.example.com/off", enabled=False),
    ]
    _stamp_order(db, ids)

    description = _rendered_description()

    assert "Connected MCP Servers: live-one" in description
    assert "switched-off" not in description, (
        f"a disabled server must not be advertised as connected. description={description!r}"
    )


def test_empty_inventory_says_none_connected(db: sqlite3.Connection) -> None:
    """Zero servers is answered in the description itself, so the model never
    spends a `list` call to discover there is nothing connected."""
    description = _rendered_description()

    assert "No MCP servers are connected." in description
    assert "Connected MCP Servers:" not in description


def test_inventory_lists_server_names_only_never_their_tools(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server's TOOLS belong to ``mcp_tools``, not to this description — putting
    them here would push the whole remote surface into every request."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    from services.mcp_tools_db import get_tools_connection
    from services.mcp_client_service import _sanitize_name

    server_id = _add("notes", "https://mcp.example.com/notes")
    db.execute("UPDATE mcp_client_servers SET status='online' WHERE id=?", (server_id,))
    db.commit()
    tool_name = f"_mcp_{_sanitize_name('notes')}_create_document"
    conn_tools = get_tools_connection()
    conn_tools.execute(
        "INSERT OR REPLACE INTO mcp_tools (server_id, tool_name, summary, raw_schema) "
        "VALUES (?, ?, ?, ?)",
        (server_id, tool_name, "Create a document", "{}"),
    )
    conn_tools.commit()

    description = _rendered_description()

    assert "Connected MCP Servers: notes" in description
    assert tool_name not in description, (
        f"tool names must never reach the summary. description={description!r}"
    )
    assert "Create a document" not in description
    inventory = [ln for ln in description.splitlines() if ln.startswith("Connected MCP Servers:")]
    assert inventory == ["Connected MCP Servers: notes"], (
        f"the inventory must be exactly one plain line. got={inventory!r}"
    )


def test_off_spine_summary_is_the_static_base_text(db: sqlite3.Connection) -> None:
    """``mp is None`` is the search-index / introspection path: it must render the
    deterministic base text with NO inventory, or the shipped index would encode
    whatever happened to be connected on the machine that built it."""
    ids = [_add("notes", "https://mcp.example.com/notes")]
    _stamp_order(db, ids)

    off_spine = McpManagerAbility().get_summary()

    assert off_spine == McpManagerAbility._BASE_SUMMARY
    assert "notes" not in off_spine
    assert "Connected MCP Servers:" not in off_spine
    assert "No MCP servers are connected." not in off_spine


def test_inventory_degrades_to_base_text_when_the_lookup_fails(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The description is reassembled on EVERY step. A DB fault must cost the
    inventory line, never the turn — and must not claim zero servers, which is a
    different fact from 'could not tell'."""
    def _boom(self: McpClientService) -> list[dict[str, object]]:
        raise RuntimeError("no connection bound")

    monkeypatch.setattr(McpClientService, "list_servers", _boom)

    description = _rendered_description()

    assert McpManagerAbility._BASE_SUMMARY in description
    assert "Connected MCP Servers:" not in description
    assert "No MCP servers are connected." not in description, (
        "an unreadable inventory must not be reported as an empty one"
    )
