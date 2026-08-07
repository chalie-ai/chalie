"""Feature tests for the ``mcp_tools`` ability — list and activate MCP tools.

Real ``McpToolsAbility.run()`` on a real ``MessageProcessor``, real temp SQLite
via the ``db`` fixture, zero mocks. Tests that touch ``mcp_tools.sqlite``
redirect ``FileMapperService._DATA_DIR`` to ``tmp_path`` (the pattern from
``test_mcp_client_service.py``) so the real data directory is never written.
"""

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from abilities._result import ToolResult
from abilities.mcp_tools import McpToolsAbility
from configs.channels import UserConfig
from contracts.params.mcp_tools_params_bag import McpToolsParamsBag
from controllers.message_processor import MessageProcessor
from services.file_mapper_service import FileMapperService
from services.mcp_client_service import McpClientService

pytestmark = pytest.mark.unit


def _run(action: str, tools: "list[str] | None" = None) -> tuple[ToolResult, dict[str, object], MessageProcessor]:
    """Drive ``McpToolsAbility.run()`` on a real ``MessageProcessor`` carrying
    the production ``UserConfig``. Returns ``(result, body, mp)`` — the mp so
    callers can assert the ``active_tools`` mutation the run performed."""
    mp = MessageProcessor(UserConfig({"hidden_input": True}), raw_input="mcp tool test")
    mp.active_tools = list(mp.config.always_available or [])
    ability = McpToolsAbility()
    ability.mp = mp
    bag = McpToolsParamsBag.from_params({"action": action, "tools": tools or []})
    assert isinstance(bag, McpToolsParamsBag), f"harness expects valid params. got={bag!r}"
    result = ability.run(bag)
    body = result.body if isinstance(result.body, dict) else {}
    return result, cast("dict[str, object]", body), mp


def _seed_online_server_with_tool(db: sqlite3.Connection) -> str:
    """Seed one enabled+online server and one synced tool row via the real
    production writers. Returns the prefixed tool name. Callers MUST have
    redirected ``_DATA_DIR`` first so mcp_tools.sqlite lands in tmp."""
    from services.mcp_tools_db import get_tools_connection

    server = McpClientService().add_server(
        name="test-server", host="https://mcp.example.com/mcp", headers={}, enabled=True,
    )
    # Manually set status (bypassing ping_and_sync, which needs network).
    db.execute("UPDATE mcp_client_servers SET status='online' WHERE id=?", (server["id"],))
    db.commit()

    tool_name = f"_mcp_{McpClientService._sanitize_name('test-server')}_fetch"
    conn_tools = get_tools_connection()
    conn_tools.execute(
        "INSERT OR REPLACE INTO mcp_tools (server_id, tool_name, summary, raw_schema) "
        "VALUES (?, ?, ?, ?)",
        (server["id"], tool_name, "Test tool", "{}"),
    )
    conn_tools.commit()
    return tool_name


# ── action "list" — every server, tools only for enabled+online ─────────────────


def test_list_returns_all_servers_with_status(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action 'list' returns every configured MCP server with its status; tool
    lists ({name, summary}) appear only for enabled+online servers."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    tool_name = _seed_online_server_with_tool(db)

    result, body, _mp = _run("list")

    assert result.status == "success"
    servers = cast("list[dict[str, object]]", body["servers"])
    assert len(servers) == 1, f"one seeded server must be listed. servers={servers!r}"
    assert servers[0]["name"] == "test-server"
    assert servers[0]["status"] == "online"
    tools = cast("list[dict[str, str]]", servers[0]["tools"])
    assert [t["name"] for t in tools] == [tool_name]
    assert tools[0]["summary"] == "Test tool"


# ── action "activate" — exact online tool name goes live for the turn ───────────


def test_activate_appends_online_tool_to_active_tools(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activating an exact synced online tool name appends it to the run's
    ``mp.active_tools`` and returns it under ``body["activated"]``."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    tool_name = _seed_online_server_with_tool(db)

    result, body, mp = _run("activate", [tool_name])

    assert result.status == "success"
    activated = cast("list[dict[str, str]]", body["activated"])
    assert [a["name"] for a in activated] == [tool_name]
    assert activated[0]["summary"] == "Test tool"
    assert tool_name in mp.active_tools, (
        f"activate must append to the run's active_tools. active={mp.active_tools!r}"
    )


def test_activate_with_unknown_tool_is_loud_no_results(db: sqlite3.Connection) -> None:
    """An unknown name is an all-miss: a loud no-results error pointing back at
    action 'list', and it never touches ``active_tools``."""
    result, _body, mp = _run("activate", ["_mcp_unknown_tool"])

    assert result.status == "error"
    assert result.code == "no-results"
    assert result.hint == "Use action 'list' to see the exact tool names."
    assert result.meta["not_found_count"] == 1
    assert "_mcp_unknown_tool" not in mp.active_tools


# ── params bag — pure returns, never throws ─────────────────────────────────────


def test_params_bag_invalid_action_returns_error_result() -> None:
    result = McpToolsParamsBag.from_params({"action": "invalid_action", "tools": []})
    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert result.code == "invalid-param"


def test_params_bag_activate_without_tools_returns_error_result() -> None:
    result = McpToolsParamsBag.from_params({"action": "activate", "tools": []})
    assert isinstance(result, ToolResult)
    assert result.status == "error"
