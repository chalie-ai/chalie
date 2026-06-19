"""Feature tests for McpClientService - network-free behaviors.

The end-to-end MCP-over-HTTP path is covered by a separate end-to-end
scenario.  These tests cover only the deterministic, network-free
behaviors exercised against a real temporary SQLite database without mocks.
"""

import pytest

from services.file_mapper_service import FileMapperService
from services.mcp_client_service import McpClientService, _sanitize_name, _tool_name

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test 1 — tool-name sanitization and _mcp_ prefixing
# ---------------------------------------------------------------------------


def test_tool_name_sanitization_produces_valid_prefix():
    """Asserts the key format _mcp_<sanitized>_<tool> that downstream dispatch,
    policy seeding, and find_tools gating all depend on.
    """
    # Server name with capitals, hyphens, spaces, and leading/trailing noise
    result = _tool_name("My-Taskie Server!", "create_document")
    assert result == "_mcp_my_taskie_server_create_document"

    # Purely numeric server name
    result = _tool_name("42services", "ping")
    assert result == "_mcp_42services_ping"

    # Unicode / special chars collapse to underscores then stripped
    result = _tool_name("home-assistant", "toggle_light")
    assert result == "_mcp_home_assistant_toggle_light"

    # All punctuation collapses to a single underscore (then stripped); falls back to 'server'
    result = _tool_name("!!!---!!!", "foo")
    assert result == "_mcp_server_foo"


# ---------------------------------------------------------------------------
# Test 2 — _resolve_tool longest-prefix routing + disabled-server guard
# ---------------------------------------------------------------------------


def test_resolve_tool_routes_to_longest_prefix_server(db):
    """_resolve_tool picks the longest-matching server prefix.

    Raises ValueError for a disabled server's tool before any network call.
    """
    svc = McpClientService()

    # Register two servers: 'task' and 'taskie' — 'task_' is a prefix of 'taskie_'
    server_a = svc.add_server(name="task", host="http://nowhere:1111", headers={}, enabled=True)
    server_b = svc.add_server(name="taskie", host="http://nowhere:2222", headers={}, enabled=True)

    # Seed tool rows directly into mcp_tools.sqlite so _resolve_tool has something to match.
    # We do NOT call ping_and_sync (that requires the network); we test _resolve_tool's
    # lookup logic in isolation by seeding the servers table only.
    # _resolve_tool reads from mcp_client_servers via list_servers(), NOT mcp_tools.sqlite.
    # The tool name encodes the server: _mcp_taskie_create_document → server 'taskie'.
    server, remote_tool = svc._resolve_tool("_mcp_taskie_create_document")
    assert server["id"] == server_b["id"], (
        "Expected _mcp_taskie_create_document to resolve to the 'taskie' server, "
        f"but got server id={server['id']!r} (name={server['name']!r})"
    )
    assert remote_tool == "create_document"

    # The shorter-prefix server ('task') still resolves its own tools correctly.
    server_a_resolved, remote_tool_a = svc._resolve_tool("_mcp_task_ping")
    assert server_a_resolved["id"] == server_a["id"]
    assert remote_tool_a == "ping"

    # Disabling server_b and then trying to resolve its tool raises ValueError.
    svc.update_server(server_b["id"], {"enabled": False})
    with pytest.raises(ValueError, match="disabled"):
        svc._resolve_tool("_mcp_taskie_create_document")


# ---------------------------------------------------------------------------
# Test 4 — get_online_mcp_tool_names gates on enabled=1 AND status=online
# ---------------------------------------------------------------------------


def test_get_online_mcp_tool_names_excludes_disabled_and_offline(db, tmp_path, monkeypatch):
    """get_online_mcp_tool_names returns tool names only for servers that are
    both enabled=1 AND status='online' - the gate find_tools uses to control
    LLM visibility.

    _DATA_DIR is redirected to tmp_path so mcp_tools.sqlite lands in a fresh
    temp directory.  The conftest db fixture patches get_shared_db_service
    (not _DATA_DIR), so the two redirects are independent.
    """
    # Redirect mcp_tools.sqlite to a fresh temp directory via the class attribute
    # that get_mcp_tools_db_path() reads.  monkeypatch auto-undoes this at teardown.
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)

    svc = McpClientService()

    # Server A: enabled=1, status='online' → should appear.
    server_a = svc.add_server(name="online_svc", host="http://a:1", headers={}, enabled=True)
    # Server B: enabled=0, status='online' (disabled) → must NOT appear.
    server_b = svc.add_server(name="disabled_svc", host="http://b:1", headers={}, enabled=False)
    # Server C: enabled=1, status='offline' → must NOT appear.
    server_c = svc.add_server(name="offline_svc", host="http://c:1", headers={}, enabled=True)

    # Manually write status values to the DB (bypassing ping_and_sync which needs network).
    with svc._db.connection() as conn:
        conn.execute(
            "UPDATE mcp_client_servers SET status='online' WHERE id=?", (server_a["id"],)
        )
        conn.execute(
            "UPDATE mcp_client_servers SET status='online' WHERE id=?", (server_b["id"],)
        )
        conn.execute(
            "UPDATE mcp_client_servers SET status='offline' WHERE id=?", (server_c["id"],)
        )
        conn.commit()

    # Seed fake tool rows directly into mcp_tools.sqlite via the real production writer.
    from services.mcp_client_service import _open_tools_db

    conn_tools = _open_tools_db()
    try:
        for srv_id, srv_name in [
            (server_a["id"], "online_svc"),
            (server_b["id"], "disabled_svc"),
            (server_c["id"], "offline_svc"),
        ]:
            tool_name = f"_mcp_{_sanitize_name(srv_name)}_fetch"
            conn_tools.execute(
                "INSERT OR REPLACE INTO mcp_tools "
                "(server_id, tool_name, summary, raw_schema) VALUES (?, ?, ?, ?)",
                (srv_id, tool_name, "test tool", "{}"),
            )
        conn_tools.commit()
    finally:
        conn_tools.close()

    online_names = svc.get_online_mcp_tool_names()

    assert "_mcp_online_svc_fetch" in online_names, (
        "Enabled+online server's tool must be discoverable"
    )
    assert "_mcp_disabled_svc_fetch" not in online_names, (
        "Disabled server's tool must NOT be discoverable (even if status=online)"
    )
    assert "_mcp_offline_svc_fetch" not in online_names, (
        "Offline server's tool must NOT be discoverable (even if enabled)"
    )


# ---------------------------------------------------------------------------
# Test 5 — add_server persists row with correct defaults + enabled flag
# ---------------------------------------------------------------------------


def test_add_server_persists_row_with_correct_defaults(db):
    """add_server persists to mcp_client_servers with status='unknown' and
    JSON-serialized headers; verifies the row is actually in the DB.
    """
    svc = McpClientService()
    server = svc.add_server(
        name="my-server",
        host="http://example.lan:9000",
        headers={"Authorization": "Bearer token123"},
        enabled=True,
    )

    assert server["status"] == "unknown", (
        "New server status must start as 'unknown' — "
        "not 'online' (which requires a successful ping)"
    )
    assert server["enabled"] is True
    assert server["name"] == "my-server"
    assert server["host"] == "http://example.lan:9000"

    # Headers round-trip correctly.
    assert server["headers"]["Authorization"] == "Bearer token123"

    # Row is actually in the DB — not just in the returned dict.
    with svc._db.connection() as conn:
        row = conn.execute(
            "SELECT id, name, status, enabled FROM mcp_client_servers WHERE id = ?",
            (server["id"],),
        ).fetchone()

    assert row is not None
    assert row[1] == "my-server"
    assert row[2] == "unknown"
    assert row[3] == 1  # enabled=True → INTEGER 1


# ---------------------------------------------------------------------------
# Test 6 — get_tool_schema round-trips the stored inputSchema
# ---------------------------------------------------------------------------


def test_get_tool_schema_round_trips_stored_input_schema(db, tmp_path, monkeypatch):
    """get_tool_schema returns the exact inputSchema written by _write_tools -
    the schema the LLM sees when find_tools surfaces an _mcp_* tool.

    Seeded via the real _write_tools writer (no hand-crafted SQL) so the full
    store-to-read path is exercised.
    """
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)

    svc = McpClientService()

    # Seed via the real production writer.  server_id can be any string since
    # mcp_tools has no FK to mcp_client_servers.
    svc._write_tools(
        "srv-1",
        "taskie",
        [
            {
                "name": "create_document",
                "description": "Create a doc",
                "inputSchema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            }
        ],
    )

    # Positive path — existing tool.
    result = svc.get_tool_schema("_mcp_taskie_create_document")

    assert result is not None, "get_tool_schema must return a dict for a stored tool"
    assert result["name"] == "_mcp_taskie_create_document"
    assert result["description"] == "Create a doc"
    assert result["input_schema"] == {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }, (
        "inputSchema must survive the json.dumps→json.loads round-trip intact — "
        "this is the schema the LLM will use to build its tool call"
    )

    # Negative path — tool absent from the DB.
    assert svc.get_tool_schema("_mcp_taskie_does_not_exist") is None, (
        "get_tool_schema must return None for an unknown tool name"
    )


# ---------------------------------------------------------------------------
# Dedup / idempotent upsert on add
# ---------------------------------------------------------------------------

from services.mcp_client_service import _normalize_host, _open_tools_db  # noqa: E402


@pytest.mark.parametrize("a,b", [
    ("https://mcp.example.com/mcp", "https://mcp.example.com/mcp/"),
    ("https://MCP.Example.com/mcp", "https://mcp.example.com/mcp"),
    ("https://mcp.example.com:443/mcp", "https://mcp.example.com/mcp"),
    ("http://mcp.example.com:80/x", "http://mcp.example.com/x"),
    ("  https://mcp.example.com/mcp  ", "https://mcp.example.com/mcp"),
    ("mcp.example.com/mcp", "https://mcp.example.com/mcp"),
])
def test_normalize_host_treats_variants_as_equal(a, b):
    assert _normalize_host(a) == _normalize_host(b)


@pytest.mark.parametrize("a,b", [
    ("https://mcp.example.com/mcp", "https://mcp.example.com/other"),
    ("https://a.example.com/mcp", "https://b.example.com/mcp"),
    ("https://mcp.example.com:9000/mcp", "https://mcp.example.com/mcp"),
    ("http://mcp.example.com/x", "https://mcp.example.com/x"),
])
def test_normalize_host_keeps_distinct_endpoints_distinct(a, b):
    assert _normalize_host(a) != _normalize_host(b)


def test_add_server_dedups_same_endpoint_variant(db):
    """Re-adding the same endpoint (trailing-slash variant) updates the existing
    row instead of creating a duplicate."""
    svc = McpClientService()
    a = svc.add_server(name="taskie", host="https://mcp.example.com/mcp",
                       headers={}, enabled=True)
    b = svc.add_server(name="taskie", host="https://mcp.example.com/mcp/",
                       headers={}, enabled=True)
    assert b["id"] == a["id"]
    assert len(svc.list_servers()) == 1


def test_add_server_upsert_updates_fields_and_reenables(db):
    """Re-adding a known endpoint refreshes headers and re-enables it."""
    svc = McpClientService()
    a = svc.add_server(name="taskie", host="https://mcp.example.com/mcp",
                       headers={}, enabled=False)
    b = svc.add_server(name="taskie", host="https://mcp.example.com/mcp",
                       headers={"Authorization": "Bearer x"}, enabled=True)
    assert b["id"] == a["id"]
    assert b["enabled"] is True
    assert b["headers"] == {"Authorization": "Bearer x"}
    assert len(svc.list_servers()) == 1


def test_add_server_distinct_endpoints_create_separate_rows(db):
    """Genuinely different endpoints are kept as separate servers."""
    svc = McpClientService()
    svc.add_server(name="a", host="https://a.example.com/mcp", headers={}, enabled=True)
    svc.add_server(name="b", host="https://b.example.com/mcp", headers={}, enabled=True)
    assert len(svc.list_servers()) == 2


def test_add_server_upsert_name_change_purges_old_prefix_rows(db, tmp_path, monkeypatch):
    """When an upsert changes the server name, the old _mcp_<oldname>_* tool and
    policy rows are purged so a later re-sync leaves no orphans."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    svc = McpClientService()
    s = svc.add_server(name="taskie", host="https://mcp.example.com/mcp",
                       headers={}, enabled=True)
    old_tool = _tool_name("taskie", "create_document")

    conn = _open_tools_db()
    try:
        conn.execute(
            "INSERT INTO mcp_tools (server_id, tool_name, summary, raw_schema) "
            "VALUES (?, ?, ?, ?)",
            (s["id"], old_tool, "", "{}"),
        )
        conn.commit()
    finally:
        conn.close()
    with svc._db.connection() as c:
        c.execute(
            "INSERT OR IGNORE INTO policy (channel, permission, setting) "
            "VALUES (?, ?, ?)",
            ("chat", old_tool, "ask"),
        )
        c.commit()

    # Re-add the SAME endpoint under a NEW name → old-prefix rows must be purged.
    svc.add_server(name="taskie2", host="https://mcp.example.com/mcp",
                   headers={}, enabled=True)

    conn = _open_tools_db()
    try:
        tool_count = conn.execute(
            "SELECT COUNT(*) FROM mcp_tools WHERE tool_name = ?", (old_tool,)
        ).fetchone()[0]
    finally:
        conn.close()
    with svc._db.connection() as c:
        policy_count = c.execute(
            "SELECT COUNT(*) FROM policy WHERE permission = ?", (old_tool,)
        ).fetchone()[0]

    assert tool_count == 0
    assert policy_count == 0
    servers = svc.list_servers()
    assert len(servers) == 1
    assert servers[0]["name"] == "taskie2"
