# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
E2E test for the MCP server external agent communication feature (TKT-438).

Tests:
1. MCP server starts and exposes the talk_to_chalie tool
2. Authenticated requests get a response
3. Unauthenticated requests are rejected
"""

import json
import os
import sys
import threading
import time

import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_TEST_PORT = 18462


@pytest.fixture(scope="module", autouse=True)
def _patch_db(tmp_path_factory):
    """Create a temp SQLite DB with required tables and patch get_shared_db_service."""
    import sqlite3
    from services import database_service

    db_dir = tmp_path_factory.mktemp("mcp_test_db")
    db_path = str(db_dir / "test.db")

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS wrapper_tokens (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            wrapper_id TEXT NOT NULL UNIQUE,
            capabilities TEXT NOT NULL DEFAULT '{}',
            permissions TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            last_seen_at TEXT,
            created_at TEXT NOT NULL,
            revoked_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_wrapper_tokens_hash
            ON wrapper_tokens(token_hash)
            WHERE revoked_at IS NULL;

        CREATE TABLE IF NOT EXISTS transcript (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_call_id TEXT,
            tool_name TEXT,
            internal INTEGER DEFAULT 0,
            deliberation_score REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            xml_migrated INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_transcript_channel
            ON transcript(channel, created_at);

        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            params TEXT DEFAULT '{}',
            result TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            ephemeral INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT,
            encrypted_value TEXT,
            is_sensitive INTEGER NOT NULL DEFAULT 0,
            value_type TEXT NOT NULL DEFAULT 'string',
            description TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS data_graph (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL DEFAULT '',
            source TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS policy_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'chat',
            state TEXT NOT NULL DEFAULT 'ask',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(action_id, context)
        );

        CREATE TABLE IF NOT EXISTS llm_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job TEXT NOT NULL,
            model TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.close()

    test_db = database_service.DatabaseService(db_path)

    original = database_service.get_shared_db_service

    def _patched():
        return test_db

    database_service.get_shared_db_service = _patched
    database_service._shared_db_service = test_db

    yield test_db

    database_service.get_shared_db_service = original
    database_service._shared_db_service = None


@pytest.fixture(scope="module")
def auth_token(_patch_db):
    """Create a test auth token in the patched DB."""
    from services.wrapper_auth_service import WrapperAuthService

    auth_svc = WrapperAuthService(_patch_db)
    raw_token, _ = auth_svc.create_token(
        name="E2E Test Agent",
        wrapper_id_override="__e2e_test__",
    )
    return raw_token


@pytest.fixture(scope="module")
def mcp_server(_patch_db):
    """Start MCP server with auth middleware in a background thread."""
    from mcp_server.server import create_mcp_server, _build_app

    mcp = create_mcp_server(host="127.0.0.1", port=_TEST_PORT)
    app = _build_app(mcp)

    config = uvicorn.Config(app, host="127.0.0.1", port=_TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="test-mcp-server")
    thread.start()

    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{_TEST_PORT}/mcp", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    yield {"port": _TEST_PORT, "url": f"http://127.0.0.1:{_TEST_PORT}"}

    server.should_exit = True
    thread.join(timeout=3)


class TestMCPServerAuth:
    """Verify auth enforcement."""

    def test_unauthenticated_request_rejected(self, mcp_server):
        """Requests without a valid bearer token should be rejected (401)."""
        import urllib.request
        import urllib.error

        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }).encode()

        req = urllib.request.Request(
            f"{mcp_server['url']}/mcp",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)

        assert exc_info.value.code == 401

    def test_invalid_token_rejected(self, mcp_server):
        """Requests with an invalid bearer token should be rejected (401)."""
        import urllib.request
        import urllib.error

        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }).encode()

        req = urllib.request.Request(
            f"{mcp_server['url']}/mcp",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer invalid_token_abc123",
            },
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)

        assert exc_info.value.code == 401


class TestMCPServerToolList:
    """Verify the MCP server exposes talk_to_chalie."""

    def _mcp_request(self, url, method, params, auth_token, session_id=None):
        """Send a JSON-RPC request to the MCP endpoint, return (response_data, headers)."""
        import urllib.request

        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {auth_token}",
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        req = urllib.request.Request(url, data=body, headers=headers)
        resp = urllib.request.urlopen(req, timeout=10)
        raw = resp.read().decode()

        # Extract session ID from response headers
        resp_session_id = resp.headers.get("Mcp-Session-Id")

        # Parse SSE response
        for line in raw.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "result" in data:
                    return data["result"], resp_session_id

        return None, resp_session_id

    def test_tool_list_contains_talk_to_chalie(self, mcp_server, auth_token):
        """The tools/list response should include talk_to_chalie."""
        url = f"{mcp_server['url']}/mcp"

        # Step 1: Initialize session
        init_result, session_id = self._mcp_request(
            url, "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "0.1.0"},
            },
            auth_token,
        )
        assert init_result is not None, "initialize did not return a result"

        # Step 2: List tools using the session
        tools_result, _ = self._mcp_request(
            url, "tools/list", {}, auth_token, session_id,
        )
        assert tools_result is not None, "tools/list did not return a result"

        tool_names = [t["name"] for t in tools_result.get("tools", [])]
        assert "talk_to_chalie" in tool_names, (
            f"Expected talk_to_chalie in {tool_names}"
        )


class TestMCPServerToolCall:
    """Verify talk_to_chalie tool execution via tools/call."""

    def _mcp_request(self, url, method, params, auth_token, session_id=None):
        """Send a JSON-RPC request to the MCP endpoint."""
        import urllib.request

        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {auth_token}",
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        req = urllib.request.Request(url, data=body, headers=headers)
        resp = urllib.request.urlopen(req, timeout=30)
        raw = resp.read().decode()

        resp_session_id = resp.headers.get("Mcp-Session-Id")

        for line in raw.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "result" in data:
                    return data["result"], resp_session_id

        return None, resp_session_id

    def test_talk_to_chalie_returns_response(self, mcp_server, auth_token):
        """Calling talk_to_chalie with a valid message returns a non-empty response."""
        from unittest.mock import patch, MagicMock

        fake_response = MagicMock()
        fake_response.text = "Hello from Chalie! I received your message."
        fake_response.tool_calls = []

        fake_provider = MagicMock()
        fake_provider.send_messages.return_value = fake_response
        fake_provider.get_context_limit.return_value = 128_000
        fake_provider.get_compact_at.return_value = 100_000
        fake_provider.estimate_payload_tokens.return_value = 500

        url = f"{mcp_server['url']}/mcp"

        # Initialize session
        init_result, session_id = self._mcp_request(
            url, "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "0.1.0"},
            },
            auth_token,
        )
        assert init_result is not None

        # Call talk_to_chalie with LLM patched
        with patch("services.providers.Providers") as mock_providers_cls:
            mock_providers_cls.instance.return_value = fake_provider

            call_result, _ = self._mcp_request(
                url, "tools/call",
                {
                    "name": "talk_to_chalie",
                    "arguments": {
                        "message": "What's on my schedule today?",
                        "agent_name": "Claude Code",
                        "project_or_task_name": "Chalie TKT-438",
                    },
                },
                auth_token,
                session_id,
            )

        assert call_result is not None, "tools/call returned no result"
        content = call_result.get("content", [])
        assert len(content) > 0, f"Expected content in response, got: {call_result}"
        text = content[0].get("text", "")
        assert len(text) > 0, "Expected non-empty text response from talk_to_chalie"
        assert "Hello from Chalie" in text
