# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""_mcp_* passthrough proxy business-logic tests."""

import json
import socket
import threading
import time

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.file_mapper_service import FileMapperService
from services.mcp_client_service import McpClientService
from services.message_processor import MessageProcessor
from tests._tool_result_harness import allow_policy, body, seed_transcript

pytestmark = pytest.mark.unit


# ── shared scaffolding ─────────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _seed_transcript(db, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="call an mcp tool")


def _mp_for(config, db) -> MessageProcessor:
    mp = MessageProcessor("call an mcp tool")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    mp.uid = _seed_transcript(db, "chat")
    return mp


def _allow(db, permission: str) -> None:
    """Seed an ``allow`` policy row so the gate runs the callback (reaches run()).

    This is the real policy table the production ``PolicyManager.wrap`` SELECTS —
    seeding it is the prod-faithful way to make an _mcp_* call resolve past the
    lazy-'ask' default, NOT a monkeypatch of the gate.
    """
    allow_policy(db, permission, channel="chat")


def _head(rendered: str) -> str:
    return rendered.splitlines()[0]


def _body(rendered: str, tool_name: str) -> str:
    return body(rendered, tool_name)


# ── UNREACHABLE: real outbound connection failure → code=mcp-unreachable ─────────


def test_unreachable_server_renders_mcp_unreachable_naming_the_server(db):
    svc = McpClientService()
    # Real registration through the production CRUD path; the host is a closed
    # localhost port so the outbound MCP connection fails fast.
    svc.add_server(
        name="ghost",
        host=f"http://127.0.0.1:{_free_port()}/mcp",
        headers={},
        enabled=True,
    )

    tool = "_mcp_ghost_do_thing"
    _allow(db, tool)
    mp = _mp_for(UserConfig({}), db)

    out = ToolDispatcher(mp).dispatch(tool, {"x": 1, "act_summary": "x"})

    assert f"[{tool}(status=error, code=mcp-unreachable" in out
    assert "code=error]" not in out  # the legacy mechanical marker is GONE
    assert f"[end:{tool}]" in out
    # The server is named so a weak model knows WHICH endpoint failed.
    assert "ghost" in out

    # The same error envelope was recorded on the act-trail.
    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert "code=mcp-unreachable" in trail[0]["result"]


def test_unknown_mcp_server_renders_stable_error_code(db):
    tool = "_mcp_nosuchserver_whatever"
    _allow(db, tool)
    mp = _mp_for(UserConfig({}), db)

    out = ToolDispatcher(mp).dispatch(tool, {"act_summary": "x"})

    assert f"[{tool}(status=error, code=mcp-unknown-tool" in out
    assert "code=error]" not in out
    assert f"[end:{tool}]" in out
    assert tool in out


# ── REACHABLE: a real local MCP server drives content shaping + isError ──────────


@pytest.fixture()
def local_mcp_server():
    port = _free_port()
    mcp = FastMCP("passthrough_probe", host="127.0.0.1", port=port)

    @mcp.tool()
    def say_text(msg: str) -> str:
        return f"the answer is {msg}"

    @mcp.tool()
    def give_struct() -> dict:
        return {"city": "Valletta", "temps": [21, 23, 19], "ok": True}

    @mcp.tool()
    def explode() -> str:
        raise ToolError("downstream blew up")

    app = mcp.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="test-mcp-out")
    thread.start()

    import urllib.request

    base = f"http://127.0.0.1:{port}/mcp"
    for _ in range(100):
        try:
            urllib.request.urlopen(base, timeout=1)
            break
        except Exception:
            time.sleep(0.05)

    yield {"port": port, "host": base}

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def registered_server(db, tmp_path, monkeypatch, local_mcp_server):
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    svc = McpClientService()
    server = svc.add_server(
        name="probe",
        host=local_mcp_server["host"],
        headers={},
        enabled=True,
    )
    result = svc.ping_and_sync(server["id"])
    assert result["reachable"] is True, (
        f"local MCP server must be reachable for the test, got {result!r}"
    )
    return svc


def test_text_tool_renders_prose_body(db, registered_server):
    tool = "_mcp_probe_say_text"
    _allow(db, tool)
    mp = _mp_for(UserConfig({}), db)

    out = ToolDispatcher(mp).dispatch(tool, {"msg": "42", "act_summary": "x"})

    assert _head(out).startswith(f"[{tool}(status=success")
    assert f"[end:{tool}]" in out
    body = _body(out, tool)
    # Prose, verbatim — not a JSON string wrapping the text.
    assert body == "the answer is 42"
    # The server is named in flat meta for provenance.
    assert "server=probe" in _head(out)


def test_struct_tool_keeps_structured_body_no_double_wrapping(db, registered_server):
    tool = "_mcp_probe_give_struct"
    _allow(db, tool)
    mp = _mp_for(UserConfig({}), db)

    out = ToolDispatcher(mp).dispatch(tool, {"act_summary": "x"})

    assert _head(out).startswith(f"[{tool}(status=success")
    assert "server=probe" in _head(out)
    body = _body(out, tool)
    # The body is parseable JSON whose top level is the structured object itself —
    # if it were double-wrapped, json.loads would yield a STRING, not a dict.
    parsed = json.loads(body)
    assert isinstance(parsed, dict), (
        f"structured content must be a JSON object, not a wrapped string: {body!r}"
    )
    assert parsed["city"] == "Valletta"
    assert parsed["temps"] == [21, 23, 19]
    assert parsed["ok"] is True
    # Rendered by the dispatcher's dict-path: COMPACT JSON (no pretty-print
    # indentation). A passed-through prose JSON string would carry the remote's
    # newline+two-space indentation — the double-wrap / non-structured smell.
    assert body == json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    assert "\n" not in body


def test_remote_iserror_renders_mcp_tool_error(db, registered_server):
    tool = "_mcp_probe_explode"
    _allow(db, tool)
    mp = _mp_for(UserConfig({}), db)

    out = ToolDispatcher(mp).dispatch(tool, {"act_summary": "x"})

    assert f"[{tool}(status=error, code=mcp-tool-error" in out
    assert "status=success" not in _head(out)
    assert f"[end:{tool}]" in out
    # The remote's error text is surfaced to the model.
    assert "downstream blew up" in out
