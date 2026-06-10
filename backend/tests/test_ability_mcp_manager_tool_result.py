# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the mcp_manager tool's ToolResult contract (TKT-910).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution of the production
``McpManagerAbility``, the real ``ToolDispatcher._prevalidate`` ACTION_REQUIRED
pre-gate, and the real ``McpClientService`` writing real rows to the real
``mcp_client_servers`` table (the ``db`` fixture) plus the real
``mcp_tools.sqlite`` runtime DB (isolated under ``tmp_path`` via the same
``FileMapperService._DATA_DIR`` monkeypatch the production service reads).

``mcp_manager`` is ``SYSTEM=True`` — it has no per-action policy gate, so no
policy row needs flipping and a headless test never hangs on a WebSocket ask.

What TKT-910 changes, exercised end to end:

* **Errors stop masquerading as success:** an unreachable host on ``add``, an
  unknown action, missing add params, and an unknown enable/disable target no
  longer ship as ``status=success`` prose — they carry a stable kebab ``code``.
* **The row is registered despite a loud error:** an unreachable ``add`` still
  persists the ``mcp_client_servers`` row (existing idempotent-upsert behaviour);
  the error is purely about the failed connect test.
* **Structured rows, not prose bullets:** ``list`` returns JSON rows
  ``{id,name,url,status,enabled,tool_count}``; empty inventory is SUCCESS
  ``[]`` with ``count=0``.

OFFLINE-safe: every "unreachable" host is ``http://127.0.0.1:1/mcp`` — port 1 is
a closed local port, so the outbound MCP connect fails fast with connection
refused, no internet required.

RED-before-change: against the pre-TKT-910 ability the unreachable add renders
``status=success`` "Registered MCP server …" prose with no ``code``; the unknown
enable/disable target renders ``status=success`` "No server found …" prose;
``list`` returns ``status=success`` prose bullet lines, not JSON rows. These
tests fail against that ability.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.mcp_manager import _classify_sync_error
from configs.channels import UserConfig
from services.file_mapper_service import FileMapperService
from services.mcp_client_service import McpClientService

pytestmark = pytest.mark.unit

# A closed local port — the outbound MCP connect fails fast (connection refused).
_DEAD_HOST = "http://127.0.0.1:1/mcp"


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the policy channel), ``uid`` and ``_uid`` (the
    act-trail / read-guard anchors). Production binds all three to the same
    transcript id; the fixture mirrors that."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid
        self._uid = uid


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "connect to an mcp server"),
    )
    db.commit()
    return cur.lastrowid


@pytest.fixture
def chat_mp(db, tmp_path, monkeypatch):
    """A real chat mp bound to the test database, with the ``mcp_tools.sqlite``
    runtime DB isolated under ``tmp_path`` so ``list``/``add`` tool-count reads
    and tool writes never touch the developer's real data dir."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _head(rendered: str) -> str:
    return rendered.splitlines()[0]


def _body(rendered: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:mcp_manager]")
    return rendered[head:tail]


def _parse_body(rendered: str) -> object:
    return json.loads(_body(rendered))


# ── 1. THE KILL SHOT: unreachable add → mcp-unreachable, but row IS registered ──


def test_add_unreachable_host_is_loud_error_but_row_registered(db, chat_mp):
    """Adding a server whose host does not answer is ``code=mcp-unreachable`` (not
    a ``status=success`` "Registered …" prose body), the hint NAMES the server,
    AND the row was still written to the real ``mcp_client_servers`` table —
    registration happened despite the loud connect-test failure."""
    out = ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "add", "name": "ghost", "host": _DEAD_HOST}
    )

    assert "[mcp_manager(status=error, code=mcp-unreachable" in out
    assert "status=success" not in out
    assert "ghost" in out  # the server is named so the model knows which failed

    # The hard guarantee: the connection row exists despite the error.
    row = db.execute(
        "SELECT name, host, enabled FROM mcp_client_servers WHERE name = 'ghost'"
    ).fetchone()
    assert row is not None
    assert row[1] == _DEAD_HOST
    assert row[2] == 1  # enabled by default on add


# ── 2. unknown action → pre-gate unknown-action + valid lists all four ──────────


def test_unknown_action_is_pre_gated(db, chat_mp):
    """An action outside the enum is rejected by the ACTION_REQUIRED pre-gate as
    ``code=unknown-action`` BEFORE run(), and ``valid:`` lists ALL four actions so
    the model can self-correct."""
    out = ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "frobnicate"}
    )

    assert "[mcp_manager(status=error, code=unknown-action" in out
    assert "status=success" not in out
    valid_line = next(line for line in out.splitlines() if line.startswith("valid:"))
    for action in ("list", "add", "enable", "disable"):
        assert action in valid_line


# ── 3. add missing name+host → ONE missing-params naming BOTH (pre-gate) ────────


def test_add_missing_name_and_host_is_one_missing_params(db, chat_mp):
    """Adding with neither name nor host is a SINGLE ``code=missing-params`` (from
    the pre-gate) that names BOTH missing params — not two drip-fed errors and not
    a success body."""
    out = ToolDispatcher(chat_mp).dispatch("mcp_manager", {"action": "add"})

    assert "[mcp_manager(status=error, code=missing-params" in out
    assert "status=success" not in out
    assert "name" in out
    assert "host" in out


# ── 4. list with no servers → success, body [], count=0 ─────────────────────────


def test_list_empty_is_success_empty_array(db, chat_mp):
    """With no servers configured, ``list`` is SUCCESS carrying ``[]`` and
    ``count=0`` — an empty inventory is a valid state, never an error and never
    prose."""
    out = ToolDispatcher(chat_mp).dispatch("mcp_manager", {"action": "list"})

    assert "[mcp_manager(status=success" in _head(out)
    assert "count=0" in _head(out)
    assert _parse_body(out) == []


# ── 5. list after add → structured rows whose values match the DB row ───────────


def test_list_returns_structured_rows_matching_db(db, chat_mp):
    """After an add, ``list`` returns structured JSON rows with the contract keys,
    and each value matches the real ``mcp_client_servers`` row."""
    ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "add", "name": "alpha", "host": _DEAD_HOST}
    )

    out = ToolDispatcher(chat_mp).dispatch("mcp_manager", {"action": "list"})

    assert "[mcp_manager(status=success" in _head(out)
    assert "count=1" in _head(out)
    rows = _parse_body(out)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"id", "name", "url", "status", "enabled", "tool_count"}

    db_row = db.execute(
        "SELECT id, name, host, status, enabled FROM mcp_client_servers "
        "WHERE name = 'alpha'"
    ).fetchone()
    assert row["id"] == db_row[0]
    assert row["name"] == db_row[1]
    assert row["url"] == db_row[2]
    assert row["status"] == db_row[3]
    assert row["enabled"] is bool(db_row[4])
    assert row["tool_count"] == 0  # dead host synced no tools


# ── 6. enable by name (dead host) → classified error, but row IS enabled ────────


def test_enable_by_name_dead_host_errors_but_row_enabled(db, chat_mp):
    """Enabling a (dead) server by name is reported as a connect error, but the
    real row's ``enabled`` flag is flipped to true — the model is told the tools
    are not live yet while the row genuinely reflects the enable."""
    # Register, then disable so there is a disabled row to enable.
    svc = McpClientService()
    server = svc.add_server(name="beta", host=_DEAD_HOST, headers={}, enabled=False)

    out = ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "enable", "name": "beta"}
    )

    assert "[mcp_manager(status=error, code=mcp-unreachable" in out
    assert "status=success" not in out
    assert "beta" in out

    row = db.execute(
        "SELECT enabled FROM mcp_client_servers WHERE id = ?", (server["id"],)
    ).fetchone()
    assert row[0] == 1  # the enable genuinely happened


# ── 7. disable by name → success {id,name,enabled:false} and DB row disabled ────


def test_disable_by_name_succeeds_and_disables_row(db, chat_mp):
    """Disabling a server by name returns a structured success body
    ``{"id","name","enabled":false}`` and the real row is disabled."""
    svc = McpClientService()
    server = svc.add_server(name="gamma", host=_DEAD_HOST, headers={}, enabled=True)

    out = ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "disable", "name": "gamma"}
    )

    assert "[mcp_manager(status=success" in _head(out)
    body = _parse_body(out)
    assert body == {"id": server["id"], "name": "gamma", "enabled": False}

    row = db.execute(
        "SELECT enabled FROM mcp_client_servers WHERE id = ?", (server["id"],)
    ).fetchone()
    assert row[0] == 0


# ── 8. enable/disable with neither server_id nor name → missing-params ──────────


def test_enable_without_target_is_missing_params(db, chat_mp):
    """Enable with neither server_id nor name is ``code=missing-params`` (from
    run()'s resolution ladder), not a success body."""
    out = ToolDispatcher(chat_mp).dispatch("mcp_manager", {"action": "enable"})

    assert "[mcp_manager(status=error, code=missing-params" in out
    assert "status=success" not in out


def test_disable_without_target_is_missing_params(db, chat_mp):
    """Disable with neither server_id nor name is ``code=missing-params``."""
    out = ToolDispatcher(chat_mp).dispatch("mcp_manager", {"action": "disable"})

    assert "[mcp_manager(status=error, code=missing-params" in out
    assert "status=success" not in out


# ── 9. enable/disable with an unknown name → not-found ──────────────────────────


def test_enable_unknown_name_is_not_found(db, chat_mp):
    """Enabling a name that matches no configured server is ``code=not-found``
    (not a success body, not a swallowed no-op)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "enable", "name": "nope"}
    )

    assert "[mcp_manager(status=error, code=not-found" in out
    assert "status=success" not in out


def test_disable_unknown_name_is_not_found(db, chat_mp):
    """Disabling a name that matches no configured server is ``code=not-found``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "mcp_manager", {"action": "disable", "name": "nope"}
    )

    assert "[mcp_manager(status=error, code=not-found" in out
    assert "status=success" not in out


# ── 10. _classify_sync_error — pure function (direct call sanctioned) ───────────


def test_classify_sync_error_auth_markers():
    """Auth-rejection markers (401/403/unauthorized/forbidden, case-insensitive)
    classify to ``auth-failed``; everything else to ``mcp-unreachable``. This is a
    pure, side-effect-free function so a direct call is sanctioned."""
    assert _classify_sync_error("HTTP 401 Unauthorized") == "auth-failed"
    assert _classify_sync_error("server returned 403") == "auth-failed"
    assert _classify_sync_error("Forbidden") == "auth-failed"
    assert _classify_sync_error("FORBIDDEN access") == "auth-failed"
    assert _classify_sync_error("Connection refused") == "mcp-unreachable"
    assert _classify_sync_error("timed out") == "mcp-unreachable"
    assert _classify_sync_error("") == "mcp-unreachable"
