# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

_TEST_PORT = 18462


@pytest.fixture(scope="module", autouse=True)
def _patch_db(_db_template: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """Real, fully-converged schema (built from schema.sql via the shared
    session-scoped ``_db_template`` fixture in ``tests/conftest.py``) copied
    into a module-local file, then installed as the process-wide singleton
    AND pointed to by the ``Database`` gateway.

    Only the singleton *value* (``database_service._shared_db_service``) is
    swapped, never the ``get_shared_db_service`` function object itself — see
    the import-time-binding hazard documented in ``tests/conftest.py``.
    Modules that did ``from services.database_service import
    get_shared_db_service`` at import time keep a reference to the real
    function, which always re-reads the (module-global) singleton value, so
    swapping only the value keeps those modules correctly wired both during
    the test and after teardown restores the prior value.

    The server's own DB access (``BearerTokenMiddleware``) goes through the
    ``Database`` gateway, which resolves its path via
    ``FileMapperService.get_db_path()`` at call time rather than consulting
    the singleton above — so that path is redirected here too, mirroring
    ``tests/conftest.py``'s ``db`` fixture and
    ``tests/test_schema_convergence.py``'s ``_gateway_to_tmp_db`` fixture.
    The built-in ``monkeypatch`` fixture is function-scoped and cannot be
    used from this module-scoped fixture, so ``pytest.MonkeyPatch()`` is
    used directly and undone by hand in teardown.
    """
    import shutil

    from services import database as _newdb
    from services import database_service
    from services.file_mapper_service import FileMapperService

    db_dir = tmp_path_factory.mktemp("mcp_test_db")
    db_path = str(db_dir / "test.db")
    shutil.copy2(_db_template, db_path)

    test_db = database_service.DatabaseService(db_path)

    database_service._local.conn = None
    database_service._local.db_path = None

    original = database_service._shared_db_service
    database_service._shared_db_service = test_db

    mp = pytest.MonkeyPatch()
    mp.setattr(FileMapperService, "get_db_path", lambda *_: Path(db_path))
    _newdb.Database.close()

    yield test_db

    test_db.close_pool()
    database_service._shared_db_service = original
    database_service._local.conn = None
    database_service._local.db_path = None
    _newdb.Database.close()
    mp.undo()


@pytest.fixture(scope="module")
def auth_token(_patch_db: object) -> str:
    from services.database_service import DatabaseService
    from services.wrapper_auth_service import WrapperAuthService

    auth_svc = WrapperAuthService(cast(DatabaseService, _patch_db))
    raw_token, _ = auth_svc.create_token(
        name="E2E Test Agent",
        wrapper_id_override="__e2e_test__",
    )
    return raw_token


@pytest.fixture(scope="module")
def mcp_server(_patch_db: object) -> Iterator[dict[str, object]]:
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
    def test_unauthenticated_request_rejected(self, mcp_server: dict[str, object]) -> None:
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

    def test_invalid_token_rejected(self, mcp_server: dict[str, object]) -> None:
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
    def _mcp_request(self, url: str, method: str, params: dict[str, object], auth_token: str, session_id: str | None = None) -> tuple[object | None, str | None]:
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

    def test_tool_list_contains_talk_to_chalie(self, mcp_server: dict[str, object], auth_token: str) -> None:
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

        tool_names = [t["name"] for t in cast(list[dict[str, object]], cast(dict[str, object], tools_result).get("tools", []))]
        assert "talk_to_chalie" in tool_names, (
            f"Expected talk_to_chalie in {tool_names}"
        )
