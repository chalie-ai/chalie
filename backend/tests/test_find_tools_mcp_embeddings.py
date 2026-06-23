from pathlib import Path
from typing import cast

import pytest

from abilities._result import ToolResult
from abilities.find_tools import FindToolsAbility
from services.file_mapper_service import FileMapperService
from services.mcp_client_service import McpClientService, _open_tools_db
from services.message_processor import MessageProcessor
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stub_proc() -> MessageProcessor:
    # The discovery roster is global (AbilityRegistry.discoverable_names()); these
    # tests seed ONLINE MCP servers, whose tool names enter the effective allow-list
    # via run()'s _get_online_mcp_names() — independent of any per-config list.
    proc = object.__new__(MessageProcessor)
    proc.config = make_stub_config()
    proc._active_tools = []
    return proc


def _run_find_tools(ability: FindToolsAbility, proc: MessageProcessor, params: dict[str, object]) -> ToolResult:
    ability.mp = proc
    return ability.run(params)


def _seed_server_online(svc: McpClientService, name: str, tools: list[dict[str, object]]) -> dict[str, object]:
    server = svc.add_server(name=name, host=f"http://fake-{name}.lan:9000",
                             headers={}, enabled=True)
    svc._write_tools(cast(str, server["id"]), name, tools)
    with svc._db.connection() as conn:
        conn.execute(
            "UPDATE mcp_client_servers SET status='online' WHERE id=?",
            (server["id"],),
        )
        conn.commit()
    return server


# Representative tool definitions from the task tracker MCP — match the kinds of tools this targets.
_TASKIE_TOOLS: list[dict[str, object]] = [
    {
        "name": "list_tickets",
        "description": "List tickets in a project, optionally filtered by status or assignee.",
        "inputSchema": {"type": "object", "properties": {"project_id": {"type": "string"}}, "required": []},
    },
    {
        "name": "create_ticket",
        "description": "Create a new bug ticket or task in the tracker.",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
    },
    {
        "name": "get_document",
        "description": "Retrieve a document by ID from the project document store.",
        "inputSchema": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
    },
    {
        "name": "add_attachment",
        "description": "Attach a file to a ticket or document record.",
        "inputSchema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]},
    },
]


# ---------------------------------------------------------------------------
# Test 1 — select advertised name
# ---------------------------------------------------------------------------

class TestSelectAdvertisedName:

    def test_bare_display_name_resolves_to_prefixed_call_name(
        self, db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TDD at baseline: _run_select has no display-alias map → found=0.

        After the fix: _run_select resolves bare name to prefixed form via
        get_online_mcp_tools_index().
        """
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        # _MCP_DB_PATH is a ClassVar resolved at import time from FileMapperService;
        # redirect it to the same tmp_path so _query_mcp / embed_server_tools open
        # the correct fresh DB.
        monkeypatch.setattr(FindToolsAbility, "_MCP_DB_PATH",
                            tmp_path / "mcp_tools.sqlite")

        svc = McpClientService()
        _seed_server_online(svc, "taskie", _TASKIE_TOOLS)

        # Discoverable: the prefixed MCP names (what effective_allow will contain).
        mcp_names = svc.get_online_mcp_tool_names()
        assert "_mcp_taskie_list_tickets" in mcp_names, (
            "Seed failed: _mcp_taskie_list_tickets should be in online tool names"
        )

        ability = FindToolsAbility()
        # No ability names in discoverable — MCP names come from get_online_mcp_tool_names()
        # which the real run() calls against the db fixture.
        proc = _stub_proc()
        result = _run_find_tools(ability, proc, {"select": ["list_tickets"]})

        assert "_mcp_taskie_list_tickets" in proc.active_tools, (
            f"Expected '_mcp_taskie_list_tickets' in active_tools after select=['list_tickets']. "
            f"Got: {proc.active_tools}. Result: {result!r}. "
            f"Failure at baseline: _run_select has no display-alias map."
        )


# ---------------------------------------------------------------------------
# Test 2 — semantic query via embed_server_tools + real EmbeddingService
# ---------------------------------------------------------------------------

class TestSemanticQueryViaEmbeddings:

    def test_query_create_task_or_ticket_returns_taskie_tools(
        self, db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_server_tools() generates embeddings via EmbeddingService (ONNX).

        At baseline: AttributeError. After the fix: dual-signal RRF surfaces
        list_tickets / create_ticket for query 'create a task or ticket'.
        """
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(FindToolsAbility, "_MCP_DB_PATH",
                            tmp_path / "mcp_tools.sqlite")

        svc = McpClientService()
        server = _seed_server_online(svc, "taskie", _TASKIE_TOOLS)

        # embed_server_tools is the new method — this MUST fail at baseline.
        svc.embed_server_tools(cast(str, server["id"]))

        ability = FindToolsAbility()
        proc = _stub_proc()
        result = _run_find_tools(ability, proc, {"query": "create a task or ticket"})

        ticket_tools = {"_mcp_taskie_list_tickets", "_mcp_taskie_create_ticket"}
        found = set(proc.active_tools) & ticket_tools
        assert found, (
            f"Expected at least one ticket tool in active_tools for query "
            f"'create a task or ticket'. Got: {proc.active_tools}. Result: {result!r}. "
            f"Failure at baseline: embed_server_tools absent."
        )


# ---------------------------------------------------------------------------
# Test 3 — loose multi-word query rescued by vector signal
# ---------------------------------------------------------------------------

class TestLooseMultiWordQuery:

    def test_multi_word_query_returns_results_via_vector_signal(
        self, db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A required keyword term that only appears in one tool's description lets
        the vector signal fill in the rest. Under the keyword grammar, '+attachment'
        is a required term (substring via trigram) and 'ticket document' are bare
        optional terms that OR-boost other tools. add_attachment must appear because
        its description contains 'attachment'.

        At baseline: no vec table → empty result.
        After the fix: FTS keyword path surfaces add_attachment directly.
        """
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(FindToolsAbility, "_MCP_DB_PATH",
                            tmp_path / "mcp_tools.sqlite")

        svc = McpClientService()
        server = _seed_server_online(svc, "taskie", _TASKIE_TOOLS)
        svc.embed_server_tools(cast(str, server["id"]))

        # Precondition: '+attachment' as a required FTS term matches add_attachment.
        # Under the new trigram tokenizer, "attachment" is a substring of the tool
        # name and/or description. At least 1 row must match so the vector can
        # act as the confirming second signal.
        from abilities._search import build_keyword_query
        fts_match, _ = build_keyword_query("+attachment ticket document")
        conn = _open_tools_db()
        try:
            fts_rows = conn.execute(
                "SELECT COUNT(*) FROM mcp_tools_fts WHERE mcp_tools_fts MATCH ?",
                (fts_match,),
            ).fetchone()[0]
        finally:
            conn.close()
        assert fts_rows >= 1, (
            f"Precondition: FTS keyword query '{fts_match}' must match at least 1 row "
            f"(add_attachment). Got fts_rows={fts_rows}. Check test data."
        )

        ability = FindToolsAbility()
        proc = _stub_proc()
        result = _run_find_tools(ability, proc, {"query": "+attachment ticket document"})

        assert len(proc.active_tools) >= 1, (
            f"Expected >=1 MCP tool for '+attachment ticket document'. "
            f"Got active_tools={proc.active_tools}. Result: {result!r}. "
            f"Failure at baseline: embed_server_tools absent, no vec table."
        )


# ---------------------------------------------------------------------------
# Test 4 — embeddings survive a heartbeat re-sync (id churn)
# ---------------------------------------------------------------------------

class TestEmbeddingsSurviveHeartbeatSync:

    def test_query_still_returns_tools_after_write_tools_rechurn(
        self, db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After embed_server_tools() writes vector rows keyed by tool_name,
        calling _write_tools() again (simulating a heartbeat re-sync) deletes and
        re-inserts mcp_tools rows — reassigning mcp_tools.id to HIGHER values.

        The vectors must STILL be queryable because embed_server_tools keys
        mcp_tool_vectors by the stable tool_name (not by the volatile id).

        ID-churn strategy (§11 adjudication):
        SQLite (INTEGER PRIMARY KEY, no AUTOINCREMENT) reuses freed rowids 1..N when a
        table is otherwise empty.  In a single-server isolated DB, taskie's DELETE+INSERT
        would get back ids 1..4 — making ids_before == ids_after and the precondition
        unsatisfiable.  Fix: seed a SECOND server FIRST so the mcp_tools rowid
        high-water mark exceeds taskie's id range.  Its DELETE+INSERT then
        reassigns ids above that mark, guaranteeing ids_before != ids_after.

        At baseline: embed_server_tools absent → AttributeError.
        After the fix: vectors survive because the join goes through tool_name.

        NOTE: Calls real ONNX model — intentionally slow.
        """
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(FindToolsAbility, "_MCP_DB_PATH",
                            tmp_path / "mcp_tools.sqlite")

        svc = McpClientService()

        # Seed taskie FIRST so its tools get the lowest rowids (e.g. 1..4).
        server = _seed_server_online(svc, "taskie", _TASKIE_TOOLS)
        svc.embed_server_tools(cast(str, server["id"]))

        # Seed a SECOND server AFTER taskie to advance the mcp_tools rowid high-water
        # mark past taskie's id range (e.g. id=5).  When taskie's _write_tools then
        # DELETEs ids 1..4 and re-INSERTs, SQLite picks max(existing)+1 = 6, 7, 8, 9
        # — guaranteeing ids_before != ids_after without AUTOINCREMENT.
        # (SQLite without AUTOINCREMENT reuses freed rowids when they are below the
        # current max; seeding "other" after taskie keeps max > taskie's range.)
        _seed_server_online(svc, "other", [
            {
                "name": "ping",
                "description": "Health-check the other service.",
                "inputSchema": {},
            }
        ])

        # Capture the mcp_tools.id values before the churn.
        conn = _open_tools_db()
        try:
            ids_before = {
                r[0] for r in conn.execute(
                    "SELECT id FROM mcp_tools WHERE server_id=?", (server["id"],)
                ).fetchall()
            }
        finally:
            conn.close()

        # Simulate a heartbeat re-sync: _write_tools deletes + re-inserts.
        # With the second server having advanced the high-water mark, taskie's
        # re-insert gets fresh ids above the previous range.
        svc._write_tools(cast(str, server["id"]), "taskie", _TASKIE_TOOLS)

        conn = _open_tools_db()
        try:
            ids_after = {
                r[0] for r in conn.execute(
                    "SELECT id FROM mcp_tools WHERE server_id=?", (server["id"],)
                ).fetchall()
            }
        finally:
            conn.close()

        assert ids_before != ids_after, (
            "Precondition: _write_tools must reassign mcp_tools.id to new values. "
            "The second server was seeded first to advance SQLite's rowid high-water mark. "
            f"ids_before={ids_before}, ids_after={ids_after}. "
            "If ids are still the same, the high-water mark strategy needs review."
        )

        # Vector query must still work despite the id churn — keyed by tool_name.
        # 'create a task or ticket' is dual-signal (vec + FTS) → RRF 0.125 → passes floor.
        ability = FindToolsAbility()
        proc = _stub_proc()
        result = _run_find_tools(ability, proc, {"query": "create a task or ticket"})

        assert len(proc.active_tools) >= 1, (
            f"Embeddings should survive _write_tools id churn (keyed by tool_name). "
            f"Got active_tools={proc.active_tools}. Result: {result!r}. "
            f"Failure at baseline: embed_server_tools absent / keyed by volatile id."
        )


# ---------------------------------------------------------------------------
# Test 5 — add-only trigger: heartbeat re-sync does NOT regenerate embeddings
# ---------------------------------------------------------------------------

class TestAddOnlyEmbeddingTrigger:

    def test_write_tools_does_not_alter_embedding_rows(
        self, db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After embed_server_tools() writes vector rows, calling _write_tools()
        again (simulating a heartbeat or enable sync) must NOT regenerate or
        alter the mcp_tool_vectors rows.

        Verified by: snapshot the (rowid, tool_name) pairs in mcp_tool_vectors
        before and after the sync; they must be identical.

        At baseline: embed_server_tools absent → AttributeError.
        After the fix: _write_tools is intentionally NOT a trigger for re-embedding;
        the vector rows are stable across non-add syncs.
        """
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(FindToolsAbility, "_MCP_DB_PATH",
                            tmp_path / "mcp_tools.sqlite")

        svc = McpClientService()
        server = _seed_server_online(svc, "taskie", _TASKIE_TOOLS)
        svc.embed_server_tools(cast(str, server["id"]))

        # Snapshot the vector rows (rowid + tool_name pairs) before the sync.
        conn = _open_tools_db()
        try:
            rows_before = set(conn.execute(
                "SELECT rowid, tool_name FROM mcp_tool_vectors ORDER BY rowid"
            ).fetchall())
        finally:
            conn.close()

        assert rows_before, "Precondition: embed_server_tools must have written mcp_tool_vectors rows."

        # Simulate a heartbeat: _write_tools is the re-sync path.
        svc._write_tools(cast(str, server["id"]), "taskie", _TASKIE_TOOLS)

        conn = _open_tools_db()
        try:
            rows_after = set(conn.execute(
                "SELECT rowid, tool_name FROM mcp_tool_vectors ORDER BY rowid"
            ).fetchall())
        finally:
            conn.close()

        assert rows_before == rows_after, (
            f"_write_tools (heartbeat) must NOT alter mcp_tool_vectors rows. "
            f"Before: {sorted(rows_before)}. After: {sorted(rows_after)}. "
            f"Failure mode: embedding trigger is wired to _write_tools (wrong) "
            f"instead of add-only (correct)."
        )


# ---------------------------------------------------------------------------
# Test 6 — ambiguous bare name dropped: two servers with same bare tool name
# ---------------------------------------------------------------------------

class TestAmbiguousBareNameDropped:

    def test_bare_name_colliding_across_servers_is_not_silently_resolved(
        self, db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Alias resolution drops ambiguous bare names and keeps unambiguous ones.

        Phase A (unambiguous): one server exposes 'ping'.  Bare name 'ping' must
        resolve to '_mcp_taskie_ping'.  This FAILS at baseline because _run_select
        has no alias map at all — same failure mode as Test 1.

        Phase B (ambiguous): a second server also exposes 'ping'.  Now bare 'ping'
        maps to two prefixed call names and the alias must be DROPPED — the caller
        must use the prefixed form.  Neither server's tool must be activated.

        At baseline: Phase A assertion fails (no alias map → found=0 for bare name).
        After the fix: Phase A passes (unambiguous alias resolved); Phase B passes
        (ambiguous alias dropped, found=0, prefixed form required).
        """
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(FindToolsAbility, "_MCP_DB_PATH",
                            tmp_path / "mcp_tools.sqlite")

        svc = McpClientService()

        # Phase A: single server — bare name 'ping' is unambiguous.
        _seed_server_online(svc, "taskie", [
            {
                "name": "ping",
                "description": "Ping the taskie service.",
                "inputSchema": {},
            }
        ])

        ability_a = FindToolsAbility()
        proc_a = _stub_proc()
        result_a = _run_find_tools(ability_a, proc_a, {"select": ["ping"]})

        assert "_mcp_taskie_ping" in proc_a.active_tools, (
            f"Unambiguous bare name 'ping' must resolve to '_mcp_taskie_ping' "
            f"when only one server exposes it. active_tools={proc_a.active_tools}. "
            f"Result: {result_a!r}. "
            f"Failure at baseline: _run_select has no display-alias map."
        )

        # Phase B: add a second server with the same bare tool name → now ambiguous.
        _seed_server_online(svc, "other", [
            {
                "name": "ping",
                "description": "Ping the other service.",
                "inputSchema": {},
            }
        ])

        mcp_names = svc.get_online_mcp_tool_names()
        assert "_mcp_taskie_ping" in mcp_names
        assert "_mcp_other_ping" in mcp_names

        ability_b = FindToolsAbility()
        proc_b = _stub_proc()
        result_b = _run_find_tools(ability_b, proc_b, {"select": ["ping"]})

        assert "_mcp_taskie_ping" not in proc_b.active_tools, (
            f"Ambiguous bare 'ping' must not silently pick _mcp_taskie_ping. "
            f"active_tools={proc_b.active_tools}. Result: {result_b!r}"
        )
        assert "_mcp_other_ping" not in proc_b.active_tools, (
            f"Ambiguous bare 'ping' must not silently pick _mcp_other_ping. "
            f"active_tools={proc_b.active_tools}. Result: {result_b!r}"
        )

        # The prefixed form must always resolve unambiguously.
        proc_c = _stub_proc()
        ability_c = FindToolsAbility()
        _run_find_tools(ability_c, proc_c, {"select": ["_mcp_taskie_ping"]})
        assert "_mcp_taskie_ping" in proc_c.active_tools, (
            f"Prefixed form '_mcp_taskie_ping' must always resolve. "
            f"active_tools={proc_c.active_tools}"
        )
