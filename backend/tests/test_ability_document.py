# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Document-specific business-logic tests migrated from the per-ability
conformance file removed in . The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
document ability's genuine behaviour tests (fuzzy-delete disambiguation,
upload structured body, search rows/count) that have no coverage elsewhere."""

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import allow_policy, parse_body, seed_transcript

if TYPE_CHECKING:
    from services.document_service import DocumentService

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="manage my documents")


def _allow_delete(db: sqlite3.Connection, channel: str = "chat") -> None:
    """Flip the real ``policy`` table so ``document.delete`` is ``allow``.
    Without this, a headless test channel would block on an ``ask`` posture."""
    allow_policy(db, "document.delete", channel=channel)


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> _MP:
    """Real chat-channel MP with a seeded transcript and ``document.delete`` flipped to allow."""
    _allow_delete(db)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _service() -> "DocumentService":
    """A real ``DocumentService`` bound to the test database — built off the
    same ``get_shared_db_service()`` singleton the ``db`` fixture patches, exactly
    as production ``DocumentAbility.run`` builds it."""
    from services.database_service import get_shared_db_service
    from services.document_service import DocumentService

    return DocumentService(get_shared_db_service())


def _ingest(db: sqlite3.Connection, tmp_path: Path, name: str, content: str) -> str:
    """Ingest a real file through the SAME mechanical path production uses
    (``ingest_file``) — writes a real file, creates the real DB row, runs the real
    synchronous extraction. Returns the new document id."""
    from abilities.document import ingest_file

    src = tmp_path / name
    src.write_text(content, encoding="utf-8")
    result = ingest_file(_service(), str(src), name=name)
    assert "error" not in result, result
    return cast(str, result["id"])


def _parse_body(rendered: str, tool: str = "document") -> object:
    """Extract and JSON-parse the body between the open tag and ``[end:<tool>]`` —
    proves the envelope the model receives is structured and machine-parseable."""
    return parse_body(rendered, tool)


def test_delete_by_ambiguous_name_lists_candidates_and_deletes_nothing(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """Three documents whose names all CONTAIN 'invoice'; ``delete name='invoice'``
    must return ``code=ambiguous-match``, list all three candidates with their ids,
    and leave ALL THREE rows intact. This is the consent-gating regression."""
    id_jan = _ingest(db, tmp_path, "invoice-jan.md", "January billing total 100 EUR.")
    id_feb = _ingest(db, tmp_path, "invoice-feb.md", "February billing total 200 EUR.")
    id_mar = _ingest(db, tmp_path, "invoice-mar.md", "March billing total 300 EUR.")

    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "delete", "name": "invoice", "act_summary": "x"}
    )

    assert "[document(status=error" in out, out
    assert "code=ambiguous-match" in out, out
    # Every candidate id must be present so the model can re-call with the exact id.
    for [document_id] in (id_jan, id_feb, id_mar):
        assert [document_id] in out, out

    # The hard guarantee: NOT ONE document was deleted.
    service = _service()
    for [document_id] in (id_jan, id_feb, id_mar):
        doc = service.get_document([document_id])
        assert doc is not None, [document_id]
        assert doc.get("deleted_at") is None, ([document_id], doc.get("deleted_at"))

    # The act-trail recorded the same error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("code=ambiguous-match" in cast(str, row["result"]) for row in trail), trail


def test_delete_single_non_exact_fuzzy_match_is_ambiguous_not_first_hit(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """A name that matches exactly ONE document, but only as a non-exact substring,
    must STILL be treated as ambiguous (force an exact id) — never silently
    deleted. Only an exact-name or id match proceeds."""
    [document_id] = _ingest(db, tmp_path, "annual-report-2025.md", "Revenue grew 12 percent.")

    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "delete", "name": "annual", "act_summary": "x"}
    )

    assert "code=ambiguous-match" in out, out
    assert [document_id] in out, out

    assert cast(dict[str, object], _service().get_document([document_id])).get("deleted_at") is None


def test_delete_by_exact_id_succeeds_and_row_is_gone(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """Deletion by an exact id proceeds — the row is soft-deleted."""
    keep = _ingest(db, tmp_path, "invoice-jan.md", "January billing.")
    target = _ingest(db, tmp_path, "invoice-feb.md", "February billing.")

    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "delete", "id": target, "act_summary": "x"}
    )

    assert "[document(status=success" in out, out
    service = _service()
    assert cast(dict[str, object], service.get_document(target)).get("deleted_at") is not None
    # The sibling is untouched.
    assert cast(dict[str, object], service.get_document(keep)).get("deleted_at") is None


def test_delete_by_unique_exact_name_succeeds(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """A single document whose name matches the query EXACTLY proceeds to delete —
    the unambiguous-match path."""
    [document_id] = _ingest(db, tmp_path, "taxes.md", "Tax notes for the year.")

    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "delete", "name": "taxes.md", "act_summary": "x"}
    )

    assert "[document(status=success" in out, out
    assert cast(dict[str, object], _service().get_document([document_id])).get("deleted_at") is not None


def test_delete_unknown_name_is_not_found(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """A name that matches nothing returns ``code=not-found`` — distinct from
    ambiguous-match."""
    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "delete", "name": "does-not-exist", "act_summary": "x"}
    )

    assert "[document(status=error" in out, out
    assert "code=not-found" in out, out


def test_upload_renders_structured_body_and_seed_regex_links_doc(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """``upload`` (mechanical, internal) renders a structured JSON body
    ``{"id","hash","name","status"}`` and the turn-0 seed path parses the id back
    out of the rendered envelope to build the transcript-doc link."""
    from services.tmp_storage import new_tmp_path

    # Upload requires a path under the Chalie temp prefix (the seed guard).
    src = new_tmp_path("_doc_upload.md")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("Quarterly numbers and commentary.")

    try:
        out = ToolDispatcher(chat_mp).dispatch(
            "document", {"action": "upload", "path": src, "name": "quarterly.md"}
        )
    finally:
        if os.path.isfile(src):
            os.remove(src)

    assert "[document(status=success" in out, out
    body = cast(dict[str, object], _parse_body(out))
    assert set(body) >= {"id", "hash", "name", "status"}, body
    assert body["name"] == "quarterly.md"
    assert body["status"] == "ready"
    assert len(cast(str, body["hash"])) == 64  # sha256 hex

    # The production seed consumer parses the id straight out of the rendered
    # envelope (no schema change to message_processor's parser may break this).
    from services.message_processor import _SEED_UPLOAD_ID_RE
    match = _SEED_UPLOAD_ID_RE.search(out)
    assert match is not None, out
    assert match.group(1) == body["id"]


def test_search_renders_rows_and_count_meta(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """``search`` renders a JSON list of document rows and a ``count=`` meta."""
    _ingest(db, tmp_path, "coral.md", "Coral bleaching is driven by warming oceans and heat stress.")

    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "search", "query": "coral bleaching", "act_summary": "x"}
    )

    assert "[document(status=success" in out, out
    assert "count=" in out, out
    body = cast(list[dict[str, object]], _parse_body(out))
    assert isinstance(body, list), body
    assert any("coral.md" in (cast(str, row.get("name")) or "") for row in body), body


def test_search_empty_is_success_count_zero(db: sqlite3.Connection, chat_mp: _MP, tmp_path: Path) -> None:
    """An empty search is a SUCCESS with ``count=0`` and an empty list — not an
    error and not prose."""
    out = ToolDispatcher(chat_mp).dispatch(
        "document", {"action": "search", "query": "nonexistent-topic-xyz", "act_summary": "x"}
    )

    assert "[document(status=success" in out, out
    assert "count=0" in out, out
    assert _parse_body(out) == []
