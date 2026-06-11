# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the vision tool's ToolResult contract (TKT-915).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch("vision", params)`` chokepoint on the CHAT channel
against a real ``mp``-shaped context, the real ``AbilityRegistry`` resolution of
the production ``VisionAbility``, the real ``PolicyManager.wrap`` gate (vision
seeds ``allow`` on chat via ``apply_seed`` in the ``db`` template — it is NOT in
``PolicyManager.INTERNAL``, so the real gate actually runs through to run()), the
real ``VisionAbility.run``, real document rows + on-disk files, a real
unreachable vision provider, and the real ``ActTrail`` write.

vision was already ToolResult-typed on every path (TKT-882), but all four error
sites carried the BANNED ``code="error"`` placeholder. This file pins TKT-915's
return-type swap: each styled error now maps onto a stable kebab code + hint,
and the OCR fallback success carries ``degraded=true``.

  * **missing-params (pre-gate + run residue).** An absent ``image`` is rejected
    by the dispatcher's ACTION_REQUIRED pre-gate as ``code=missing-params`` before
    run(); a whitespace-only ``image`` slips the truthiness pre-gate and run()
    rejects it the same way.
  * **not-found.** An unknown doc id → ``code=not-found`` with the id still
    visible in the body and a recovery hint.
  * **no-file-on-disk.** A real row with no stored ``file_path`` →
    ``code=no-file-on-disk`` with a hint.
  * **vision-failed.** A configured-but-unreachable vision provider surfaces the
    raw provider error as ``code=vision-failed`` (never swallowed, per TKT-838).
  * **degraded OCR fallback.** No vision provider configured → the OCR text +
    no-vision note returns ``ok`` BUT with ``degraded=true`` in the head so the
    model can tell a downgraded read from a real vision read.

OFFLINE-DETERMINISTIC: the unreachable provider host ``http://127.0.0.1:1`` fails
fast, and the OCR fallback (RapidOCR) runs locally — no network. Bodies are PROSE
strings rendered verbatim, not JSON.
"""

import base64
import hashlib

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from configs.channels import UserConfig
from services.act_trail import ActTrail
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.file_mapper_service import FileMapperService
from services.provider_db_service import ProviderDbService
from tests._tool_result_harness import MP, body, head, seed_transcript

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────


def _png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor. Vision seeds ``allow`` on chat in the db template, so the real gate
    passes through to the production run()."""
    return MP(seed_transcript(db, content="what is in this image"), UserConfig({}))


def _head(rendered: str) -> str:
    return head(rendered, "vision")


def _body(rendered: str) -> str:
    return body(rendered, "vision")


def _force_vision_provider(db) -> int:
    """Create a real, RESOLVABLE-but-unreachable vision provider and select it."""
    svc = ProviderDbService(get_shared_db_service())
    provider = svc.create_provider(
        {
            "name": "probe-vision",
            "platform": "ollama",
            "model": "llava",
            "host": "http://127.0.0.1:1",
            "api_key": "",
        }
    )
    pid = provider["id"]
    db.execute("UPDATE providers SET supports_vision = 1 WHERE id = ?", (pid,))
    db.commit()
    svc.set_vision_provider(pid)
    return pid


def _real_image_doc(db, rel_dir: str = "vis001") -> str:
    """Create a real document row AND materialise a real PNG at the resolved
    documents path so the ability resolves a genuine file on disk."""
    doc_svc = DocumentService(get_shared_db_service())
    doc_id = doc_svc.create_document(
        original_name="pic.png",
        mime_type="image/png",
        file_size=len(_png_bytes()),
        file_path=f"{rel_dir}/pic.png",
        file_hash=hashlib.sha256(_png_bytes()).hexdigest(),
    )
    abs_path = FileMapperService.get_documents_path(rel_dir, "pic.png")
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(_png_bytes())
    return doc_id


# ── 1. absent image param → dispatcher pre-gate missing-params ──────────────────


def test_absent_image_is_missing_params(db, chat_mp):
    """No ``image`` param at all → the ACTION_REQUIRED pre-gate rejects it as
    ``code=missing-params`` BEFORE run() — never the legacy ``code=error``."""
    out = ToolDispatcher(chat_mp).dispatch("vision", {"query": "x", "act_summary": "x"})

    assert "[vision(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "image" in out
    assert "[end:vision]" in out


# ── 2. whitespace-only image → run() residue check missing-params ───────────────


def test_whitespace_only_image_is_missing_params(db, chat_mp):
    """A whitespace-only ``image`` slips the truthiness pre-gate and reaches run(),
    which rejects the stripped-empty doc id as ``code=missing-params``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": "   ", "query": "x", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=error" in head
    assert "code=missing-params" in head
    assert "code=error]" not in out
    assert "image" in out


# ── 3. unknown doc id → not-found, id visible, hint present ─────────────────────


def test_unknown_doc_id_is_not_found(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": "deadbeef", "query": "x", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=error" in head
    assert "code=not-found" in head
    assert "code=error]" not in out
    assert "deadbeef" in _body(out)
    assert any(ln.startswith("hint:") for ln in out.splitlines())


# ── 4. real row with no stored file_path → no-file-on-disk + hint ───────────────


def test_doc_with_no_file_path_is_no_file_on_disk(db, chat_mp):
    """A real document row whose ``file_path`` is empty (no stored file) →
    ``code=no-file-on-disk`` with a recovery hint."""
    doc_id = "facefeed"
    db.execute(
        "INSERT INTO documents (id, original_name, mime_type, file_size_bytes, "
        "file_path, file_hash) VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, "pic.png", "image/png", len(_png_bytes()), "", "abc"),
    )
    db.commit()

    out = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": doc_id, "query": "x", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=error" in head
    assert "code=no-file-on-disk" in head
    assert "code=error]" not in out
    assert doc_id in _body(out)
    assert any(ln.startswith("hint:") for ln in out.splitlines())


# ── 5. provider failure surface → vision-failed, raw error visible, hint ────────


def test_provider_failure_is_vision_failed(db, chat_mp):
    """A configured-but-unreachable vision provider with a real doc + real on-disk
    PNG surfaces the raw provider error as ``code=vision-failed`` — never swallowed
    into a false success (TKT-838)."""
    _force_vision_provider(db)
    doc_id = _real_image_doc(db)

    out = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": doc_id, "query": "what is this", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=error" in head
    assert "code=vision-failed" in head
    assert "code=error]" not in out
    # The raw failure is visible, not an empty / generic message.
    assert _body(out)
    assert "Vision is currently experiencing issues" in _body(out)
    assert any(ln.startswith("hint:") for ln in out.splitlines())


# ── 6. OCR fallback success → degraded=true, note rides the body ────────────────


def test_ocr_fallback_success_is_degraded(db, chat_mp):
    """No vision provider configured → the OCR fallback returns ``ok`` BUT with
    ``degraded=true`` in the head, and the no-vision-provider note still rides the
    body verbatim."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)
    assert ProviderDbService(get_shared_db_service()).get_vision_provider() is None

    doc_id = _real_image_doc(db, rel_dir="vis002")

    out = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": doc_id, "query": "what is this", "act_summary": "x"}
    )

    head = _head(out)
    assert "status=success" in head
    assert "degraded=true" in head
    assert "vision provider" in _body(out).lower()


# ── 7. no legacy markers anywhere across the error/success envelopes ─────────────


def test_no_legacy_markers_anywhere(db, chat_mp):
    """Across the missing/not-found/no-file/provider-failure and the degraded
    success envelopes the ``code=error`` placeholder never appears."""
    absent = ToolDispatcher(chat_mp).dispatch("vision", {"query": "x", "act_summary": "x"})
    not_found = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": "deadbeef", "query": "x", "act_summary": "x"}
    )

    db.execute(
        "INSERT INTO documents (id, original_name, mime_type, file_size_bytes, "
        "file_path, file_hash) VALUES (?, ?, ?, ?, ?, ?)",
        ("0badf00d", "pic.png", "image/png", 1, "", "abc"),
    )
    db.commit()
    no_file = ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": "0badf00d", "query": "x", "act_summary": "x"}
    )

    _force_vision_provider(db)
    provider_fail = ToolDispatcher(chat_mp).dispatch(
        "vision",
        {"image": _real_image_doc(db, rel_dir="vis003"), "query": "x", "act_summary": "x"},
    )

    ProviderDbService(get_shared_db_service()).set_vision_provider(None)
    degraded = ToolDispatcher(chat_mp).dispatch(
        "vision",
        {"image": _real_image_doc(db, rel_dir="vis004"), "query": "x", "act_summary": "x"},
    )

    for out in (absent, not_found, no_file, provider_fail, degraded):
        assert "code=error]" not in out
        assert "code=error," not in out
        assert "code=error)" not in out


# ── 8. an error dispatch still records the rendered envelope on the act-trail ────


def test_error_dispatch_writes_act_trail(db, chat_mp):
    """The not-found error dispatch records the same non-empty vision envelope
    against the transcript anchor — the cross-step write really lands."""
    ToolDispatcher(chat_mp).dispatch(
        "vision", {"image": "deadbeef", "query": "x", "act_summary": "x"}
    )

    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any(
        "[vision(status=error, code=not-found" in row["result"] for row in trail
    )


# ── 9. registry metadata + schema frozen (2 required params) ─────────────────────


def test_vision_registered_with_metadata():
    ability = AbilityRegistry.get("vision")
    assert ability.get_name() == "vision"
    assert ability.get_summary()
    assert ability.get_examples()
    assert ability.get_search_tooltip()

    schema = ability.get_parameters()
    assert sorted(schema["required"]) == ["image", "query"]
