"""Feature test: turn-0 attachment ingest dispatches a FILE PATH, never bytes.

Root cause of the deepseek ``400 The prompt is too long``: the turn-0
upload path base64-encoded the whole attachment and dispatched
``document.upload(content=<base64>)``.  ``ToolDispatcher.dispatch`` records every
call's params verbatim into ``tool_calls.params`` (``ActTrail.record``), and
``ActTrail.render`` inlines ``[document] {params} -> {result}`` straight into the
next prompt.  A ~3 MB image became a ~4 MB base64 string sitting in the act-trail,
blowing the context window — structurally, not occasionally.

The fix is an invariant, not a redaction band-aid: ``document.upload`` takes a
**path**, never bytes.  A path is ~50 chars, so the act-trail can never carry a
file blob again.  This test pins that invariant by driving the REAL production
hot path:

    MessageProcessor._seed_turn_zero()  (real UserConfig, real PNG on disk,
    real document.upload ingest, real OCR fork — no vision provider)

and asserting the trail row the dispatch produced:

  * exactly one ``document`` tool_call row for this transcript,
  * its params carry the attachment PATH and NO ``content`` / base64 blob,
  * the whole params JSON stays small (a path can't bloat the window),
  * the upload still landed a ``ready`` document row with a content hash
    (the ``"id":``/``"hash":`` keys of the structured upload body the
    transcript-doc link depends on — ).

Zero mocks.  The upload never touches the LLM boundary (no vision provider -> the
deterministic OCR fork), so no provider seam is needed.
"""

import io
import json
import re
import sqlite3
from typing import cast

import pytest

from configs.channels import UserConfig
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.message_processor import MessageProcessor
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path

pytestmark = pytest.mark.unit


def _png_with_text(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, 72)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((20, 40), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_attachment(label: str) -> str:
    """Write a PNG under the Chalie temp prefix (so the ingest path guard passes)."""
    path = new_tmp_path(f"upload_path_{label}.png")
    with open(path, "wb") as fh:
        fh.write(_png_with_text(label.upper()))
    return path


def _build_parent(attachments: list[str]) -> MessageProcessor:
    """A real UserConfig MessageProcessor in the exact state ``_seed_turn_zero``
    fires from: input row written (by the real constructor), ``active_tools``
    seeded, attachments on metadata, thinking OFF (no LLM boundary involved in
    this test — ``_seed_turn_zero`` is invoked directly so the attachment
    ingest is isolated from the chat-completion step)."""
    config = UserConfig()
    parent = MessageProcessor(config, -1, "What is in this image?", {"attachments": attachments})
    parent.active_tools = list(config.always_available or [])
    parent.thinking_level = "low"
    return parent


def test_turn0_upload_records_path_not_base64_blob(db: sqlite3.Connection) -> None:
    """The turn-0 ``document.upload`` trail row carries a PATH, not file bytes.

    Pre-fix the dispatch sends ``content=<base64>`` so ``tool_calls.params`` holds
    a multi-KB+ blob -> the assertions below FAIL.  Post-fix it sends ``path=`` so
    the row is tiny and base64-free -> they PASS.
    """
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)  # OCR fork

    attachment = _write_attachment("invoice")
    parent = _build_parent([attachment])

    # The REAL production method: memory seed -> upload barrier -> (no thinking).
    parent._seed_turn_zero()

    rows = get_shared_db_service().fetch_all(
        "SELECT params, result FROM tool_calls "
        "WHERE transcript_id = ? AND tool_name = 'document' ORDER BY id",
        (parent.uid,),
    )
    assert len(rows) == 1, f"expected exactly one document upload trail row, got {len(rows)}"

    params_raw = cast(str, rows[0]["params"])
    params = json.loads(params_raw)

    assert params.get("action") == "upload"
    # The structural invariant: a path went over the wire, not bytes.
    assert "content" not in params, (
        "upload dispatched file bytes via 'content' — the act-trail will carry a "
        "base64 blob and blow the context window"
    )
    assert params.get("path") == attachment, (
        f"upload must dispatch the file PATH; got params={params!r}"
    )
    # A path is tiny; a base64 image is not. This bounds the act-trail row hard.
    assert len(params_raw) < 4096, (
        f"upload trail params is {len(params_raw)} bytes — far larger than a path, "
        "so a file blob leaked into the act-trail"
    )

    # The upload still landed a real, ready document with a content hash — the
    # structured ``{"id","hash",...}`` body the chat refresh link parses ().
    result = cast(str, rows[0]["result"])
    assert '"id":"' in result and '"hash":"' in result, (
        f"upload result lost its id/hash keys: {result!r}"
    )
    from services.message_processor import _SEED_UPLOAD_ID_RE
    doc_id = cast(re.Match[str], _SEED_UPLOAD_ID_RE.search(result)).group(1)

    doc = DocumentService(get_shared_db_service()).get_document(doc_id)
    assert doc is not None, f"document {doc_id} was not persisted"
    assert doc.get("status") == "ready", f"document {doc_id} not ready: {doc.get('status')!r}"
    assert doc.get("file_hash"), f"document {doc_id} has no content hash"
