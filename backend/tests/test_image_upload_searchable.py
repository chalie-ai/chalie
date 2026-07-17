"""Feature test: an uploaded image is findable by its visual content.

END-TO-END SEARCHABILITY PROOF — the heart of the Vision Subagent addendum.

Drives the REAL ``DocumentAbility.run`` upload + search entry points against the
real test DB and real files. ZERO mocks: the OCR description produced by the
no-vision-provider fork flows through the SAME production pipeline
(``create_document_artifacts`` -> data_graph embed + FTS5) that every upload
uses, and ``document(action='search')`` recalls it via the REAL recall path
(FTS5 + vector). Proves the doc_id guardrail is truthful: an image with words
becomes a searchable document.

Second test proves a description is MANDATORY: a textless image with no vision
provider yields no description on either rung (provider, then OCR), so the
upload FAILS VISIBLY rather than persisting a hollow, contentless document.

The OCR fixture word ('INVOICE') was EMPIRICALLY confirmed readable by RapidOCR
in this environment via ``image_context_service.analyze`` — ``ocr_text ==
'INVOICE'``. Both fixtures come from tests.helpers so the bytes are described in
exactly one place.
"""

import sqlite3
from typing import cast

import pytest

from abilities.document import DocumentAbility
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path
from tests.helpers import blank_png_bytes, ocrable_png_bytes

pytestmark = pytest.mark.unit


def _png_at_tmp_path(name: str, data: bytes) -> str:
    """Materialise real bytes under the temp prefix the upload pipeline accepts."""
    path = new_tmp_path(name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _drain_search_index() -> None:
    """Drive the REAL async search-expander pipeline synchronously against the
    bound test DB — the exact production code path, no mocks. In prod the
    search_expander_worker daemon does this continuously; a test must do it
    explicitly because no worker runs under pytest."""
    from services.search_expander_service import SearchExpanderService
    svc = SearchExpanderService()
    svc._self_heal()
    item = svc._dequeue()
    while item is not None:
        svc._process(item)
        item = svc._dequeue()


def test_uploaded_image_is_findable_via_document_search(db: sqlite3.Connection) -> None:
    """No vision provider -> OCR description -> the existing embed+FTS5 pipeline ->
    document.search finds the image by content (real recall, FTS5 + vector)."""
    ProviderDbService().set_vision_provider(None)

    ability = DocumentAbility(mp=None)
    up = ability.run({
        "action": "upload",
        "path": _png_at_tmp_path("inv.png", ocrable_png_bytes()),
    })
    # : upload now returns a structured ToolResult body
    # ``{"id","hash","name","status"}`` (Task 6 hash preserved as a body key).
    assert up.status == "success", up
    assert cast(dict[str, object], up.body)["id"], up.body
    assert cast(dict[str, object], up.body)["hash"], up.body

    # The FTS/vec posting is written by the async search-expander pipeline
    # (services/search_expander_service.py), never synchronously by save() —
    # drive it explicitly since no worker daemon runs under pytest.
    _drain_search_index()

    found = ability.run({"action": "search", "query": "invoice"})
    # : search returns a JSON list of document rows; the matched image
    # surfaces via the REAL recall path under its original_name.
    assert any("inv.png" in cast(str, cast(dict[str, object], row).get("name") or "") for row in cast(list[object], found.body)), found.body


def test_textless_image_upload_fails_visibly(db: sqlite3.Connection) -> None:
    """A textless image with no vision provider exhausts both describe rungs (no
    provider, then empty OCR). A description is mandatory, so the upload surfaces
    an ERROR — never a hollow 'ready' document with nothing indexed in it."""
    ProviderDbService().set_vision_provider(None)

    ability = DocumentAbility(mp=None)
    up = ability.run({
        "action": "upload",
        "path": _png_at_tmp_path("blank.png", blank_png_bytes()),
    })

    assert up.status == "error", up
    assert up.body, up
