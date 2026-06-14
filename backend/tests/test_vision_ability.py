"""Feature tests for VisionAbility + describe_image.

Drives the real describe_image() core and the real VisionAbility.run() entry
point against the real test DB, with real provider rows created by the
production ProviderDbService factory. No mocks.

Covers the two never-switched forks of describe_image:
  * no vision provider  -> OCR fallback, vision_used False, a note.
  * vision provider configured-but-unreachable -> the provider error is
    SURFACED, never swallowed into a fake success.

Real no-swallow surface (verified from message_processor._loop): a provider
send_messages exception PROPAGATES out of MessageProcessor.process() — _loop
calls self.providers.send() with no try/except and _run/process do not wrap it.
So describe_image() RAISES on the provider path, and VisionAbility.run() — whose
single styled-error except wraps describe_image — returns status='error' with
the raw error visible in the message.
"""

import base64
import threading

import pytest

from services.database_service import get_shared_db_service
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.provider_db_service import ProviderDbService

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def _make_user_mp() -> MessageProcessor:
    from configs.channels.user import UserConfig

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "x", {})
    mp.config = UserConfig()
    mp.uid = None
    mp.current_iteration = 0
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    mp.thinking_override = None
    return mp


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


def test_describe_image_no_vision_provider_falls_back_to_ocr(db, tmp_path):
    from abilities.vision import describe_image

    ProviderDbService(get_shared_db_service()).set_vision_provider(None)
    assert ProviderDbService(get_shared_db_service()).get_vision_provider() is None

    img = tmp_path / "blank.png"
    img.write_bytes(_png_bytes())

    out = describe_image(
        str(img), "image/png", "what is this",
        policy_channel=ProcessorConfig.PolicyChannel.CHAT,
    )

    assert out["vision_used"] is False
    assert "description" in out
    assert out["note"] is not None
    assert "vision provider" in out["note"].lower()


def test_describe_image_provider_path_surfaces_provider_error(db, tmp_path):
    """Configured-but-unreachable vision provider: the provider error must NOT be
    swallowed. describe_image drives the real MessageProcessor.process, whose
    provider exception propagates — so describe_image RAISES."""
    from abilities.vision import describe_image

    _force_vision_provider(db)

    img = tmp_path / "blank.png"
    img.write_bytes(_png_bytes())

    with pytest.raises(Exception):  # noqa: B017,PT011 — the real provider error bubbles up
        describe_image(
            str(img), "image/png", "what is this",
            policy_channel=ProcessorConfig.PolicyChannel.CHAT,
        )


def test_vision_run_provider_error_returns_visible_error(db):
    """The styled-error surface: VisionAbility.run wraps describe_image's raise in
    its one allowed except and returns status='error' with the raw error visible
    — never a false success / empty string."""
    from abilities.vision import VisionAbility
    from services.document_service import DocumentService
    from services.file_mapper_service import FileMapperService

    _force_vision_provider(db)

    # A real document row + a real on-disk image file the ability resolves.
    import hashlib

    doc_svc = DocumentService(get_shared_db_service())
    doc_id = doc_svc.create_document(
        original_name="pic.png",
        mime_type="image/png",
        file_size=len(_png_bytes()),
        file_path="vis001/pic.png",
        file_hash=hashlib.sha256(_png_bytes()).hexdigest(),
    )
    # Materialise the file at the resolved documents path.
    abs_path = FileMapperService.get_documents_path("vis001", "pic.png")
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(_png_bytes())

    ability = VisionAbility(mp=_make_user_mp())
    result = ability.run({"image": doc_id, "query": "what is this"})

    assert result.status == "error"
    assert result.body  # non-empty: the failure is visible, not swallowed


def test_vision_run_unknown_doc_id_is_error(db):
    from abilities.vision import VisionAbility

    ability = VisionAbility(mp=_make_user_mp())
    result = ability.run({"image": "deadbeef", "query": "x"})

    assert result.status == "error"
    assert "deadbeef" in result.body


def test_rich_index_prompt_mentions_searchable():
    from abilities.vision import RICH_INDEX_PROMPT

    assert "searchable" in RICH_INDEX_PROMPT


# ===========================================================================
# Migrated from test_ability_vision_tool_result.py (TKT-975)
# Ability-specific business-logic tests for the ToolResult contract (TKT-915).
# ===========================================================================

import hashlib  # noqa: E402 — appended to existing file

from abilities._dispatcher import ToolDispatcher  # noqa: E402
from configs.channels import UserConfig  # noqa: E402
from services.document_service import DocumentService  # noqa: E402
from services.file_mapper_service import FileMapperService  # noqa: E402
from tests._tool_result_harness import MP, body, head, seed_transcript  # noqa: E402


@pytest.fixture
def _vision_chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor. Vision seeds ``allow`` on chat in the db template, so the real gate
    passes through to the production run()."""
    return MP(seed_transcript(db, content="what is in this image"), UserConfig({}))


def _vision_head(rendered: str) -> str:
    return head(rendered, "vision")


def _vision_body(rendered: str) -> str:
    return body(rendered, "vision")


def _real_image_doc_for_vision(db, rel_dir: str = "vis_tr001") -> str:
    """Create a real document row AND materialise a real PNG at the resolved
    documents path so the ability resolves a genuine file on disk."""
    png = _png_bytes()
    doc_svc = DocumentService(get_shared_db_service())
    doc_id = doc_svc.create_document(
        original_name="pic.png",
        mime_type="image/png",
        file_size=len(png),
        file_path=f"{rel_dir}/pic.png",
        file_hash=hashlib.sha256(png).hexdigest(),
    )
    abs_path = FileMapperService.get_documents_path(rel_dir, "pic.png")
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(png)
    return doc_id


@pytest.mark.unit
def test_whitespace_only_image_is_missing_params(db, _vision_chat_mp):
    """A whitespace-only ``image`` slips the truthiness pre-gate and reaches run(),
    which rejects the stripped-empty doc id as ``code=missing-params``."""
    out = ToolDispatcher(_vision_chat_mp).dispatch(
        "vision", {"image": "   ", "query": "x", "act_summary": "x"}
    )

    h = _vision_head(out)
    assert "status=error" in h
    assert "code=missing-params" in h
    assert "code=error]" not in out
    assert "image" in out


@pytest.mark.unit
def test_doc_with_no_file_path_is_no_file_on_disk(db, _vision_chat_mp):
    """A real document row whose ``file_path`` is empty (no stored file) →
    ``code=no-file-on-disk`` with a recovery hint."""
    doc_id = "facefeed2"
    db.execute(
        "INSERT INTO documents (id, original_name, mime_type, file_size_bytes, "
        "file_path, file_hash) VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, "pic.png", "image/png", len(_png_bytes()), "", "abc"),
    )
    db.commit()

    out = ToolDispatcher(_vision_chat_mp).dispatch(
        "vision", {"image": doc_id, "query": "x", "act_summary": "x"}
    )

    h = _vision_head(out)
    assert "status=error" in h
    assert "code=no-file-on-disk" in h
    assert "code=error]" not in out
    assert doc_id in _vision_body(out)
    assert any(ln.startswith("hint:") for ln in out.splitlines())


@pytest.mark.unit
def test_ocr_fallback_success_is_degraded(db, _vision_chat_mp):
    """No vision provider configured → the OCR fallback returns ``ok`` BUT with
    ``degraded=true`` in the head, and the no-vision-provider note still rides the
    body verbatim."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)
    assert ProviderDbService(get_shared_db_service()).get_vision_provider() is None

    doc_id = _real_image_doc_for_vision(db, rel_dir="vis_tr002")

    out = ToolDispatcher(_vision_chat_mp).dispatch(
        "vision", {"image": doc_id, "query": "what is this", "act_summary": "x"}
    )

    h = _vision_head(out)
    assert "status=success" in h
    assert "degraded=true" in h
    assert "vision provider" in _vision_body(out).lower()
