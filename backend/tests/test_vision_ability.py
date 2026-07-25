"""Feature tests for VisionAbility + ImageDescription.

Drives the real ``ImageDescription`` core and the real VisionAbility.run() entry
point against the real test DB, with real provider rows created by the
production ProviderDbService factory and real RapidOCR. No mocks.

ImageDescription is a two-rung ladder, and a description is MANDATORY:
  * vision provider unset OR failed -> degrade to OCR (logged server-side only).
  * OCR also fails or reads nothing -> RAISE.

The degradation is deliberately invisible to the caller: it gets a real
description or an exception — never an empty string or a half-success it would
have to interpret. VisionAbility.run() converts the raise into status='error'
with the cause visible.

``image`` is an absolute filepath on disk, resolved directly; a path with no
file at it is ``code=not-found``.
"""

import logging
import sqlite3
from pathlib import Path

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from services.image_description import ImageDescription
from services.provider_db_service import ProviderDbService
from tests.helpers import (
    blank_png_bytes,
    force_vision_provider,
    ocrable_png_bytes,
    ocrable_unmimed_bytes,
)

pytestmark = pytest.mark.unit


def _make_user_mp() -> MessageProcessor:

    return MessageProcessor(UserConfig(), raw_input="x")


def test_image_description_no_provider_degrades_to_ocr(
    db: sqlite3.Connection, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No vision provider configured: OCR carries the description. The degradation
    is logged server-side and NOT smuggled into the returned text."""
    ProviderDbService().set_vision_provider(None)
    assert ProviderDbService().get_vision_provider() is None

    img = tmp_path / "invoice.png"
    img.write_bytes(ocrable_png_bytes())

    with caplog.at_level(logging.WARNING, logger="services.image_description"):
        out = ImageDescription(str(img), "what is this").get_value()

    assert "INVOICE" in out.upper()
    assert "description loaded via OCR" in caplog.text


def test_image_description_provider_failure_degrades_to_ocr(
    db: sqlite3.Connection, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Configured-but-unreachable vision provider + an image with words: the OCR
    rung rescues the description. The provider failure is logged, never surfaced
    — the caller gets a real description, not an error it must interpret."""
    force_vision_provider(db)

    img = tmp_path / "invoice.png"
    img.write_bytes(ocrable_png_bytes())

    with caplog.at_level(logging.WARNING, logger="services.image_description"):
        out = ImageDescription(str(img), "what is this").get_value()

    assert "INVOICE" in out.upper()
    assert "description loaded via OCR" in caplog.text


def test_image_description_unmimed_format_still_reaches_ocr(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A readable image in a format Pillow decodes but has no mime for (DDS) must
    still be described. The media-type sniff feeds the PROVIDER rung only — gating
    the whole ladder on it stranded 23 decodable formats on an unreadable-file
    error while OCR could read them perfectly."""
    force_vision_provider(db)

    img = tmp_path / "texture.dds"
    img.write_bytes(ocrable_unmimed_bytes())

    out = ImageDescription(str(img), "what is this").get_value()

    assert "INVOICE" in out.upper()


def test_image_description_raises_when_no_provider_and_ocr_reads_nothing(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Both rungs exhausted (no provider, textless image -> empty OCR). A
    description is mandatory, so this RAISES rather than returning ''."""
    ProviderDbService().set_vision_provider(None)

    img = tmp_path / "blank.png"
    img.write_bytes(blank_png_bytes())

    with pytest.raises(RuntimeError, match="empty description"):
        ImageDescription(str(img), "what is this").get_value()


def test_image_description_raises_when_provider_fails_and_ocr_reads_nothing(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Both rungs exhausted the other way: the provider is configured but dead AND
    the image is textless. Still a raise — never a silent empty description."""
    force_vision_provider(db)

    img = tmp_path / "blank.png"
    img.write_bytes(blank_png_bytes())

    with pytest.raises(RuntimeError, match="empty description"):
        ImageDescription(str(img), "what is this").get_value()


def test_vision_run_provider_error_returns_visible_error(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    """The styled-error surface: VisionAbility.run wraps ImageDescription's raise in
    its one allowed except and returns status='error' with the cause visible
    — never a false success / empty string."""
    from abilities.vision import VisionAbility
    from contracts.params.vision_params_bag import VisionParamsBag

    force_vision_provider(db)

    # A real on-disk image file the ability resolves.
    img = tmp_path / "pic.png"
    img.write_bytes(blank_png_bytes())

    ability = VisionAbility(mp=_make_user_mp())
    result = ability.run(
        built(VisionParamsBag.from_params({"image": str(img), "instructions": "what is this"}))
    )

    assert result.status == "error"
    assert result.body  # non-empty: the failure is visible, not swallowed


def test_vision_run_unknown_filepath_is_error(db: sqlite3.Connection) -> None:
    from abilities.vision import VisionAbility
    from contracts.params.vision_params_bag import VisionParamsBag

    ability = VisionAbility(mp=_make_user_mp())
    result = ability.run(built(VisionParamsBag.from_params({"image": "deadbeef.png", "instructions": "x"})))

    assert result.status == "error"
    assert "deadbeef.png" in result.body


# ===========================================================================
# Migrated from test_ability_vision_tool_result.py ()
# Ability-specific business-logic tests for the ToolResult contract ().
# ===========================================================================

from tests._tool_result_harness import body, built, head, seed_transcript  # noqa: E402


@pytest.fixture
def _vision_chat_mp(db: sqlite3.Connection) -> MessageProcessor:
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor (mp.uid — dispatch's ``ToolCallService._transcript_id()`` falls back to
    it when ``current_transcript_id`` is unset). Vision seeds ``allow`` on chat in
    the db template, so the real gate passes through to the production run()."""
    mp = MessageProcessor(UserConfig({}), raw_input="what is in this image")
    mp.active_tools = list(mp.config.always_available or [])
    mp.uid = seed_transcript(db, content="what is in this image")
    return mp


def _vision_head(rendered: str) -> str:
    return head(rendered, "vision")


def _vision_body(rendered: str) -> str:
    return body(rendered, "vision")


def _real_image_path(tmp_path: Path, png: bytes | None = None, name: str = "pic.png") -> str:
    """Materialise a real PNG on disk and return its absolute path — the vision
    contract's ``image`` param. ``png`` selects which rung of the describe
    ladder the image lands on: textless (default) exhausts both, and
    ocrable_png_bytes() lets OCR carry it."""
    png = blank_png_bytes() if png is None else png
    path = tmp_path / name
    path.write_bytes(png)
    return str(path)


@pytest.mark.unit
def test_nonexistent_filepath_is_not_found(
    db: sqlite3.Connection, _vision_chat_mp: MessageProcessor, tmp_path: Path
) -> None:
    """A filepath with no file at it on disk → ``code=not-found`` with a
    recovery hint. The doc_id/no-file-on-disk fork is gone: ``image`` is now
    resolved directly as a filepath, so a missing file IS the not-found case."""
    missing_path = str(tmp_path / "does-not-exist.png")

    out = _vision_chat_mp.dispatch_service.dispatch(
        "vision", {"image": missing_path, "instructions": "x", "act_summary": "x"}
    )

    h = _vision_head(out)
    assert "status=error" in h
    assert "code=not-found" in h
    assert "code=error]" not in out
    assert missing_path in _vision_body(out)
    assert any(ln.startswith("hint:") for ln in out.splitlines())


@pytest.mark.unit
def test_ocr_fallback_success_is_an_ordinary_success(
    db: sqlite3.Connection, _vision_chat_mp: MessageProcessor, tmp_path: Path
) -> None:
    """No vision provider → OCR carries the description → an ORDINARY success at
    the dispatch surface. The degradation is a server-side log line only: no
    ``degraded`` flag and no note in the body, so the LLM sees a description
    exactly like a provider-served one and never has to reason about the rung."""
    ProviderDbService().set_vision_provider(None)
    assert ProviderDbService().get_vision_provider() is None

    image_path = _real_image_path(tmp_path, ocrable_png_bytes())

    out = _vision_chat_mp.dispatch_service.dispatch(
        "vision", {"image": image_path, "instructions": "what is this", "act_summary": "x"}
    )

    h = _vision_head(out)
    assert "status=success" in h
    assert "degraded" not in h
    assert "INVOICE" in _vision_body(out).upper()
    assert "vision provider" not in _vision_body(out).lower()


@pytest.mark.unit
def test_textless_image_with_no_provider_is_a_visible_error(
    db: sqlite3.Connection, _vision_chat_mp: MessageProcessor, tmp_path: Path
) -> None:
    """Both rungs exhausted at the dispatch surface: no provider AND a textless
    image → ``status=error`` with a recovery hint, never a hollow success."""
    ProviderDbService().set_vision_provider(None)

    image_path = _real_image_path(tmp_path)

    out = _vision_chat_mp.dispatch_service.dispatch(
        "vision", {"image": image_path, "instructions": "what is this", "act_summary": "x"}
    )

    h = _vision_head(out)
    assert "status=error" in h
    assert "code=vision-failed" in h
    assert any(ln.startswith("hint:") for ln in out.splitlines())
