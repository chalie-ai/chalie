"""Bug fix: ``_seed_turn_zero`` dispatched the ``thinking`` pass BEFORE the
attachment upload block, so the single-pass thinking dispatch captured a parent
act-trail without the uploaded image — reasoning in a vacuum about what it could
not see and fetch (tools disabled). Fix: fire thinking after the upload barrier so
the trail row is present.

Prod hot path: ``MessageProcessor._seed_turn_zero`` on a real ``UserConfig``
channel, a real PNG attachment, the real ``document.upload`` ingest (deterministic
OCR fork with no vision provider), and the real ``thinking`` dispatch through the
real dispatcher. The ONLY stand-in is ``Providers._resolve`` — captured by
``_RecordingProvider`` below. Zero internal mocks.
"""

import io
import os
import sqlite3
from typing import cast

import pytest
from unittest.mock import patch

from configs.channels import UserConfig
from services.database_service import get_shared_db_service
from services.provider_api import ProviderApiRequest, ProviderApiResponse, ThinkingLevel
from services.message_processor import MessageProcessor
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path
from services.transcript_service import Transcript

pytestmark = pytest.mark.unit

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


class _RecordingProvider:
    """Stand-in for the resolved LLM provider — captures every ``send(dto)`` and
    returns a one-shot ``NOTHING`` response, with token estimate of 1 so pre-flight
    over-cap always passes."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self) -> None:
        self.sends: list[dict[str, object]] = []

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        return 1

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        self.sends.append({
            "system": dto.system,
            "messages": dto.messages,
            "tools": dto.tools,
            "thinking_mode": dto.thinking_mode,
        })
        return ProviderApiResponse(text="NOTHING", model="recorder", tool_calls=None)


def _png_with_text(text: str) -> bytes:
    """A small high-contrast PNG rendering *text*."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.ImageFont import FreeTypeFont, ImageFont as CoreFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    loaded_font: FreeTypeFont | CoreFont | None = None
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "DejaVuSans.ttf"):
        try:
            loaded_font = ImageFont.truetype(candidate, 72)
            break
        except Exception:
            continue
    if loaded_font is None:
        loaded_font = ImageFont.load_default()
    draw.text((20, 40), text, fill="black", font=loaded_font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_attachment(label: str) -> str:
    """Write a PNG under the Chalie temp prefix (so ``_read_attachment``'s
    realpath guard passes) and return its path."""
    path = new_tmp_path(f"seed_thinking_{label}.png")
    with open(path, "wb") as fh:
        fh.write(_png_with_text(label.upper()))
    return path


def _build_parent(attachments: list[str]) -> MessageProcessor:
    """A real UserConfig MessageProcessor in the exact state ``_seed_turn_zero``
    fires from: turn boundary allocated, input row written ON that turn,
    ``active_tools`` seeded, attachments on metadata, and the thinking gate already
    resolved to 'high'."""
    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, "What is in this image?", {"attachments": attachments})
    parent.config = UserConfig()
    # Mirror production _setup() exactly: write_input_row opens the next turn for
    # the channel ATOMICALLY inside the INSERT; read the allocated turn_id back by
    # row id. That (channel, turn_id) boundary is what the upload's act-trail row
    # and the thinking pass's _render_act_trail snapshot both key off — without it
    # the snapshot is blind to the upload.
    parent.uid = Transcript.write_input_row("user", "user", "What is in this image?")
    parent.turn_id = Transcript.turn_id_of_row(parent.uid)
    parent.active_tools = list(parent.config.always_available or [])
    # The gate (user channel only) would set this before _seed_turn_zero fires.
    parent.thinking_level = "high"
    return parent


def test_turn0_thinking_pass_sees_uploaded_image_in_its_snapshot(db: sqlite3.Connection) -> None:
    """The thinking pass fires AFTER the upload, so the parent body it snapshots
    already carries the upload's act-trail row.

    Pre-fix (thinking dispatched before the upload block) the snapshot has no
    upload row and this FAILS; post-fix the upload row is present and it PASSES.
    """
    # No vision provider -> the deterministic OCR fork; the upload never touches
    # the LLM boundary, so the only recorded send is the thinking pass.
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    attachment = _write_attachment("invoice")
    name = os.path.basename(attachment)

    recorder = _RecordingProvider()
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        # The REAL production method: memory seed -> uploads (barrier) -> thinking.
        _build_parent([attachment])._seed_turn_zero()

    # Exactly one high-deliberation send reached the boundary — the thinking pass.
    # thinking_mode is now a ThinkingLevel enum, not a plain string.
    high_sends = [s for s in recorder.sends if s["thinking_mode"] == ThinkingLevel.HIGH]
    assert len(high_sends) == 1, (
        f"expected exactly one high-deliberation send, got {len(high_sends)} "
        f"(total sends={len(recorder.sends)})"
    )

    content = cast(str, cast(dict[str, object], cast(list[object], high_sends[0]["messages"])[0])["content"])
    # The deliberation snapshot must contain the upload's act-trail row — proof the
    # thinking pass fired AFTER the vision-bearing upload, not before. The upload's
    # structured success body carries the doc name + status=ready ().
    assert "[document(status=success" in content and name in content, (
        "thinking pass snapshot is missing the uploaded document — it fired before "
        "the upload barrier, so the deliberation cannot reason about the image"
    )
