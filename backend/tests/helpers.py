"""Test data factories — produce realistic row tuples matching actual DB column orders.

All factories return tuples unless noted otherwise. Override any field via keyword argument.
"""

from __future__ import annotations

import base64
import io
import sqlite3
from datetime import datetime, timezone
from typing import cast

from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


# ─── ProcessorConfig test stub ───────────────────────────────────────
# A minimal concrete ProcessorConfig for tests that need a config instance
# (policy-channel derivation, tool-finder embeddings). Prompt assembly lives in
# PromptService, so the stub carries no prompt-builder surface.

class StubProcessorConfig(ProcessorConfig):
    def __init__(
        self,
        *,
        channel: str,
        role: str,
        policy_channel: PolicyChannel,
        always_available: list[str],
        skip_transcript: bool,
        skip_input_row: bool,
        suppress_history: bool,
        memory_seed: bool,
    ) -> None:
        super().__init__(
            channel=channel,
            role=role,
            policy_channel=policy_channel,
            always_available=always_available,
            skip_transcript=skip_transcript,
            skip_input_row=skip_input_row,
            suppress_history=suppress_history,
            memory_seed=memory_seed,
        )


def make_stub_config(
    *,
    always_available: list[str] | None = None,
    channel: str = "user",
    role: str = "user",
    policy_channel: PolicyChannel | None = None,
) -> StubProcessorConfig:
    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=policy_channel or PolicyChannel.CHAT,
        always_available=list(always_available or []),
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=False,
        memory_seed=False,
    )


# ─── scheduled_items ─────────────────────────────────────────────────
# Column order matches schema.sql (the real 5-field crontab engine):
#   message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow,
#   enabled, channel, created_by_session, created_at
# ``id`` is INTEGER PRIMARY KEY AUTOINCREMENT — never supplied on insert, always
# read back via cursor.lastrowid (it also doubles as the schedule's turn_id on
# the 'schedule' channel). All five cron_* columns are TEXT NOT NULL DEFAULT '*'
# crontab expressions (services.cron_schedule.validate_cron / matches); '*' =
# every. start_at is a UTC activation floor.
#
# This is the SINGLE shared seed path for a real scheduled_items row — every
# test that needs a pre-existing row (rather than driving one through the
# ability/REST create path) calls this, so the INSERT's column list can never
# drift out of step with schema.sql in more than one place (the exact bug
# class a hand-rolled per-file INSERT reintroduces).

def insert_scheduled_item(
    db: sqlite3.Connection,
    message: str = "Test reminder",
    start_at: str | None = None,
    cron_minute: str = "*",
    cron_hour: str = "*",
    cron_dom: str = "*",
    cron_month: str = "*",
    cron_dow: str = "*",
    enabled: int = 1,
    channel: str | None = "general",
    created_by_session: str | None = None,
    created_at: str | None = None,
) -> int:
    """Insert a real ``scheduled_items`` row and return its auto-assigned id."""
    start_at = start_at or datetime.now(timezone.utc).isoformat()
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    cur = db.execute(
        """
        INSERT INTO scheduled_items
          (message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow,
           enabled, channel, created_by_session, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow,
            enabled, channel, created_by_session, created_at,
        ),
    )
    db.commit()
    return cast(int, cur.lastrowid)


# ─── images ──────────────────────────────────────────────────────────
# The two poles of the ImageDescription contract: an image RapidOCR can read
# and one it cannot. Real, decodable bytes — the OCR fork runs for real.

def blank_png_bytes() -> bytes:
    """A 1x1 white PNG. RapidOCR returns '' for it (error=None, no text)."""
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        b"+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def _ocrable_bytes(word: str, fmt: str) -> bytes:
    """``word`` in large black type on white, encoded as ``fmt``."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font: object = None
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, 72)
            break
        except OSError:
            continue
    draw.text((20, 40), word, fill="black", font=cast("ImageFont.FreeTypeFont | None", font) or ImageFont.load_default())
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def ocrable_png_bytes(word: str = "INVOICE") -> bytes:
    """A PNG bearing ``word``. The default word was empirically confirmed readable
    by RapidOCR in this environment via ``image_context_service.analyze``
    (ocr_text == 'INVOICE')."""
    return _ocrable_bytes(word, "PNG")


def ocrable_unmimed_bytes(word: str = "INVOICE") -> bytes:
    """The same readable image in DDS — a format Pillow DECODES but registers no
    mime for (``Image.MIME.get('DDS') is None``; 23 such formats exist here).

    Pins the rung-gating contract: no mime means the provider rung cannot be fed,
    NOT that the image is unreadable. OCR reads these bytes fine, so the ladder
    must fall through to it instead of failing the whole describe."""
    return _ocrable_bytes(word, "DDS")


def force_vision_provider(db: sqlite3.Connection, host: str = "http://127.0.0.1:1") -> int:
    """Create a REAL provider row via the production factory and make it the
    resolvable vision provider. ``host`` defaults to a dead port, so the provider
    is configured-but-unreachable — the state that exercises the failure fork.

    create_provider runs a live vision probe against ``host``; when that host is
    unreachable it records supports_vision=0, so the column is forced to 1 —
    exactly the state a successful probe would have written.
    """
    from services.provider_db_service import ProviderDbService

    svc = ProviderDbService()
    provider = cast("dict[str, object]", svc.create_provider({
        "name": "probe-vision",
        "platform": "ollama",
        "model": "llava",
        "host": host,
        "api_key": "",
    }))
    pid = cast(int, provider["id"])
    db.execute("UPDATE providers SET supports_vision = 1 WHERE id = ?", (pid,))
    db.commit()
    svc.set_vision_provider(pid)
    return pid
