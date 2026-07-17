# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ImageDescription — the single place an image becomes text.

A two-rung ladder: the vision provider first, then RapidOCR via
``image_context_service.analyze`` when the provider is unset or produced an
empty result. A description is mandatory — exhausting both rungs raises
``RuntimeError`` rather than returning an empty string.

Always runs on the subconscious policy channel: no human is watching a vision
turn, so we never route through a user-visible channel.
"""

from __future__ import annotations

import logging
from typing import cast

from configs.enums.policy_channel import PolicyChannel

logger = logging.getLogger(__name__)

RICH_INDEX_PROMPT = (
    "Describe this image in great detail. Your description will be embedded and "
    "used to make the image searchable in a semantic database, so be thorough and "
    "concrete. Cover at least 20 distinct aspects: the overall layout and "
    "composition; every subject, person, animal, or object and what they are doing; "
    "dominant and accent colours; any title, heading, label, or visible text "
    "(transcribe it verbatim); the setting and background; style, mood, and "
    "lighting; and anything distinctive someone might later search for."
)


class ImageDescription:
    """Describe an image on disk via the vision provider (primary) or OCR (fallback)."""

    def __init__(self, file_path: str, prompt: str) -> None:
        self._file_path = file_path
        self._prompt = prompt

    def get_value(self) -> str:
        """Run the two-rung ladder and return the image description.

        Raises ``RuntimeError`` if both rungs produce nothing.
        """
        from services.provider_db_service import ProviderDbService  # noqa: PLC0415

        media_type = self._media_type()

        description: str | None = None
        if media_type and ProviderDbService().get_vision_provider():
            from controllers.message_processor import MessageProcessor  # noqa: PLC0415
            from configs.channels.vision import VisionConfig  # noqa: PLC0415

            description = MessageProcessor.process(
                VisionConfig(PolicyChannel.SUBCONSCIOUS),
                raw_input=self._prompt,
                metadata={"image_path": self._file_path, "mime_type": media_type},
            ).result()

        if not description or not description.strip():
            from services import image_context_service  # noqa: PLC0415

            logger.warning("[VISION] vision provider failed/not set, description loaded via OCR")
            with open(self._file_path, "rb") as fh:
                ocr = image_context_service.analyze(fh.read())
            if ocr.get("error"):
                raise RuntimeError(f"OCR failed: {ocr['error']}")
            description = (cast(str, ocr.get("ocr_text")) or "").strip()
            if not description:
                raise RuntimeError("OCR produced an empty description")
        return description

    def _media_type(self) -> str | None:
        """The media type Pillow sniffs from the file bytes, or None when it cannot
        name one — the bytes are not a decodable image, or Pillow decoded the format
        but registers no mime for it (DDS, QOI, WMF, CUR … — 23 of them).

        Advisory, never fatal: only the provider rung needs a media type to send, so
        None skips that rung and drops to OCR, which reads the bytes directly. This
        is not a swallowed error — a file neither rung can read still fails loudly,
        raised by OCR where the real reason is known.
        """
        from PIL import Image  # noqa: PLC0415

        Image.init()
        try:
            with Image.open(self._file_path) as img:
                fmt = img.format or ""
        except OSError:
            return None
        return Image.MIME.get(fmt)
