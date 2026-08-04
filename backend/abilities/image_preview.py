# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ImagePreviewAbility — show one image to the user as a rich card.

Takes a local path or a direct http(s) URL. A URL is handed straight to the
browser as an ``<img src>``: it is already fetchable from where the picture is
being looked at, so copying it through this process would buy nothing and cost a
download. A local file is proved to be an image by its BYTES and then served by
the existing attachment route — in place when it already lives under the
documents root, otherwise copied in by the same ``FileParserService.place`` every
chat attachment goes through, so the card survives the source moving.
"""

from __future__ import annotations

import os
from typing import ClassVar
from urllib.parse import unquote, urlparse

from abilities._ability import Ability
from abilities._result import ToolResult, truncate
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.image_preview_params_bag import ImagePreviewParamsBag
from contracts.params.param_bag import ParamBag
from services.file_mapper_service import FileMapperService
from services.file_parser_service import FileParserService
from tools.image_preview.sniff import sniff_image

_SNIFF_BYTES = 512  # covers every signature, including an SVG prolog
_SUBTITLE_WORD_LIMIT = 8


class ImagePreviewAbility(Ability[ImagePreviewParamsBag]):
    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty file_path as code=missing-params before run() is reached.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.file_path,)}
    NAME: ClassVar[str] = "image_preview"
    # Media, not Conversation History — that heading is where the transcript
    # retrieval tools live, and a model hunting for "how do I show a picture"
    # would never look under it.
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.MEDIA

    PARAMS: ClassVar[type[ParamBag] | None] = ImagePreviewParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = (
        "show image",
        "display image",
        "preview image",
        "show picture",
    )

    def get_summary(self) -> str:
        return (
            "Use this tool to show an image to the user. It takes a local file path "
            "or an image URL and renders the picture directly in the conversation, "
            "with an optional short caption. This is the only way the user actually "
            "SEES an image — describing one in text does not show it."
        )

    def get_examples(self) -> list[str]:
        return [
            "show me that photo",
            "display the chart you just made",
            "let me see the image you found",
            "show me what the northern lights look like",
            "put that picture in the chat",
            "show me the diagram",
        ]

    def get_search_tooltip(self) -> str:
        return "show an image to the user"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.file_path: {
                "type": "string",
                "description": (
                    "The image to show: either an absolute path to a file on this "
                    "machine, or a direct http(s) URL to an image."
                ),
            },
            Keys.subtitle: {
                "type": "string",
                "description": (
                    "Optional short caption shown under the image. "
                    "At most 8 words — anything longer is trimmed."
                ),
            },
        },
        "required": [Keys.file_path],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: ImagePreviewParamsBag) -> ToolResult:
        source = params.file_path.strip()

        # A remote image is rendered from its own URL — the browser fetches it,
        # nothing is downloaded, sniffed or stored here. An <img> renders images
        # and nothing else, so there is no byte gate to run on bytes we never hold.
        if source.lower().startswith(("http://", "https://")):
            return self._card(source, self._url_name(source), params.subtitle)

        resolved = self._resolve_local(source)
        if isinstance(resolved, ToolResult):
            return resolved
        path, name = resolved

        # THE GATE. image_preview is INTERNAL — it bypasses the policy layer on
        # every channel — so this content check is the only thing stopping a
        # non-image server file being copied into the served store. Bytes only.
        with open(path, "rb") as handle:
            head = handle.read(_SNIFF_BYTES)
        if sniff_image(head) is None:
            return ToolResult.err(
                "That file is not an image.",
                code="not-an-image",
                hint="pass a path or URL to a real image (png, jpg, gif, svg, webp)",
            )

        return self._card(f"/api/files/preview/{self._land(path)}", name, params.subtitle)

    @staticmethod
    def _card(url: str, name: str, subtitle: str | None) -> ToolResult:
        """Build the one success shape: the rich card, plus what the model is told.

        The caption is clamped to eight words and MARKED via
        ``subtitle_truncated`` — decoration sized to the card, cut loudly rather
        than dropped.
        """
        caption, truncated = truncate(
            subtitle or "", _SUBTITLE_WORD_LIMIT, words=True, suffix="…"
        )
        body = f"Showed '{name}' to the user."
        if caption:
            body += f" Caption: {caption}"
        return ToolResult.ok(
            body,
            rich={"url": url, "subtitle": caption, "alt": name},
            subtitle_truncated=truncated,
        )

    @staticmethod
    def _resolve_local(source: str) -> tuple[str, str] | ToolResult:
        """Resolve a local path to (realpath, display name), or an error result."""
        path = os.path.realpath(os.path.expanduser(source))
        if not os.path.isfile(path):
            return ToolResult.err(
                f"There is no file at '{source}'.",
                code="image-not-found",
                hint="pass the full path to an existing image, or an http(s) URL",
            )
        return path, os.path.basename(path)

    @staticmethod
    def _url_name(url: str) -> str:
        """Best-effort filename for a URL, for the card's alt text."""
        return os.path.basename(unquote(urlparse(url).path)) or "image"

    @staticmethod
    def _land(path: str) -> str:
        """Return the documents-root-relative path the preview route serves.

        The route only ever serves what is under the documents root, so a file
        from elsewhere on disk is copied in by the SAME primitive chat
        attachments use — no second notion of a servable file, and no bespoke
        copier to keep in step with it. A file already inside the root is served
        in place: copying the user's own file there would duplicate it to no end.
        """
        if not FileMapperService.validate_document_path(path):
            path = FileParserService().place(path)
        root = str(FileMapperService.get_documents_path())
        return os.path.relpath(path, root).replace(os.sep, "/")
