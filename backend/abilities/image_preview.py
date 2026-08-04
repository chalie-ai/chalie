# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ImagePreviewAbility — show one image to the user as a rich card.

Takes a local path or an http(s) URL, proves the bytes really are an image, and
lands it under the documents root so the existing ``/api/files/preview/`` route
serves it. Nothing here hotlinks: a remote image is downloaded once, so the card
survives the source 404ing and no third-party request fires each time the
conversation is re-rendered.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import ClassVar
from urllib.parse import unquote, urlparse

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.image_preview_params_bag import ImagePreviewParamsBag
from contracts.params.param_bag import ParamBag
from exceptions import DownloadTooLarge
from services.file_mapper_service import FileMapperService
from tools.image_preview.ingest import land_image, sniff_image

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 32 * 1024 * 1024  # per-image download cap, as image_search
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
        remote = source.lower().startswith(("http://", "https://"))

        tmp_path = ""
        try:
            if remote:
                # Local import to avoid import cycles (precedent: image_search.py).
                from services import tmp_storage  # noqa: PLC0415

                tmp_path = tmp_storage.new_tmp_path(f"imgpreview_{uuid.uuid4().hex[:8]}")
                failure = self._download(source, tmp_path)
                if failure is not None:
                    return failure
                path, name = tmp_path, self._url_name(source)
            else:
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

            relpath = self._land(path, name, remote=remote)
            subtitle, truncated = self._trim_subtitle(params.subtitle)

            body = f"Showed '{name}' to the user."
            if subtitle:
                body += f" Caption: {subtitle}"
            return ToolResult.ok(
                body,
                rich={
                    "url": f"/api/files/preview/{relpath}",
                    "subtitle": subtitle,
                    "alt": name,
                },
                subtitle_truncated=truncated,
            )
        finally:
            # Never let temp cleanup mask a propagating failure; the tmp-sweep
            # worker can delete it first, so tolerate the already-gone race.
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _download(url: str, dest: str) -> ToolResult | None:
        """Fetch ``url`` into ``dest``; return an error result, or None on success."""
        # Local import to avoid import cycles (precedent: image_search.py).
        from services.web_fetch import stream_to_file  # noqa: PLC0415

        try:
            stream_to_file(url, dest, max_bytes=_MAX_IMAGE_BYTES)
        except DownloadTooLarge:
            return ToolResult.err(
                "That image is larger than the 32 MB limit.",
                code="image-too-large",
                hint="pick a smaller image",
            )
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            logger.warning("[IMAGE_PREVIEW] download failed for %r: %s", url, exc)
            return ToolResult.err(
                f"Could not download that image; {exc}",
                code="image-fetch-failed",
                hint="check the URL points directly at an image, or try another",
            )
        return None

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
    def _land(path: str, name: str, *, remote: bool) -> str:
        """Return the documents-root-relative path the preview route serves.

        A local file already inside the documents root is served IN PLACE —
        copying it would duplicate the user's own file to no purpose. Everything
        else (downloads, files elsewhere on disk) is landed under previews/.
        """
        if not remote and FileMapperService.validate_document_path(path):
            root = str(FileMapperService.get_documents_path())
            return os.path.relpath(path, root).replace(os.sep, "/")
        return land_image(path, name)

    @staticmethod
    def _trim_subtitle(raw: str | None) -> tuple[str, bool]:
        """Clamp the caption to eight words, returning (caption, was_truncated).

        The one sanctioned truncation on this path: the caption is decoration
        sized to the card, and an over-long one breaks the layout. It is cut and
        MARKED (``subtitle_truncated`` in meta), never silently dropped.
        """
        words = (raw or "").split()
        if len(words) <= _SUBTITLE_WORD_LIMIT:
            return " ".join(words), False
        return " ".join(words[:_SUBTITLE_WORD_LIMIT]) + "…", True
