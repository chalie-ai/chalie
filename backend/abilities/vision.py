# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VisionAbility — read/see an image via the brain's Vision Provider.

The describe ladder (vision provider primary, OCR fallback) lives in
``services/image_description.py`` as ``ImageDescription``; this file is the
user-facing tool that drives it through ``DocumentService`` + ``FileMapperService``.

The paired channel config (``VisionConfig``) lives in
``configs/channels/vision.py`` — delegate configs are ProcessorConfig subclasses
and belong with the other channels, not in the ability."""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from abilities._delegate import DelegateAbility
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.vision_params_bag import VisionParamsBag

logger = logging.getLogger(__name__)



class VisionAbility(DelegateAbility[VisionParamsBag]):
    def get_name(self) -> str:
        return "vision"

    def get_summary(self) -> str:
        return (
            "Use this tool to read and see the contents of an image. It gives you "
            "vision capability even if you don't natively support it."
        )

    def get_examples(self) -> list[str]:
        return [
            "what is in this image",
            "describe the photo the user attached",
            "read the text from this screenshot",
            "what does the chart in this image show",
            "is there a person in this picture",
            "what colour is the car in the image",
            "transcribe the handwriting in this scan",
            "what product is shown in this photo",
        ]

    def get_search_tooltip(self) -> str:
        return "read & see image contents"

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty image/instructions as code=missing-params before run() is reached
    # (precedent: save_graph.py, save_pattern.py). The pre-gate is
    # truthiness-based; the bag's from_params rejects the whitespace-only residue.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.image, Keys.instructions)}

    # The typed input contract: the dispatch seam builds the bag via
    # VisionParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = VisionParamsBag

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.image: {
                "type": "string",
                "description": (
                    "The 8-character document id (doc_id). If this is not available "
                    "in context, use the `document.search` tool to look it up."
                ),
            },
            Keys.instructions: {
                "type": "string",
                "description": "What to find out about the image, in natural language.",
            },
        },
        "required": [Keys.image, Keys.instructions],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: VisionParamsBag) -> ToolResult:
        doc_id = params.image
        instructions = params.instructions

        from services.document_service import DocumentService  # noqa: PLC0415
        from services.file_mapper_service import FileMapperService  # noqa: PLC0415

        doc = DocumentService().get_document(doc_id)
        if not doc:
            return ToolResult.err(
                f"No document found for id={doc_id}.",
                code="not-found",
                hint="use document.search to look up the document id",
            )
        if not doc.get("file_path"):
            return ToolResult.err(
                f"Document {doc_id} has no file on disk.",
                code="no-file-on-disk",
                hint="the document has no stored file; re-upload the image",
            )

        abs_path = str(FileMapperService.get_documents_path(cast(str, doc["file_path"])))
        try:
            from services.image_description import ImageDescription  # noqa: PLC0415

            description = ImageDescription(abs_path, instructions).get_value()
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            logger.exception("[VISION] describe failed")
            return ToolResult.err(
                f"Vision is currently experiencing issues; {exc}",
                code="vision-failed",
                hint="check the vision provider in the brain interface or try again",
            )

        return ToolResult.ok(description)
