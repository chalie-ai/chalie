# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VisionAbility — read/see an image via the brain's Vision Provider.

describe_image() is the shared describe core: it forks ONCE on whether a vision
provider is configured — provider -> a single-shot MessageProcessor on the vision
provider (image attached via VisionConfig.get_image); none -> RapidOCR
(image_context_service) + a note. The fork is never switched. Two callers:
VisionAbility.run() (the user-facing tool) and text_extractor (upload-time
indexing).

The paired channel config (``VisionConfig``) lives in
``configs/channels/vision.py`` — delegate configs are ProcessorConfig subclasses
and belong with the other channels, not in the ability."""

from __future__ import annotations

import logging
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.channels.vision import VisionConfig

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

_NO_VISION_NOTE = (
    "NOTE: A vision provider is not available. The system can only read text from "
    "images. If the user wants the system to view the images, ask them to set up a "
    "vision provider in the brain interface."
)


def describe_image(image_path: str, mime_type: str, query: str, *, policy_channel) -> dict:
    """Describe an image. Forks ONCE on vision-provider-configured.
    Returns {"description": str, "vision_used": bool, "note": str|None}.
    Provider path RAISES on provider failure (never swallowed); the OCR fallback
    is ONLY for the not-configured path."""
    from services.database_service import get_shared_db_service  # noqa: PLC0415
    from services.provider_db_service import ProviderDbService  # noqa: PLC0415

    if ProviderDbService(get_shared_db_service()).get_vision_provider():
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        description = MessageProcessor.process(
            query,
            VisionConfig(policy_channel),
            metadata={"image_path": image_path, "mime_type": mime_type},
        )
        return {"description": description, "vision_used": True, "note": None}

    from services import image_context_service  # noqa: PLC0415

    with open(image_path, "rb") as fh:
        ocr = image_context_service.analyze(fh.read(), mime_type)
    if ocr.get("error"):
        logger.warning("[VISION] OCR extraction error: %s", ocr["error"])
    return {
        "description": (ocr.get("ocr_text") or "").strip(),
        "vision_used": False,
        "note": _NO_VISION_NOTE,
    }


class VisionAbility(Ability):
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
    # or empty image/query as code=missing-params before run() is reached
    # (precedent: save_graph.py, save_pattern.py, file_permissions.py). The
    # pre-gate is truthiness-based, so whitespace-only residue still reaches run().
    ACTION_REQUIRED: ClassVar[dict] = {"": ("image", "query")}

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": (
                    "The 8-character document id (doc_id). If this is not available "
                    "in context, use the `document.search` tool to look it up."
                ),
            },
            "query": {
                "type": "string",
                "description": "What to find out about the image, in natural language.",
            },
        },
        "required": ["image", "query"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        # The dispatcher pre-gate is truthiness-based, so a non-empty but
        # whitespace-only image/query slips past it and must be rejected here
        # (precedent: save_graph.py, file_permissions.py).
        doc_id = (params.get("image") or "").strip()
        query = (params.get("query") or "").strip()
        if not doc_id or not query:
            missing = ", ".join(
                name for name, val in (("image", doc_id), ("query", query)) if not val
            )
            return ToolResult.err(
                f"Missing required parameter(s): {missing}.",
                code="missing-params",
                valid=("image", "query"),
            )

        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.document_service import DocumentService  # noqa: PLC0415
        from services.file_mapper_service import FileMapperService  # noqa: PLC0415

        doc = DocumentService(get_shared_db_service()).get_document(doc_id)
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

        abs_path = str(FileMapperService.get_documents_path(doc["file_path"]))
        mime_type = doc.get("mime_type") or "image/png"
        try:
            out = describe_image(
                abs_path, mime_type, query, policy_channel=self.mp.config.policy_channel
            )
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            logger.exception("[VISION] describe failed")
            return ToolResult.err(
                f"Vision is currently experiencing issues; {exc}",
                code="vision-failed",
                hint="check the vision provider in the brain interface or try again",
            )

        body = out["description"]
        if out["note"]:
            body = f"{body}\n\n{out['note']}" if body else out["note"]
        # The OCR fallback (no vision provider configured) is a downgraded read —
        # mark it degraded so the model can tell it from a real vision read
        # (loudness precedent: find_skills FTS fallback, programming_docs_search).
        if not out["vision_used"]:
            return ToolResult.ok(body, degraded=True)
        return ToolResult.ok(body)
