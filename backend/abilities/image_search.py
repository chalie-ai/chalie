# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ImageSearchAbility — find images on the web for a text query.

Uses the image_search engine (DDG images) to fetch the top image results for a
natural language query. Returns a formatted list with titles, URLs, and source
sites."""

from __future__ import annotations

import logging
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from tools.image_search import fetcher

logger = logging.getLogger(__name__)


class ImageSearchAbility(Ability):
    def get_name(self) -> str:
        return "image_search"

    def get_summary(self) -> str:
        return (
            "Use this tool to find real images on the web for a text query. "
            "It returns the top few results with their titles, page URLs, and source sites."
        )

    def get_examples(self) -> list[str]:
        return [
            "find a picture of a golden retriever",
            "show me images of the eiffel tower at night",
            "get photos of mid-century modern living rooms",
            "search for pictures of sushi restaurants in tokyo",
            "find images of solar panel installation",
            "show me pictures of the northern lights",
            "get photos of scandinavian furniture design",
        ]

    def get_search_tooltip(self) -> str:
        return "find images on the web"

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty query as code=missing-params before run() is reached
    # (precedent: save_graph.py, save_pattern.py, file_permissions.py). The
    # pre-gate is truthiness-based, so whitespace-only residue still reaches run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.query,)}

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "string",
                "description": "What to find images of, in natural language.",
            },
        },
        "required": [Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        # The dispatcher pre-gate is truthiness-based, so a non-empty but
        # whitespace-only query slips past it and must be rejected here
        # (precedent: save_graph.py, file_permissions.py).
        query = (cast(str, params.get(Keys.query)) or "").strip()
        if not query:
            return ToolResult.err(
                "Missing required parameter: query.",
                code="missing-params",
                valid=("query",),
            )

        try:
            results = fetcher.fetch(query, limit=5)
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            logger.exception("[IMAGE_SEARCH] engine failed")
            return ToolResult.err(
                f"Image search is currently unavailable; {exc}",
                code="image-search-failed",
                hint="try again in a moment",
            )

        if not results:
            return ToolResult.ok(f'No images found for "{query}".', count=0)

        lines: list[str] = [f"Found {len(results)} image(s) for \"{query}\":"]
        for idx, result in enumerate(results, start=1):
            title = (result.get("title") or "").strip() or "(untitled)"
            url = result.get("url", "")
            source = result.get("source", "")
            lines.append(f"{idx}. {title}")
            lines.append(f"   URL: {url}")
            lines.append(f"   Source: {source}")
            lines.append("")

        body = "\n".join(lines)
        return ToolResult.ok(body, count=len(results))
