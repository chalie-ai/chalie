"""
Episodic Memory Worker — Utility functions.
"""

import json
import logging
import re


def _extract_json(text: str) -> str:
    """
    Extract JSON from text, handling markdown code fences.

    Strips markdown fences (```json ... ``` or ``` ... ```), handles commentary
    before/after JSON, and multiple fenced blocks (takes first).
    """
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


def _safe_json_load(text: str) -> dict | None:
    """
    Safely load JSON with graceful fallback for parse errors.

    Extracts JSON from markdown fences and attempts parsing.
    On failure, logs error and returns None instead of crashing.
    """
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logging.error("[EPISODIC] Failed to parse JSON from LLM output")
        logging.debug(f"[EPISODIC] Raw output: {cleaned[:500]}")
        return None


