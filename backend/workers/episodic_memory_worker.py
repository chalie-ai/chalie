import json
import logging
import re
from typing import cast


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


def _safe_json_load(text: str) -> dict[str, object] | None:
    cleaned = _extract_json(text)
    try:
        return cast(dict[str, object], json.loads(cleaned))
    except json.JSONDecodeError:
        logging.error("[EPISODIC] Failed to parse JSON from LLM output")
        logging.debug(f"[EPISODIC] Raw output: {cleaned[:500]}")
        return None


