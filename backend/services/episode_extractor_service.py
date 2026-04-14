# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Episode Extractor Service — produces structured episodes from transcript windows.

Takes a list of transcript entries and sends them to an LLM which handles:
- Boundary detection within the window (goal shifts, emotional register changes, new entities, causal breaks)
- Salience filtering (omits trivial/routine segments)
- Structured episode extraction with transcript ID references

The extractor is a pure function — it does NOT store episodes. The caller handles storage.
"""

import json
import logging
import re
from typing import Optional

from services.config_service import ConfigService
from services.llm_service import create_llm_service

logger = logging.getLogger(__name__)

_REQUIRED_EPISODE_FIELDS = {
    'intent', 'context', 'action', 'emotion', 'outcome', 'gist',
    'salience_factors', 'transcript_ids',
}


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


def _safe_json_load(text: str) -> Optional[list]:
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("[EXTRACTOR] Failed to parse JSON from LLM output")
        logger.debug(f"[EXTRACTOR] Raw output: {cleaned[:500]}")
        return None


class EpisodeExtractorService:
    def __init__(self):
        config = ConfigService.resolve_agent_config("episode-extraction")
        self._llm = create_llm_service(config)
        self._prompt_template = ConfigService.get_agent_prompt("episode-extraction")

    def extract(self, entries: list[dict], channel: str) -> list[dict]:
        """
        Extract episodes from a window of transcript entries.

        Args:
            entries: List of dicts with keys: id, role, content, created_at, tool_name
            channel: The channel/thread these entries belong to

        Returns:
            List of episode dicts. Each episode has:
            - intent: dict with type and direction
            - context: str
            - action: str
            - emotion: dict with valence and intensity
            - outcome: str
            - gist: str (structured summary, not one-liner)
            - salience_factors: dict
            - open_loops: list
            - transcript_ids: list[int] — which entry IDs this episode covers
            - entities: list[str] — entities mentioned
            - goal_tags: list[str] — active goal tags detected
            - emotional_valence: float (-1.0 to 1.0)
            - emotional_arousal: float (0.0 to 1.0)
            - traits: list[dict] — each has {key, value, kind, decay_class}
        """
        if not entries:
            return []

        if not self._prompt_template:
            logger.error("[EXTRACTOR] No prompt template loaded for episode-extraction")
            return []

        transcript_window = self._format_entries(entries)
        prompt = self._prompt_template.replace('{{transcript_window}}', transcript_window)
        prompt = prompt.replace('{{topic}}', channel)

        try:
            response = self._llm.send_message("", prompt)
        except Exception as e:
            logger.error(f"[EXTRACTOR] LLM call failed: {e}")
            return []

        raw = response.text if response else ""
        parsed = _safe_json_load(raw)

        if parsed is None:
            return []

        if not isinstance(parsed, list):
            logger.warning("[EXTRACTOR] LLM returned non-list JSON — expected array of episodes")
            return []

        valid_entry_ids = {entry['id'] for entry in entries if 'id' in entry}
        episodes = []
        for ep in parsed:
            if not isinstance(ep, dict):
                continue
            if not _REQUIRED_EPISODE_FIELDS.issubset(ep.keys()):
                missing = _REQUIRED_EPISODE_FIELDS - ep.keys()
                logger.warning(f"[EXTRACTOR] Episode missing fields {missing} — skipping")
                continue

            ep['emotional_valence'] = self._clamp(ep.get('emotional_valence'), -1.0, 1.0)
            ep['emotional_arousal'] = self._clamp(ep.get('emotional_arousal'), 0.0, 1.0)

            ids = ep.get('transcript_ids', [])
            ep['transcript_ids'] = [i for i in ids if i in valid_entry_ids]

            if not isinstance(ep.get('traits'), list):
                ep['traits'] = []

            if not isinstance(ep.get('entities'), list):
                ep['entities'] = []

            if not isinstance(ep.get('goal_tags'), list):
                ep['goal_tags'] = []

            if not isinstance(ep.get('open_loops'), list):
                ep['open_loops'] = []

            if not ep['transcript_ids']:
                logger.warning("[EXTRACTOR] Episode has no valid transcript_ids after filtering — skipping")
                continue

            ep['transcript_id_start'] = min(ep['transcript_ids'])
            ep['transcript_id_end'] = max(ep['transcript_ids'])

            episodes.append(ep)

        return episodes

    def _format_entries(self, entries: list[dict]) -> str:
        lines = []
        for entry in entries:
            entry_id = entry.get('id', '?')
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            tool_name = entry.get('tool_name')
            created_at = entry.get('created_at', '')

            if tool_name:
                lines.append(f"[{entry_id}] ({created_at}) {role} [{tool_name}]: {content}")
            else:
                lines.append(f"[{entry_id}] ({created_at}) {role}: {content}")

        return "\n".join(lines)

    def _clamp(self, value, min_val: float, max_val: float) -> Optional[float]:
        if value is None:
            return None
        try:
            return max(min_val, min(max_val, float(value)))
        except (TypeError, ValueError):
            return None
