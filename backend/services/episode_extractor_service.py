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
- Structured episode extraction with positional entry_range references

The extractor is a pure function — it does NOT store episodes. The caller handles storage.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from services.config_service import ConfigService
from services.llm_service import create_llm_service

logger = logging.getLogger(__name__)

_REQUIRED_EPISODE_FIELDS = {
    'intent', 'context', 'action', 'emotion', 'outcome', 'gist',
    'salience_factors', 'entry_range',
}

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "episode-extraction.md"


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
        logger.warning(f"[EXTRACTOR] Raw output: {cleaned[:500]}")
        return None


class EpisodeExtractorService:
    def __init__(self):
        config = ConfigService.resolve_agent_config("frontal-cortex-unified")
        self._llm = create_llm_service(config)
        try:
            self._prompt_template = _PROMPT_PATH.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"[EXTRACTOR] Failed to load prompt from {_PROMPT_PATH}: {e}")
            self._prompt_template = ""

    def extract(self, entries: list[dict], channel: str) -> list[dict]:
        """
        Extract episodes from a window of transcript entries.

        Args:
            entries: List of dicts with keys: id, role, content, created_at, tool_name
            channel: The channel/thread these entries belong to

        Returns:
            List of episode dicts. Each episode has:
            - intent, context, action, emotion, outcome, gist, salience_factors
            - open_loops, entities, goal_tags, emotional_valence, emotional_arousal, traits
            - transcript_ids: list[int] — computed from entry_range
            - transcript_id_start, transcript_id_end: int — min/max of transcript_ids
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

            entry_range = ep.get('entry_range', [])
            if not isinstance(entry_range, list) or len(entry_range) != 2:
                logger.warning("[EXTRACTOR] Episode has invalid entry_range — skipping")
                continue

            try:
                start, end = int(entry_range[0]), int(entry_range[1])
            except (TypeError, ValueError):
                logger.warning(f"[EXTRACTOR] entry_range {entry_range} not numeric — skipping")
                continue

            if end < start:
                start, end = end, start

            # entry_range refers to transcript row IDs (what the window prints as [id]).
            # Select all transcript rows within the window whose id falls in [start, end].
            window_ids = [entry['id'] for entry in entries if 'id' in entry]
            transcript_ids = [tid for tid in window_ids if start <= tid <= end]

            if not transcript_ids:
                logger.warning(
                    f"[EXTRACTOR] entry_range [{start}, {end}] matched no ids in window "
                    f"(window ids: {window_ids[0] if window_ids else '-'}..{window_ids[-1] if window_ids else '-'}) — skipping"
                )
                continue

            ep['transcript_ids'] = transcript_ids
            ep['transcript_id_start'] = min(transcript_ids)
            ep['transcript_id_end'] = max(transcript_ids)

            if not isinstance(ep.get('traits'), list):
                ep['traits'] = []

            if not isinstance(ep.get('entities'), list):
                ep['entities'] = []

            if not isinstance(ep.get('goal_tags'), list):
                ep['goal_tags'] = []

            if not isinstance(ep.get('open_loops'), list):
                ep['open_loops'] = []

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
            channel = entry.get('channel', '')

            if tool_name:
                lines.append(f"[{entry_id}] ({created_at}) {role} [{tool_name}]: {content}")
            elif channel:
                lines.append(f"[{entry_id}] ({created_at}) {role} [{channel}]: {content}")
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
