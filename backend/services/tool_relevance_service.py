"""
Tool Relevance Service — Embedding-based tool matching.

Replaces regex-based TOOL_HINT_PATTERNS with cosine similarity against
pre-embedded tool/skill descriptions and user-intent examples.
Single semantic space via EmbeddingService.

Singleton. On first call, loads or builds a disk-persisted embedding cache.
Cache invalidation: mtime-based. Cache is rebuilt if any manifest file or
this file (which contains SKILL_DESCRIPTIONS) is newer than the cache.

Each tool/skill gets multiple embeddings (description + each example).
Per-request: embeds prompt, scores relevance via max dot product across
all embeddings for that tool/skill.

Cache files (configs/generated/):
  tool_relevance_cache.npz   — numpy arrays, keys: "{kind}__{name}__{idx}"
  tool_relevance_cache.json  — {"created_at": str, "skills": {...}, "tools": {...}}
"""

import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Singleton instance
_instance = None

# User-intent phrases per innate skill.
# First entry is the capability description; rest are example user phrases.
SKILL_DESCRIPTIONS = {
    'recall': [
        'Search memory, retrieve stored information, look up what I know',
        'Do you remember what I told you about X?',
        'What do you know about this topic?',
        'Look up that thing I mentioned earlier',
        'Can you recall our previous conversation about Y?',
        'Find that fact I shared with you last week',
        'What did I say about this before?',
        'Search your memory for anything related to this',
    ],
    'memorize': [
        'Store information, save a note, remember this fact',
        'Remember this for later',
        'Write this down so you don\'t forget',
        'Save my preference for next time',
        'Keep a note that I like X',
        'Memorize this — I\'ll need it later',
        'Store this as a fact',
        'Make a mental note of this',
    ],
    'introspect': [
        'Self-examine internal state, check how much I know, feeling of knowing',
        'How confident are you about this?',
        'What do you currently know about this topic?',
        'How are you doing right now?',
        'Check your internal state',
        'How well do you remember our conversations?',
        'What is your current context on this?',
    ],
    'associate': [
        'Find related concepts, explore connections, brainstorm associations',
        'What else is related to this idea?',
        'Find connections between these topics',
        'What concepts are linked to X?',
        'Brainstorm around this subject',
        'Explore what this is associated with',
        'What other things come to mind from this?',
    ],
    'list': [
        'Manage named lists: add, remove, check off, or view items in shopping, to-do, and other lists',
        'Add milk to my shopping list',
        'Remove eggs from the grocery list',
        'What\'s on my shopping list?',
        'Check off bread from the list',
        'Create a new to-do list',
        'Show me my list',
        'I need to pick up milk, eggs, and butter',
        'Put bananas on the list',
        'Cross that off my list',
        'Clear everything from my grocery list',
        'Rename my shopping list to groceries',
        'Add those to my list',
        'Put them on my shopping list',
        'Add all of that to the grocery list',
        'Can you add these to my list?',
    ],
    'schedule': [
        'Set reminders, schedule tasks, create appointments and recurring events',
        'Remind me to call the doctor at 2pm tomorrow',
        'Schedule a dentist appointment for Friday',
        'Set a reminder in 30 minutes',
        'Add a task for next week',
        'Create a reminder every Monday morning',
        'Remind me about this tonight',
        'Don\'t let me forget to send that email',
        'Set an alarm for my meeting',
    ],
    'goal': [
        'Create, update, and track personal goals and objectives',
        'I want to set a goal to exercise three times a week',
        'Show me my current goals',
        'Mark my fitness goal as complete',
        'Add a new goal to learn Python',
        'How am I doing on my goals?',
        'I\'d like to work toward reading more books',
        'Track my progress on this goal',
        'Delete that goal',
    ],
    'focus': [
        'Start and manage deep focus or work sessions, Pomodoro-style timers',
        'Start a focus session for 25 minutes',
        'I need to focus right now',
        'Enter deep work mode',
        'Stop my focus session',
        'Begin a Pomodoro',
        'I\'m going to focus on writing for an hour',
        'Block distractions for 45 minutes',
    ],
    'autobiography': [
        'Generate a personal autobiography or life summary based on stored memories and interactions',
        'Tell me about myself based on what you know',
        'Generate my autobiography',
        'What have you learned about me over time?',
        'Summarize who I am',
        'Write my personal story',
        'What kind of person am I according to you?',
        'Give me a portrait of myself from our conversations',
    ],
}

MIN_RELEVANCE_THRESHOLD = 0.35

# Cache paths (resolved relative to this file's repo root)
_REPO_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _REPO_ROOT / "configs" / "generated"
_CACHE_NPZ = _CACHE_DIR / "tool_relevance_cache.npz"
_CACHE_META = _CACHE_DIR / "tool_relevance_cache.json"


class ToolRelevanceService:
    """Embedding-based tool and skill relevance scoring with multi-embedding max-score."""

    def __new__(cls, *args, **kwargs):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # name -> list of embeddings
        self._skill_embeddings: Dict[str, List[np.ndarray]] = {}
        self._tool_embeddings: Dict[str, List[np.ndarray]] = {}
        self._embeddings_ready = False
        self._emb_service = None

    # ── Cache validity ───────────────────────────────────────────────

    def _cache_is_valid(self) -> bool:
        """
        Return True if the cache is up-to-date.

        The cache is stale if any of these is newer than the .npz file:
          - Any tools/*/manifest.json  (tool examples / descriptions changed)
          - This file itself           (SKILL_DESCRIPTIONS changed)
        """
        if not _CACHE_NPZ.exists():
            return False

        cache_mtime = _CACHE_NPZ.stat().st_mtime

        # Check this source file (catches SKILL_DESCRIPTIONS edits)
        if Path(__file__).stat().st_mtime > cache_mtime:
            logger.info("[TOOL RELEVANCE] Service file newer than cache — will rebuild")
            return False

        # Check all manifest files
        tools_dir = _REPO_ROOT / "tools"
        for manifest_path in tools_dir.glob("*/manifest.json"):
            if manifest_path.stat().st_mtime > cache_mtime:
                logger.info(
                    f"[TOOL RELEVANCE] {manifest_path.name} newer than cache — will rebuild"
                )
                return False

        return True

    # ── Disk cache ───────────────────────────────────────────────────

    def _load_cache(self) -> bool:
        """Load embeddings from disk. Returns True on success."""
        try:
            data = np.load(_CACHE_NPZ)

            for key in data.files:
                kind, name, idx = key.split("__", 2)
                vec = data[key]
                if kind == "skill":
                    self._skill_embeddings.setdefault(name, []).append(vec)
                else:
                    self._tool_embeddings.setdefault(name, []).append(vec)

            total_skills = sum(len(v) for v in self._skill_embeddings.values())
            total_tools = sum(len(v) for v in self._tool_embeddings.values())
            logger.info(
                f"[TOOL RELEVANCE] Loaded cache — "
                f"{total_skills} skill + {total_tools} tool embeddings"
            )
            return True

        except Exception as e:
            logger.warning(f"[TOOL RELEVANCE] Cache load failed: {e}")
            self._skill_embeddings.clear()
            self._tool_embeddings.clear()
            return False

    def _save_cache(self):
        """Persist current embeddings to disk."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

            arrays = {}
            for name, vecs in self._skill_embeddings.items():
                for idx, vec in enumerate(vecs):
                    arrays[f"skill__{name}__{idx}"] = vec
            for name, vecs in self._tool_embeddings.items():
                for idx, vec in enumerate(vecs):
                    arrays[f"tool__{name}__{idx}"] = vec

            np.savez(_CACHE_NPZ, **arrays)

            meta = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "skills": {k: len(v) for k, v in self._skill_embeddings.items()},
                "tools": {k: len(v) for k, v in self._tool_embeddings.items()},
            }
            _CACHE_META.write_text(json.dumps(meta, indent=2))

            logger.info(
                f"[TOOL RELEVANCE] Cache saved ({len(arrays)} vectors)"
            )
        except Exception as e:
            logger.warning(f"[TOOL RELEVANCE] Cache save failed: {e}")

    # ── Embedding build ──────────────────────────────────────────────

    def _ensure_embeddings(self):
        """Load from disk cache or (re)build and save if any source is newer."""
        if self._embeddings_ready:
            return

        try:
            from services.embedding_service import EmbeddingService
            self._emb_service = EmbeddingService()

            if self._cache_is_valid() and self._load_cache():
                self._embeddings_ready = True
                return

            # Cache miss — compute from scratch
            logger.info("[TOOL RELEVANCE] Building embeddings from scratch...")

            # --- Innate skills ---
            skill_keys = []
            skill_texts = []
            for skill_name, phrases in SKILL_DESCRIPTIONS.items():
                for idx, phrase in enumerate(phrases):
                    skill_keys.append((skill_name, idx))
                    skill_texts.append(phrase)

            if skill_texts:
                vecs = self._emb_service.generate_embeddings_batch(skill_texts)
                for (skill_name, _), vec in zip(skill_keys, vecs):
                    self._skill_embeddings.setdefault(skill_name, []).append(vec)

            # --- Dynamic tools from registry ---
            try:
                from services.tool_registry_service import ToolRegistryService
                registry = ToolRegistryService()

                tool_keys = []
                tool_texts = []

                for tool_name, tool in registry.tools.items():
                    manifest = tool['manifest']
                    if manifest.get('trigger', {}).get('type') != 'on_demand':
                        continue

                    desc = manifest.get('description', tool_name)
                    tool_keys.append((tool_name, 0))
                    tool_texts.append(desc)

                    for idx, ex in enumerate(manifest.get('examples', []), start=1):
                        phrase = ex.get('description', '').strip()
                        if phrase:
                            tool_keys.append((tool_name, idx))
                            tool_texts.append(phrase)

                if tool_texts:
                    vecs = self._emb_service.generate_embeddings_batch(tool_texts)
                    for (tool_name, _), vec in zip(tool_keys, vecs):
                        self._tool_embeddings.setdefault(tool_name, []).append(vec)

            except Exception as e:
                logger.debug(f"[TOOL RELEVANCE] Tool registry not available: {e}")

            self._save_cache()
            self._embeddings_ready = True

        except Exception as e:
            logger.error(f"[TOOL RELEVANCE] Failed to build embedding cache: {e}")
            self._embeddings_ready = False

    # ── Scoring ──────────────────────────────────────────────────────

    def _max_score(self, prompt_vec: np.ndarray, vecs: List[np.ndarray]) -> float:
        """Return the highest cosine similarity between prompt and any embedding."""
        return float(max(np.dot(prompt_vec, v) for v in vecs))

    def score_relevance(self, prompt_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Score tool/skill relevance for a prompt using max cosine similarity
        across all embeddings for each tool/skill.

        Returns:
            {
                'max_relevance_score': float,
                'relevant_tools': list of {'name': str, 'score': float, 'type': 'tool'|'skill'},
                'relevant_skills': list of str (skill names above threshold),
                'all_scores': dict of name -> score,
            }
        """
        try:
            self._ensure_embeddings()

            if not self._embeddings_ready or not self._emb_service:
                return self._empty_result()

            prompt_vec = self._emb_service.generate_embedding_np(prompt_text)

            all_scores = {}
            for name, vecs in self._skill_embeddings.items():
                all_scores[name] = self._max_score(prompt_vec, vecs)
            for name, vecs in self._tool_embeddings.items():
                all_scores[name] = self._max_score(prompt_vec, vecs)

            above_threshold = [
                (name, score) for name, score in all_scores.items()
                if score >= MIN_RELEVANCE_THRESHOLD
            ]
            above_threshold.sort(key=lambda x: x[1], reverse=True)
            top_items = above_threshold[:top_k]

            max_score = max(all_scores.values()) if all_scores else 0.0

            relevant_tools = []
            relevant_skills = []
            for name, score in top_items:
                item_type = 'skill' if name in SKILL_DESCRIPTIONS else 'tool'
                relevant_tools.append({'name': name, 'score': score, 'type': item_type})
                if item_type == 'skill':
                    relevant_skills.append(name)

            return {
                'max_relevance_score': max_score,
                'relevant_tools': relevant_tools,
                'relevant_skills': relevant_skills,
                'all_scores': all_scores,
            }

        except Exception as e:
            logger.warning(f"[TOOL RELEVANCE] Scoring failed: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'max_relevance_score': 0.0,
            'relevant_tools': [],
            'relevant_skills': [],
            'all_scores': {},
        }
