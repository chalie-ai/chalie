"""
Unit tests for FailureAnalysisService.

All tests use mocked LLM and embedding services so no real network calls are made.
SQLite I/O tests use a real temp-file database created with the ``procedural_memory``
schema so the full read-modify-write path is exercised without touching production data.

Pytest markers: @pytest.mark.unit (all tests).
"""

import json
import os
import sqlite3
import uuid
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.database_service import DatabaseService
from services.failure_analysis_service import (
    CONFIDENCE_THRESHOLD,
    DEDUP_SIMILARITY_THRESHOLD,
    MAX_LESSONS_PER_ACTION,
    FailureAnalysisService,
)

pytestmark = pytest.mark.unit


# ── Fixtures ────────────────────────────────────────────────────────────────


_PROCEDURAL_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS procedural_memory (
    id TEXT PRIMARY KEY,
    action_name TEXT NOT NULL UNIQUE,
    total_attempts INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    avg_reward REAL DEFAULT 0.0,
    weight REAL DEFAULT 1.0,
    reward_history TEXT DEFAULT '[]',
    context_stats TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)
"""


@pytest.fixture
def tmp_db(tmp_path):
    """
    Real SQLite database (temp file) with the ``procedural_memory`` schema.

    Yields a :class:`DatabaseService` pointed at the temp file.  Each test gets
    its own isolated database.
    """
    db_path = str(tmp_path / "test_failure.db")
    conn = sqlite3.connect(db_path)
    conn.execute(_PROCEDURAL_MEMORY_DDL)
    conn.commit()
    conn.close()

    db = DatabaseService(db_path)
    yield db


@pytest.fixture
def fas(tmp_db):
    """
    :class:`FailureAnalysisService` wired to the temp SQLite database.

    LLM and embedding service must be patched per-test as needed.
    """
    return FailureAnalysisService(tmp_db)


def _make_llm_mock(response_text: str) -> MagicMock:
    """
    Create a mock LLM service whose ``send_message`` returns ``response_text``.

    Args:
        response_text: JSON string to return as the LLM response.

    Returns:
        Configured MagicMock.
    """
    from services.llm_service import LLMResponse
    mock = MagicMock()
    mock.send_message.return_value = LLMResponse(
        text=response_text,
        model="test-model",
        provider="mock",
    )
    return mock


def _make_emb_mock(vector_map: dict) -> MagicMock:
    """
    Create a mock embedding service that returns deterministic numpy vectors.

    ``vector_map`` maps text strings to 1-D numpy arrays (must be L2-normalised
    by the caller to ensure dot-product == cosine similarity).  Texts not in the
    map receive a zero vector (cosine similarity 0 with everything).

    Args:
        vector_map: Dict[str, np.ndarray] of text → embedding vector.

    Returns:
        Configured MagicMock.
    """
    mock = MagicMock()

    def _embed_np(text: str) -> np.ndarray:
        """Return the configured embedding for ``text`` or a zero vector."""
        return vector_map.get(text, np.zeros(4, dtype=np.float32))

    mock.generate_embedding_np.side_effect = _embed_np
    return mock


def _unit_vec(*components) -> np.ndarray:
    """
    Return an L2-normalised numpy vector from the given components.

    Args:
        *components: Float values for each dimension.

    Returns:
        L2-normalised numpy float32 array.
    """
    v = np.array(components, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _good_analysis(**overrides) -> dict:
    """
    Return a minimal valid analysis dict with confidence above the threshold.

    Args:
        **overrides: Key-value pairs that override the defaults.

    Returns:
        Analysis dict suitable for passing to :meth:`FailureAnalysisService.store_lesson`.
    """
    base = {
        "blame": "tool_choice",
        "root_cause": "Wrong tool selected for the task",
        "lesson": "Always verify the tool supports the required operation before invoking it.",
        "affected_skill": "web_search",
        "severity": "minor",
        "confidence": 0.80,
        "generalizable": True,
    }
    base.update(overrides)
    return base


def _seed_lesson(fas: FailureAnalysisService, action_name: str, lesson: dict):
    """
    Directly write a lesson dict into the ``procedural_memory`` context_stats column.

    Bypasses embedding dedup so precise lesson counts can be set up for tests.

    Args:
        fas: Service instance (provides ``db_service``).
        action_name: The action_name row to target.
        lesson: Lesson dict to store inside ``__failure_lessons``.
    """
    with fas.db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO procedural_memory (id, action_name, total_attempts, total_successes, weight)
            VALUES (?, ?, 0, 0, 1.0)
            ON CONFLICT (action_name) DO NOTHING
            """,
            (str(uuid.uuid4()), action_name),
        )
        cursor.execute(
            "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
            (action_name,),
        )
        row = cursor.fetchone()
        raw = row[0] if not isinstance(row, dict) else row["context_stats"]
        cs = json.loads(raw) if raw else {}
        cs.setdefault("__failure_lessons", []).append(lesson)
        cursor.execute(
            "UPDATE procedural_memory SET context_stats = ? WHERE action_name = ?",
            (json.dumps(cs), action_name),
        )
        cursor.close()


# ── analyze() ───────────────────────────────────────────────────────────────


class TestAnalyze:
    """Tests for :meth:`FailureAnalysisService.analyze`."""

    def test_analyze_returns_structured_output(self, fas):
        """
        Happy path: valid JSON response with confidence > threshold → analysis returned.
        """
        payload = json.dumps({
            "blame": "tool_choice",
            "root_cause": "Wrong tool selected",
            "lesson": "Verify tool capability before use.",
            "affected_skill": "web_search",
            "severity": "minor",
            "confidence": 0.85,
            "generalizable": True,
        })
        llm_mock = _make_llm_mock(payload)

        failure_context = {
            "original_request": "Search for recent news",
            "action_type": "web_search",
            "action_intent": {"query": "recent news"},
            "action_result": {"status": "error", "result": "tool not found"},
            "error_signals": {"status": "error", "error_text": "tool not supported"},
        }

        with patch("services.llm_service.create_llm_service", return_value=llm_mock), \
             patch("services.config_service.ConfigService.resolve_agent_config", return_value={}):
            result = fas.analyze(failure_context)

        assert result is not None
        assert result["blame"] == "tool_choice"
        assert "lesson" in result
        assert result["confidence"] >= CONFIDENCE_THRESHOLD
        assert result["severity"] in ("minor", "major")

    def test_analyze_below_confidence_returns_none(self, fas):
        """
        LLM returns confidence below threshold → :meth:`analyze` returns ``None``.
        """
        payload = json.dumps({
            "blame": "ambiguous_goal",
            "root_cause": "Goal was unclear",
            "lesson": "Ask for clarification when intent is ambiguous.",
            "affected_skill": "plan_step",
            "severity": "minor",
            "confidence": 0.40,
            "generalizable": False,
        })
        llm_mock = _make_llm_mock(payload)

        with patch("services.llm_service.create_llm_service", return_value=llm_mock), \
             patch("services.config_service.ConfigService.resolve_agent_config", return_value={}):
            result = fas.analyze({"original_request": "Do something", "action_type": "plan_step",
                                  "error_signals": {}})

        assert result is None

    def test_analyze_parses_fenced_json(self, fas):
        """
        LLM wraps JSON in a markdown code fence → fence is stripped and parsed correctly.
        """
        inner = {
            "blame": "external",
            "root_cause": "Network timeout",
            "lesson": "Retry with exponential back-off on network errors.",
            "affected_skill": "http_fetch",
            "severity": "minor",
            "confidence": 0.75,
            "generalizable": True,
        }
        fenced = f"```json\n{json.dumps(inner)}\n```"
        llm_mock = _make_llm_mock(fenced)

        failure_context = {
            "action_type": "http_fetch",
            "error_signals": {"status": "timeout", "error_text": "connection timeout after 30s"},
        }

        with patch("services.llm_service.create_llm_service", return_value=llm_mock), \
             patch("services.config_service.ConfigService.resolve_agent_config", return_value={}):
            result = fas.analyze(failure_context)

        assert result is not None
        assert result["blame"] == "external"

    def test_analyze_unparseable_response_returns_none(self, fas):
        """
        LLM returns plain prose with no JSON-extractable block → ``None`` returned.
        """
        llm_mock = _make_llm_mock("I cannot determine the cause of this failure at this time.")

        with patch("services.llm_service.create_llm_service", return_value=llm_mock), \
             patch("services.config_service.ConfigService.resolve_agent_config", return_value={}):
            result = fas.analyze({"action_type": "recall", "error_signals": {}})

        assert result is None


# ── _sanity_check() ──────────────────────────────────────────────────────────


class TestSanityCheck:
    """Tests for :meth:`FailureAnalysisService._sanity_check`."""

    def test_sanity_check_reduces_confidence_tool_choice_no_evidence(self, fas):
        """
        ``blame='tool_choice'`` with no tool-related keywords and status 'success'
        → confidence reduced by 0.20.
        """
        analysis = _good_analysis(blame="tool_choice", confidence=0.80)
        error_signals = {"status": "success", "error_text": ""}

        result = fas._sanity_check(analysis, error_signals)

        assert result["confidence"] == pytest.approx(0.60, abs=0.01)

    def test_sanity_check_passes_valid_tool_choice(self, fas):
        """
        ``blame='tool_choice'`` with "tool" keyword in error signals
        → confidence unchanged.
        """
        analysis = _good_analysis(blame="tool_choice", confidence=0.80)
        error_signals = {"status": "error", "error_text": "unknown tool handler"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["confidence"] == pytest.approx(0.80, abs=0.01)

    def test_sanity_check_external_no_network_keywords(self, fas):
        """
        ``blame='external'`` with no network/timeout keywords → confidence reduced.
        """
        analysis = _good_analysis(blame="external", confidence=0.80)
        error_signals = {"status": "error", "error_text": "key error in mapping"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["confidence"] < 0.80

    def test_sanity_check_external_with_timeout_keyword(self, fas):
        """
        ``blame='external'`` with "timeout" in error text → confidence unchanged.
        """
        analysis = _good_analysis(blame="external", confidence=0.80)
        error_signals = {"status": "error", "error_text": "connection timeout exceeded"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["confidence"] == pytest.approx(0.80, abs=0.01)

    def test_sanity_check_stale_memory_non_memory_action(self, fas):
        """
        ``blame='stale_memory'`` for a non-memory action type → confidence reduced.
        """
        analysis = _good_analysis(
            blame="stale_memory",
            affected_skill="web_search",
            confidence=0.80,
        )
        error_signals = {"status": "error"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["confidence"] < 0.80

    def test_sanity_check_stale_memory_memory_action(self, fas):
        """
        ``blame='stale_memory'`` for a memory action type → confidence unchanged.
        """
        analysis = _good_analysis(
            blame="stale_memory",
            affected_skill="recall",
            confidence=0.80,
        )
        error_signals = {"status": "error"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["confidence"] == pytest.approx(0.80, abs=0.01)

    def test_sanity_check_demotes_major_to_minor_when_low_confidence(self, fas):
        """
        ``severity='major'`` but post-check confidence falls below 0.50
        → severity demoted to ``'minor'``.
        """
        # Start at 0.65; no tool keywords so -0.20 → 0.45 < 0.50 → demote
        analysis = _good_analysis(blame="tool_choice", severity="major", confidence=0.65)
        error_signals = {"status": "success", "error_text": "some unrelated message"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["severity"] == "minor"
        assert result["confidence"] < 0.50

    def test_sanity_check_keeps_major_above_threshold(self, fas):
        """
        ``severity='major'`` with tool keyword → confidence stays above 0.50
        → severity remains ``'major'``.
        """
        analysis = _good_analysis(blame="tool_choice", severity="major", confidence=0.80)
        error_signals = {"status": "error", "error_text": "tool not recognised"}

        result = fas._sanity_check(analysis, error_signals)

        assert result["severity"] == "major"


# ── store_lesson() ───────────────────────────────────────────────────────────


class TestStoreLesson:
    """Tests for :meth:`FailureAnalysisService.store_lesson`."""

    def test_store_lesson_new(self, fas):
        """
        First lesson for an action_name → stored and retrievable via raw DB query.
        """
        analysis = _good_analysis()
        same_vec = _unit_vec(1, 0, 0, 0)
        emb_mock = _make_emb_mock({analysis["lesson"]: same_vec})

        with patch("services.embedding_service.get_embedding_service", return_value=emb_mock):
            result = fas.store_lesson(analysis, "web_search")

        assert result is True

        # Verify lesson is in the DB.
        with fas.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                ("web_search",),
            )
            row = cursor.fetchone()
            cursor.close()

        assert row is not None
        cs = json.loads(row[0] if not isinstance(row, dict) else row["context_stats"])
        lessons = cs.get("__failure_lessons", [])
        assert len(lessons) == 1
        assert lessons[0]["blame"] == "tool_choice"
        assert lessons[0]["times_seen"] == 1

    def test_store_lesson_dedup_merge(self, fas):
        """
        Second lesson with cosine similarity ≥ threshold → ``times_seen`` incremented,
        no new entry created.
        """
        first_text = "Always verify the tool supports the required operation before invoking it."
        second_text = "Verify tool supports the operation before invoking."

        # Both texts map to the SAME vector → similarity = 1.0 > threshold.
        shared_vec = _unit_vec(1, 0, 0, 0)
        emb_mock = _make_emb_mock({
            first_text: shared_vec,
            second_text: shared_vec,
        })

        analysis1 = _good_analysis(lesson=first_text)
        analysis2 = _good_analysis(lesson=second_text, confidence=0.90)

        with patch("services.embedding_service.get_embedding_service", return_value=emb_mock):
            fas.store_lesson(analysis1, "web_search")
            fas.store_lesson(analysis2, "web_search")

        with fas.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                ("web_search",),
            )
            row = cursor.fetchone()
            cursor.close()

        cs = json.loads(row[0] if not isinstance(row, dict) else row["context_stats"])
        lessons = cs["__failure_lessons"]
        assert len(lessons) == 1, "Duplicate lesson should be merged, not appended"
        assert lessons[0]["times_seen"] == 2
        # Confidence should be updated to the higher value.
        assert lessons[0]["confidence"] == pytest.approx(0.90, abs=0.01)

    def test_store_lesson_dedup_separate(self, fas):
        """
        Second lesson with cosine similarity below threshold → both lessons stored
        as separate entries.
        """
        text_a = "Always verify the tool before invoking."
        text_b = "Check network connectivity before making external calls."

        # Orthogonal vectors → dot product = 0.0 < threshold.
        vec_a = _unit_vec(1, 0, 0, 0)
        vec_b = _unit_vec(0, 1, 0, 0)
        emb_mock = _make_emb_mock({text_a: vec_a, text_b: vec_b})

        analysis_a = _good_analysis(lesson=text_a, blame="tool_choice")
        analysis_b = _good_analysis(lesson=text_b, blame="external")

        with patch("services.embedding_service.get_embedding_service", return_value=emb_mock):
            fas.store_lesson(analysis_a, "combined_action")
            fas.store_lesson(analysis_b, "combined_action")

        with fas.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                ("combined_action",),
            )
            row = cursor.fetchone()
            cursor.close()

        cs = json.loads(row[0] if not isinstance(row, dict) else row["context_stats"])
        lessons = cs["__failure_lessons"]
        assert len(lessons) == 2, "Dissimilar lessons should be stored as separate entries"
        blames = {l["blame"] for l in lessons}
        assert "tool_choice" in blames
        assert "external" in blames

    def test_store_lesson_cap_eviction(self, fas):
        """
        When the lesson count reaches the cap + 1, the entry with the lowest
        ``times_seen`` (and oldest ``created_at`` as tiebreak) is evicted.
        """
        # Seed MAX_LESSONS_PER_ACTION lessons directly (bypasses dedup).
        for i in range(MAX_LESSONS_PER_ACTION):
            _seed_lesson(fas, "recall", {
                "lesson_hash": f"hash_{i:03d}",
                "blame": "stale_memory",
                "root_cause": f"Root cause {i}",
                "lesson": f"Lesson text number {i} to fill the store.",
                "affected_skill": "recall",
                "severity": "minor",
                "confidence": 0.70,
                "generalizable": True,
                "times_seen": i + 1,           # ascending: lesson 0 has times_seen=1
                "created_at": f"2026-01-{i+1:02d}T00:00:00",
                "updated_at": f"2026-01-{i+1:02d}T00:00:00",
            })

        # Verify we seeded exactly MAX lessons.
        with fas.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                ("recall",),
            )
            row = cursor.fetchone()
            cursor.close()
        cs = json.loads(row[0] if not isinstance(row, dict) else row["context_stats"])
        assert len(cs["__failure_lessons"]) == MAX_LESSONS_PER_ACTION

        # Store one more lesson — should evict the one with times_seen=1 (index 0).
        new_lesson_text = "The 51st lesson that triggers eviction of the oldest."
        new_vec = _unit_vec(0, 0, 0, 1)
        existing_vecs = {
            f"Lesson text number {i} to fill the store.": _unit_vec(1, 0, 0, 0)
            for i in range(MAX_LESSONS_PER_ACTION)
        }
        existing_vecs[new_lesson_text] = new_vec
        emb_mock = _make_emb_mock(existing_vecs)

        with patch("services.embedding_service.get_embedding_service", return_value=emb_mock):
            fas.store_lesson(
                _good_analysis(lesson=new_lesson_text, blame="wrong_assumption"),
                "recall",
            )

        with fas.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                ("recall",),
            )
            row = cursor.fetchone()
            cursor.close()
        cs = json.loads(row[0] if not isinstance(row, dict) else row["context_stats"])
        lessons = cs["__failure_lessons"]

        assert len(lessons) == MAX_LESSONS_PER_ACTION, "Cap must be enforced after eviction"

        # The lesson with times_seen=1 (Lesson text number 0) must be gone.
        lesson_texts = [l["lesson"] for l in lessons]
        assert "Lesson text number 0 to fill the store." not in lesson_texts
        assert new_lesson_text in lesson_texts


# ── get_relevant_lessons() ────────────────────────────────────────────────────


class TestGetRelevantLessons:
    """Tests for :meth:`FailureAnalysisService.get_relevant_lessons`."""

    def _make_lesson(self, lesson_text: str, severity: str, times_seen: int,
                     updated_at: str = "2026-01-10T00:00:00") -> dict:
        """
        Build a minimal lesson dict for seeding into ``procedural_memory``.

        Args:
            lesson_text: The lesson string.
            severity: ``'minor'`` or ``'major'``.
            times_seen: How many times this lesson has been observed.
            updated_at: ISO timestamp for recency sorting.

        Returns:
            Lesson dict ready for :func:`_seed_lesson`.
        """
        import hashlib
        return {
            "lesson_hash": hashlib.md5(lesson_text.encode()).hexdigest()[:32],
            "blame": "tool_choice",
            "root_cause": "Some root cause",
            "lesson": lesson_text,
            "affected_skill": "web_search",
            "severity": severity,
            "confidence": 0.75,
            "generalizable": True,
            "times_seen": times_seen,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": updated_at,
        }

    def test_get_relevant_lessons_filters_minor_seen_once(self, fas):
        """
        Lessons with ``severity='minor'`` and ``times_seen=1`` are excluded
        (below both qualifying criteria).
        """
        _seed_lesson(fas, "web_search", self._make_lesson("Minor one-off lesson.", "minor", 1))

        result = fas.get_relevant_lessons("web_search")

        assert result == [], "Minor lessons seen only once must not be returned"

    def test_get_relevant_lessons_returns_major_seen_once(self, fas):
        """
        A single lesson with ``severity='major'`` qualifies even if ``times_seen=1``.
        """
        _seed_lesson(fas, "web_search", self._make_lesson("Critical major failure.", "major", 1))

        result = fas.get_relevant_lessons("web_search")

        assert len(result) == 1
        assert result[0]["severity"] == "major"

    def test_get_relevant_lessons_returns_minor_seen_twice(self, fas):
        """
        A minor lesson with ``times_seen=2`` qualifies via the recurrence criterion.
        """
        _seed_lesson(fas, "recall", self._make_lesson("Recurring minor issue.", "minor", 2))

        result = fas.get_relevant_lessons("recall")

        assert len(result) == 1
        assert result[0]["lesson"] == "Recurring minor issue."

    def test_get_relevant_lessons_limit_3(self, fas):
        """
        Five qualifying lessons → only the top 3 (by ``times_seen`` descending) returned.
        """
        for i, times in enumerate([10, 8, 6, 4, 2]):
            _seed_lesson(
                fas, "plan_step",
                self._make_lesson(
                    f"Qualifying lesson {i}.",
                    "major",
                    times,
                    updated_at=f"2026-01-{i+1:02d}T00:00:00",
                ),
            )

        result = fas.get_relevant_lessons("plan_step")

        assert len(result) == 3
        # Top 3 should have times_seen 10, 8, 6 in that order.
        assert result[0]["times_seen"] == 10
        assert result[1]["times_seen"] == 8
        assert result[2]["times_seen"] == 6

    def test_get_relevant_lessons_strips_internal_fields(self, fas):
        """
        ``lesson_hash`` (internal bookkeeping) must not appear in returned lesson dicts.
        """
        _seed_lesson(fas, "web_search", self._make_lesson("Major insight.", "major", 1))

        result = fas.get_relevant_lessons("web_search")

        assert len(result) == 1
        assert "lesson_hash" not in result[0]

    def test_get_relevant_lessons_unknown_action_returns_empty(self, fas):
        """
        Querying an ``action_name`` that has no ``procedural_memory`` row returns ``[]``.
        """
        result = fas.get_relevant_lessons("nonexistent_action")

        assert result == []


# ── get_stats() ───────────────────────────────────────────────────────────────


class TestGetStats:
    """Tests for :meth:`FailureAnalysisService.get_stats`."""

    def test_get_stats_aggregation(self, fas):
        """
        Multiple action_names with lessons → correct blame distribution, counts,
        and total.
        """
        import hashlib

        def _l(text: str, blame: str, times: int) -> dict:
            """Build a minimal lesson dict for direct seeding."""
            return {
                "lesson_hash": hashlib.md5(text.encode()).hexdigest()[:32],
                "blame": blame,
                "root_cause": "Root cause",
                "lesson": text,
                "affected_skill": "test",
                "severity": "minor",
                "confidence": 0.75,
                "generalizable": True,
                "times_seen": times,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-05T00:00:00",
            }

        # Seed two action_names with different blame distributions.
        _seed_lesson(fas, "recall", _l("Recall lesson A.", "stale_memory", 3))
        _seed_lesson(fas, "recall", _l("Recall lesson B.", "stale_memory", 1))
        _seed_lesson(fas, "web_search", _l("Web lesson.", "tool_choice", 5))

        stats = fas.get_stats()

        assert stats["total_lessons"] == 3
        assert stats["blame_distribution"]["stale_memory"] == 2
        assert stats["blame_distribution"]["tool_choice"] == 1
        assert stats["lesson_counts_by_action"]["recall"] == 2
        assert stats["lesson_counts_by_action"]["web_search"] == 1

    def test_get_stats_empty_db(self, fas):
        """
        Empty ``procedural_memory`` table → stats dict returned with zero totals.
        """
        stats = fas.get_stats()

        assert stats["total_lessons"] == 0
        assert stats["blame_distribution"] == {}
        assert stats["lesson_counts_by_action"] == {}
        assert stats["top_lessons_by_frequency"] == []
        assert stats["recent_lessons"] == []

    def test_get_stats_top_lessons_ordered_by_times_seen(self, fas):
        """
        ``top_lessons_by_frequency`` entries are ordered by ``times_seen`` descending.
        """
        import hashlib

        for i, times in enumerate([1, 5, 3]):
            _seed_lesson(fas, "recall", {
                "lesson_hash": hashlib.md5(f"lesson {i}".encode()).hexdigest()[:32],
                "blame": "tool_choice",
                "root_cause": "Root",
                "lesson": f"Lesson {i}",
                "affected_skill": "recall",
                "severity": "minor",
                "confidence": 0.70,
                "generalizable": True,
                "times_seen": times,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            })

        stats = fas.get_stats()
        top = stats["top_lessons_by_frequency"]

        assert top[0]["times_seen"] == 5
        assert top[1]["times_seen"] == 3
        assert top[2]["times_seen"] == 1
