"""Feature tests for SubagentProcessor class constants, timeout override clamping,
pre-seed construction, and blocklist enforcement.

Tests that require the real embedding pipeline (test_pre_seed_*) are skipped
when abilities.sqlite or the embedding model is unavailable — this is expected
in CI environments without the ONNX model asset downloaded.

The one test that patches find_tools_query (test_pre_seed_blocklist_enforced)
is explicitly documented below. It is the minimum viable mock: patching at the
import site of abilities.find_tools so we can inject a known return value that
includes blocklisted names — there is no other way to construct this scenario
without manipulating the embedding index.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# B7–B9. _max_timeout_seconds property
# ---------------------------------------------------------------------------


class TestMaxTimeoutOverride:
    def test_per_instance_max_timeout_override_clamps_to_class_constant(self):
        """Override larger than MAX_TIMEOUT (7200) must be clamped to 7200."""
        from services.subagent_processor import SubagentProcessor

        proc = SubagentProcessor(raw_input="x", max_timeout_override=99999)
        assert proc._max_timeout_seconds == 7200

    def test_per_instance_max_timeout_override_below_class_constant_used_as_is(self):
        """Override of 300 is below 7200 — must be honoured as-is."""
        from services.subagent_processor import SubagentProcessor

        proc = SubagentProcessor(raw_input="x", max_timeout_override=300)
        assert proc._max_timeout_seconds == 300

    def test_per_instance_no_override_falls_back_to_class_constant(self):
        """No override → falls back to class MAX_TIMEOUT (7200)."""
        from services.subagent_processor import SubagentProcessor

        proc = SubagentProcessor(raw_input="x")
        assert proc._max_timeout_seconds == 7200


# ---------------------------------------------------------------------------
# B10. Pre-seed runs once at construction (real embedding + abilities.sqlite)
# ---------------------------------------------------------------------------


class TestPreSeedRealDB:
    def test_pre_seed_runs_once_at_construction(self):
        """_preseeded_tools is populated at __init__ against the real abilities.sqlite.

        Skipped when the embedding model or DB asset is unavailable (CI without
        downloaded ONNX model).
        """
        from pathlib import Path

        abilities_db = Path(__file__).resolve().parent.parent / "abilities" / "assets" / "abilities.sqlite"
        if not abilities_db.exists():
            pytest.skip("abilities.sqlite not present — skipping pre-seed real-DB test")

        try:
            from services.embedding_service import EmbeddingService
            EmbeddingService().generate_embedding("test")
        except Exception as e:
            pytest.skip(f"Embedding model unavailable: {e}")

        from services.subagent_processor import SubagentProcessor

        proc = SubagentProcessor(raw_input="research current weather patterns")
        tools = proc._preseeded_tools

        assert isinstance(tools, list)
        assert len(tools) <= 3

        # Each entry must be a dict with name/description/input_schema
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "input_schema" in t

        # "weather" is expected to surface for a weather-pattern query
        names = [t["name"] for t in tools]
        assert "weather" in names, (
            f"Expected 'weather' in pre-seeded tools for weather query, got: {names}"
        )


# ---------------------------------------------------------------------------
# B11. Blocklist enforced in pre-seed
# ---------------------------------------------------------------------------


class TestPreSeedBlocklist:
    def test_pre_seed_blocklist_enforced(self):
        """Blocklisted names returned by find_tools_query must be filtered before
        being resolved into ability schemas.

        NOTE: find_tools_query is patched here because we cannot force the
        embedding index to return blocklisted names — they are filtered from the
        DISCOVERABLE allowlist before the query, so the real DB never returns
        them. This patch operates at the import site in subagent_processor
        (abilities.find_tools.find_tools_query) and replaces only the return
        value, not production dispatch logic.
        """
        from unittest.mock import patch

        blocklisted_names = ["subagent", "save_graph", "detect_pattern", "weather"]
        with patch("abilities.find_tools.find_tools_query", return_value=blocklisted_names):
            from services.subagent_processor import SubagentProcessor
            proc = SubagentProcessor(raw_input="test blocklist enforcement")

        tool_names = [t["name"] for t in proc._preseeded_tools]
        assert "subagent" not in tool_names
        assert "save_graph" not in tool_names
        assert "detect_pattern" not in tool_names
        # Non-blocklisted "weather" may or may not be present (registry may
        # resolve it) — what matters is the blocked names are gone.


# ---------------------------------------------------------------------------
# B12. Blocklist constant membership
# ---------------------------------------------------------------------------


class TestBlocklistConstant:
    def test_blocklist_constant_includes_subagent_save_graph_detect_pattern(self):
        from services.subagent_processor import _BLOCKED

        assert isinstance(_BLOCKED, frozenset)
        assert "subagent" in _BLOCKED
        assert "save_graph" in _BLOCKED
        assert "detect_pattern" in _BLOCKED


# ---------------------------------------------------------------------------
# B13. Class constants
# ---------------------------------------------------------------------------


class TestProcessorClassConstants:
    def test_processor_class_constants(self):
        from services.subagent_processor import SubagentProcessor, _BLOCKED

        assert SubagentProcessor.CHANNEL == "subagent"
        assert SubagentProcessor.ROLE == "subagent"
        assert SubagentProcessor.MAX_ITERATIONS == 50
        assert SubagentProcessor.MAX_TIMEOUT == 7200
        assert SubagentProcessor.ALWAYS_AVAILABLE == ["find_tools"]

        discoverable = SubagentProcessor.DISCOVERABLE
        expected_tools = [
            "browser", "code_eval", "document", "list", "memory", "news",
            "programming_docs_search", "read", "review_tool_calls",
            "schedule", "search", "weather",
        ]
        for name in expected_tools:
            assert name in discoverable, f"'{name}' missing from DISCOVERABLE"

        # No blocklisted names must appear in DISCOVERABLE
        for blocked in _BLOCKED:
            assert blocked not in discoverable, (
                f"Blocklisted '{blocked}' found in DISCOVERABLE"
            )
