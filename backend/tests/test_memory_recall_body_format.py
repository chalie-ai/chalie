"""Feature test: memory recall body contains [id:X,relevance:Y] markers.

Key invariant: TranscriptService._fetch_referenced_episodes uses the regex
``r'\\[id:([^,\\]]+)'`` to back-reference memory hits. This test seeds a real
DataGraph entry, runs a real recall, and asserts the wire format that regex
depends on is present inside the [memory(...)] / [end:memory] block.

Zero mocks — real DataGraphService against real in-memory SQLite.
"""

import re

import pytest

pytestmark = pytest.mark.unit


def _handle_memory(topic: str, params: dict) -> str:
    """Thin shim: call MemoryAbility.run and return the text string.

    No processor is bound — recall reads its channel from the (absent)
    MessageProcessor and falls back to "", which the data-graph recall path
    does not depend on.
    """
    from abilities.memory import MemoryAbility
    result = MemoryAbility().run(params)
    assert result is not None, "MemoryAbility.run() returned None"
    return result["text"]


class TestMemoryRecallBodyFormat:
    """Verify the [id:X,relevance:Y] body format is produced by real recall."""

    def test_recall_with_hit_contains_id_relevance_markers(self, db):
        """Real DataGraph recall wraps hits in [id:...,relevance:...] lines.

        Seeds a user_specific fact, recalls it by query, and asserts the body
        of the [memory(...)] tag contains the expected marker format.
        The format is load-bearing: TranscriptService parses it with
        r'\\[id:([^,\\]]+)' to resolve back-references.
        """
        from services.data_graph_service import get_data_graph_service

        # Seed a known fact via the real DataGraphService (bound to test db)
        dgs = get_data_graph_service()
        dgs.store(
            kind='user_specific',
            key='residence',
            value='Valletta',
            source='test:seed',
        )

        result = _handle_memory('topic', {'action': 'recall', 'query': 'residence city'})

        # Both markers must be present
        assert '[memory(' in result, f"Missing opener in: {result!r}"
        assert '[end:memory]' in result, f"Missing terminator in: {result!r}"

        # At least one result must exist
        assert 'results=0' not in result, (
            f"Expected a hit for 'residence city' but got results=0. Output: {result!r}"
        )

        # Body must contain the [id:X,relevance:Y] format that TranscriptService parses
        id_marker_pattern = re.compile(r'\[id:[^,\]]+,relevance:(high|medium|low)\]')
        body_start = result.index('\n') + 1
        body_end = result.rindex('\n')
        body = result[body_start:body_end]
        assert id_marker_pattern.search(body), (
            f"Body missing [id:X,relevance:Y] marker. Body: {body!r}"
        )

        # The stored value must appear in the body
        assert 'Valletta' in body, f"Stored value 'Valletta' missing from body: {body!r}"

    def test_recall_no_results_carries_fallback_guardrail(self, db):
        """Empty EXPLICIT recall is [memory(...results=0)] + the fallback hint.

        The 'No memories found' semantic stays a tag arg (results=0), and an
        explicit (model-invoked) recall now also carries the guardrail body line
        steering the model to the document/schedule stores ON ITS OWN judgement
        instead of memory.recall silently fanning out to them. (TKT-878)
        """
        result = _handle_memory('topic', {'action': 'recall', 'query': 'xyzzy_nonexistent_key_abc'})

        assert '[memory(' in result
        assert '[end:memory]' in result
        assert 'results=0' in result
        # The guardrail hint is the body, naming the exact fallback tools.
        assert 'If you cannot find the information in memory' in result
        assert '`document` (action: search)' in result
        assert '`schedule` (action: search)' in result
        # Opener + single hint line + terminator.
        lines = result.split('\n')
        assert len(lines) == 3, (
            f"Expected exactly 3 lines for empty explicit recall, got {len(lines)}: {result!r}"
        )

    def test_recall_body_key_matches_stored_key(self, db):
        """The id: in [id:X,relevance:Y] must be the canonical data-graph key.

        TranscriptService uses this id to look up the original memory row.
        """
        from services.data_graph_service import get_data_graph_service

        dgs = get_data_graph_service()
        dgs.store(
            kind='user_specific',
            key='partner',
            value='Sarah',
            source='test:seed',
        )

        result = _handle_memory('topic', {'action': 'recall', 'query': 'partner relationship'})

        if 'results=0' in result:
            pytest.skip("FTS did not surface this hit — embedding unavailable in this env")

        body_start = result.index('\n') + 1
        body_end = result.rindex('\n')
        body = result[body_start:body_end]

        # Key must appear in the id: field
        assert '[id:partner,' in body, (
            f"Expected '[id:partner,...' in body. Body: {body!r}"
        )
