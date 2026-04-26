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
        from services.innate_skills.memory_skill import handle_memory

        # Seed a known fact via the real DataGraphService (bound to test db)
        dgs = get_data_graph_service()
        dgs.store(
            kind='user_specific',
            key='residence',
            value='Valletta',
            source='test:seed',
        )

        result = handle_memory('topic', {'action': 'recall', 'query': 'residence city'})

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

    def test_recall_no_results_has_results_zero_arg(self, db):
        """Empty recall produces [memory(query=..., results=0)]\\n[end:memory].

        Verifies the 'No memories found' semantic is now expressed as a tag
        arg rather than a body string, keeping the wire format consistent.
        """
        from services.innate_skills.memory_skill import handle_memory

        result = handle_memory('topic', {'action': 'recall', 'query': 'xyzzy_nonexistent_key_abc'})

        assert '[memory(' in result
        assert '[end:memory]' in result
        assert 'results=0' in result
        # No body between the markers (empty recall has no body line)
        lines = result.split('\n')
        assert len(lines) == 2, (
            f"Expected exactly 2 lines for empty recall, got {len(lines)}: {result!r}"
        )

    def test_recall_body_key_matches_stored_key(self, db):
        """The id: in [id:X,relevance:Y] must be the canonical data-graph key.

        TranscriptService uses this id to look up the original memory row.
        """
        from services.data_graph_service import get_data_graph_service
        from services.innate_skills.memory_skill import handle_memory

        dgs = get_data_graph_service()
        dgs.store(
            kind='user_specific',
            key='partner',
            value='Sarah',
            source='test:seed',
        )

        result = handle_memory('topic', {'action': 'recall', 'query': 'partner relationship'})

        if 'results=0' in result:
            pytest.skip("FTS did not surface this hit — embedding unavailable in this env")

        body_start = result.index('\n') + 1
        body_end = result.rindex('\n')
        body = result[body_start:body_end]

        # Key must appear in the id: field
        assert '[id:partner,' in body, (
            f"Expected '[id:partner,...' in body. Body: {body!r}"
        )
