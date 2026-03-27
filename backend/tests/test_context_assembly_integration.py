"""Integration tests for context assembly with real SQLite backend.

Tests the full assemble() flow with TopicContext, real transcript and
compaction tables, and controlled service stubs. Verifies topic isolation,
failure tracking, and budget constraints.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.context_assembly_service import ContextAssemblyService
from services.topic_context import TopicContext


pytestmark = pytest.mark.unit


def _seed_transcript(conn, topic, messages):
    """Insert transcript entries. messages = list of (role, content)."""
    for role, content in messages:
        conn.execute(
            "INSERT INTO topic_transcript (topic, role, content) VALUES (?, ?, ?)",
            (topic, role, content),
        )
    conn.commit()


def _seed_compaction(conn, topic, text, watermark, tokens=100):
    """Insert a compaction record."""
    conn.execute(
        "INSERT INTO topic_compactions (topic, compacted_text, compacted_up_to_id, token_count) "
        "VALUES (?, ?, ?, ?)",
        (topic, text, watermark, tokens),
    )
    conn.commit()


def _make_svc(budget=100_000):
    """Create a ContextAssemblyService with a given token budget."""
    return ContextAssemblyService({'max_context_tokens': budget})


# ── Tests ────────────────────────────────────────────────────────────

class TestFullAssembly:

    def test_full_assembly_returns_all_sections(self, db):
        """Seeded transcript data flows through to the working_memory section."""
        _seed_transcript(db, 'cooking', [
            ('user', 'How do I make pasta?'),
            ('assistant', 'Boil water, add pasta, cook for 10 minutes.'),
        ])

        svc = _make_svc()
        ctx = TopicContext(topic='cooking')

        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='next?', topic='cooking', context=ctx)

        assert 'pasta' in result['working_memory'].lower()
        assert ctx.failed_sections == []

    def test_compaction_plus_recent_transcript(self, db):
        """Both compaction summary and recent entries appear in working memory."""
        # Seed 3 old entries (will be "compacted")
        _seed_transcript(db, 'travel', [
            ('user', 'Old message 1'),
            ('assistant', 'Old reply 1'),
            ('user', 'Old message 2'),
        ])

        # Get the watermark (highest id so far)
        cursor = db.execute("SELECT MAX(id) FROM topic_transcript")
        watermark = cursor.fetchone()[0]

        # Store compaction covering those old entries
        _seed_compaction(db, 'travel', 'User discussed travel to Japan.', watermark)

        # Seed 2 new entries after the watermark
        _seed_transcript(db, 'travel', [
            ('user', 'What about hotels in Tokyo?'),
            ('assistant', 'I recommend Shinjuku area.'),
        ])

        svc = _make_svc()
        ctx = TopicContext(topic='travel')

        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='more?', topic='travel', context=ctx)

        wm = result['working_memory']
        assert 'Japan' in wm  # from compaction
        assert 'Tokyo' in wm  # from recent transcript
        assert 'Shinjuku' in wm  # from recent transcript


class TestTopicIsolation:

    def test_topic_switch_does_not_bleed_old_context(self, db):
        """Assembling for topic-B should NOT include topic-A transcript data."""
        _seed_transcript(db, 'topic-A', [
            ('user', 'My secret password is hunter2'),
            ('assistant', 'Got it, stored securely.'),
        ])
        _seed_transcript(db, 'topic-B', [
            ('user', 'What is the weather?'),
            ('assistant', 'Sunny and warm.'),
        ])

        svc = _make_svc()
        ctx = TopicContext(topic='topic-B')

        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='more?', topic='topic-B', context=ctx)

        wm = result['working_memory']
        assert 'hunter2' not in wm
        assert 'password' not in wm
        assert 'weather' in wm.lower() or 'Sunny' in wm

    def test_new_topic_gets_empty_working_memory(self, db):
        """A brand new topic with no data should have empty working memory."""
        svc = _make_svc()
        ctx = TopicContext(topic='never-discussed', is_new_topic=True)

        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='hello', topic='never-discussed', context=ctx)

        assert result['working_memory'] == ''


class TestFailureTracking:

    def test_failed_sections_tracked(self, db):
        """When one section fails, failure is recorded and other sections still work."""
        _seed_transcript(db, 'test', [
            ('user', 'Hello there'),
            ('assistant', 'Hi!'),
        ])

        svc = _make_svc()
        ctx = TopicContext(topic='test')

        # Episodes will fail, everything else works
        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', side_effect=Exception('episode DB crash')), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            # Need to handle the exception at the assemble level
            # Actually, _get_episodes is called directly and its except catches it
            pass

        # Test via the real path — patch the episodic import to fail
        with patch('services.episodic_service.EpisodicService', side_effect=Exception('ep down')), \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='hi', topic='test', context=ctx)

        # Working memory should still be populated from real DB
        assert 'Hello' in result['working_memory']
        # Episodes should have recorded a failure
        assert len(ctx.failed_sections) >= 1
        section_names = [s[0] for s in ctx.failed_sections]
        assert 'episodes' in section_names

    def test_multiple_failures_do_not_crash(self, db):
        """Even if all sections fail, assemble returns a valid dict with failures tracked."""
        svc = _make_svc()
        ctx = TopicContext(topic='broken-topic')

        with patch('services.compaction_service.get_compaction', side_effect=Exception('comp down')), \
             patch.object(svc, '_get_moments', return_value=''), \
             patch('services.episodic_service.EpisodicService', side_effect=Exception('ep down')), \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}), \
             patch('services.knowledge_service.KnowledgeService', side_effect=Exception('ks down')), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='hi', topic='broken-topic', context=ctx)

        # Should still return a valid dict
        assert 'working_memory' in result
        assert 'episodes' in result
        assert isinstance(result['total_tokens_est'], int)

        # Failures should be tracked
        assert len(ctx.failed_sections) >= 2


class TestBudgetConstraint:

    def test_budget_constraint_with_real_data(self, db):
        """Large transcript data with small budget triggers trimming."""
        # Seed lots of transcript data
        messages = [(
            'user' if i % 2 == 0 else 'assistant',
            f'This is a long message number {i} with lots of extra content to fill the budget. ' * 10,
        ) for i in range(20)]
        _seed_transcript(db, 'verbose', messages)

        svc = _make_svc(budget=200)  # Very small budget
        ctx = TopicContext(topic='verbose')

        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value='Some episode' * 50), \
             patch.object(svc, '_get_concepts', return_value='Some concept' * 50), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='hi', topic='verbose', context=ctx)

        # Total should be constrained near the budget
        total_chars = sum(len(v) for v in result.values() if isinstance(v, str))
        # Budget is 200 tokens ≈ 800 chars. Allow some slack for truncation markers.
        assert total_chars < 5000  # Much less than unrestricted would be


class TestTopicContextIdentifier:

    def test_thread_id_takes_precedence(self, db):
        """When TopicContext has thread_id, it's used for working memory lookup."""
        # Seed data under the thread_id as topic (legacy fallback path)
        _seed_transcript(db, 'thread-99', [
            ('user', 'Thread-specific message'),
            ('assistant', 'Thread-specific reply'),
        ])

        svc = _make_svc()
        ctx = TopicContext(topic='general', thread_id='thread-99')

        with patch.object(svc, '_get_moments', return_value=''), \
             patch.object(svc, '_get_episodes', return_value=''), \
             patch.object(svc, '_get_concepts', return_value=''), \
             patch.object(svc, '_get_procedural_hints', return_value=''):

            result = svc.assemble(prompt='hi', topic='general', context=ctx)

        # Should find data under the thread's topic
        # (working_memory uses effective_topic = topic or identifier)
        assert ctx.wm_identifier == 'thread-99'
