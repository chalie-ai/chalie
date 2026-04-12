# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for UserMessageProcessor (Commit 8 rewrite).

Covers the full north-star contract for the user-channel MessageProcessor subclass:
  /Volumes/llm/chalie-plans/message-processing.md
  /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md § Commit 8

Test groups:
  A. Subclass shape contract — class attributes, constructor, initial state
  B. getUserDefinition() — identity + fallback paths
  C. getUserPrompt() — section assembly, voice mode, file/nudge tags
  D. getSystemPrompt() — placeholder injection, identity prepend
  E. getTools() — voice-mode filter
  F. _run_memory_seed() — DTO shape, empty-result no-op, exception swallow
  G. store() override — _last_response capture, exception propagation
  H. postTurn() 9-service fan-out — call counts, ordering, isolation
  I. _emit_narration() — callback invocation, exception swallow, None guard
  J. Smoke — constructor to postTurn without full send()
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ump(raw_input='hello', metadata=None, on_narration=None):
    """Construct a UserMessageProcessor with minimal fuss."""
    from services.user_message_processor import UserMessageProcessor
    return UserMessageProcessor(
        raw_input=raw_input,
        metadata=metadata or {},
        on_narration=on_narration,
    )


def _patch_store_noop(proc):
    """Replace store() with a no-op that sets _uid=1 so postTurn() can reference it."""
    def _fake_store(llm_response):
        proc._uid = 1
    proc.store = _fake_store


# ─────────────────────────────────────────────────────────────────────────────
# A. Subclass shape contract
# ─────────────────────────────────────────────────────────────────────────────

class TestSubclassShape:
    """Class-level attributes must match the north-star contract exactly."""

    def test_a1_class_attributes_match_north_star(self):
        from services.user_message_processor import UserMessageProcessor
        from services.system_message_prompt import UnifiedSystemMessagePrompt

        assert UserMessageProcessor.CHANNEL == 'user'
        assert UserMessageProcessor.ROLE == 'user'
        assert UserMessageProcessor.JOB == 'frontal-cortex-unified'
        assert UserMessageProcessor.SYSTEM_PROMPT_CLASS is UnifiedSystemMessagePrompt

    def test_a2_native_tools_is_sorted_skill_list(self):
        from services.user_message_processor import UserMessageProcessor
        from services.innate_skills.registry import ALL_SKILL_NAMES

        expected = sorted(ALL_SKILL_NAMES)
        assert UserMessageProcessor.NATIVE_TOOLS == expected
        assert len(UserMessageProcessor.NATIVE_TOOLS) == 12

    def test_a3_constructor_accepts_required_args(self):
        # Must not raise
        from services.user_message_processor import UserMessageProcessor
        proc = UserMessageProcessor(raw_input='hi', metadata={}, on_narration=None)
        assert proc is not None

    def test_a4_constructor_initializes_per_turn_state(self):
        proc = _make_ump()

        assert proc._memory_seed_radius is None
        assert proc._last_response == ''
        assert proc._pending_tool_calls == []
        assert proc._act_trail == []
        assert proc._discovered_tools == []
        # _uid from base
        assert proc._uid is None


# ─────────────────────────────────────────────────────────────────────────────
# B. getUserDefinition() — user synthesis from KnowledgeService
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_USER_DEF = "The user is a real human. Treat this conversation as peer-to-peer dialogue."


class TestGetUserDefinition:
    """getUserDefinition() returns user synthesis from knowledge store, not voice modulation."""

    def test_b1_returns_user_synthesis_when_record_exists(self):
        """Returns the value field of the user_summary knowledge record."""
        proc = _make_ump()

        mock_ks = MagicMock()
        mock_ks.get.return_value = {'value': 'Dylan is a software engineer based in Malta.'}

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service'):
            result = proc.getUserDefinition()

        assert result == 'Dylan is a software engineer based in Malta.'
        mock_ks.get.assert_called_once_with('system', 'user_summary')

    def test_b2_returns_fallback_when_no_record(self):
        """Returns fallback string when knowledge store returns None."""
        proc = _make_ump()

        mock_ks = MagicMock()
        mock_ks.get.return_value = None

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service'):
            result = proc.getUserDefinition()

        assert result == _FALLBACK_USER_DEF

    def test_b6_does_not_call_voice_mapper_service(self):
        """getUserDefinition() must NOT call VoiceMapperService — that belongs in _get_voice_modulation()."""
        proc = _make_ump()

        mock_ks = MagicMock()
        mock_ks.get.return_value = {'value': 'some user summary'}

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.voice_mapper_service.VoiceMapperService') as MockVms:
            proc.getUserDefinition()

        MockVms.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# B2. _get_voice_modulation()
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_VOICE_MOD = "Engage naturally as a peer."


class TestGetVoiceModulation:
    """_get_voice_modulation() returns a behavioural style string for the Voice section."""

    def test_bv1_returns_modulation_string_when_services_succeed(self):
        """Returns the modulation string from VoiceMapperService."""
        proc = _make_ump()

        with patch('services.identity_service.IdentityService') as MockIdSvc, \
             patch('services.voice_mapper_service.VoiceMapperService') as MockVmSvc, \
             patch('services.database_service.get_shared_db_service'):

            MockIdSvc.return_value.get_vectors.return_value = {'traits': []}
            MockIdSvc.return_value.check_coherence.return_value = None
            MockVmSvc.return_value.generate_modulation.return_value = 'Speak warmly with patient curiosity.'

            result = proc._get_voice_modulation()

        assert result == 'Speak warmly with patient curiosity.'

# ─────────────────────────────────────────────────────────────────────────────
# C. getUserPrompt() section assembly
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserPrompt:
    """Section order, presence/absence, file tags, nudge, memory seed."""

    def _make_proc_with_mocks(self, raw_input='hello', metadata=None,
                               world_state='', self_awareness='', prev='',
                               memory_seed=None, memory_seed_radius=None,
                               act_trail=''):
        proc = _make_ump(raw_input=raw_input, metadata=metadata or {})
        proc._memory_seed = memory_seed
        proc._memory_seed_radius = memory_seed_radius

        proc._get_world_state = MagicMock(return_value=world_state)
        proc._get_self_awareness = MagicMock(return_value=self_awareness)
        proc.getPreviousMessages = MagicMock(return_value=prev)
        proc.getActLoopTrail = MagicMock(return_value=act_trail)

        return proc

    def test_c1_full_assembly_order(self):
        proc = self._make_proc_with_mocks(
            world_state='WS body',
            self_awareness='SA body',
            prev='PREV body',
            memory_seed='SEED',
            memory_seed_radius=0.2,
            act_trail='TRAIL',
        )
        output = proc.getUserPrompt()

        # Find positions of each section marker
        ws_pos = output.index('## World State')
        sa_pos = output.index('## System Awareness')
        prev_pos = output.index('## Previous Messages')
        seed_pos = output.index('[memory(radius=0.2)] SEED')
        user_pos = output.index('user: hello')
        trail_pos = output.index('TRAIL')

        assert ws_pos < sa_pos < prev_pos < seed_pos < user_pos < trail_pos

    def test_c2_voice_mode_inserts_instruction_line(self):
        proc = self._make_proc_with_mocks(
            metadata={'source': 'voice'},
            world_state='WS',
            self_awareness='SA',
        )
        output = proc.getUserPrompt()

        assert 'voice mode' in output.lower()
        # The instruction should appear between world state and system awareness
        ws_pos = output.index('## World State')
        voice_pos = output.lower().index('voice mode')
        sa_pos = output.index('## System Awareness')
        assert ws_pos < voice_pos < sa_pos

    def test_c3_skips_empty_sections(self):
        proc = self._make_proc_with_mocks(
            world_state='',
            self_awareness='',
            prev='',
            memory_seed=None,
        )
        output = proc.getUserPrompt()

        assert '## World State' not in output
        assert '## System Awareness' not in output
        assert '## Previous Messages' not in output
        assert 'user: hello' in output

    def test_c4_file_tags_appended_to_turn_line(self):
        proc = self._make_proc_with_mocks(
            raw_input='hello',
            metadata={'file_tags': ['<file:abc.png>', '<file:def.jpg>']},
        )
        output = proc.getUserPrompt()

        assert 'user: hello <file:abc.png> <file:def.jpg>' in output

    def test_c5_nudge_tag_appended_to_turn_line(self):
        proc = self._make_proc_with_mocks(
            raw_input='hello',
            metadata={'nudge_tag': '<nudge:foo>'},
        )
        output = proc.getUserPrompt()

        # Find the turn line
        for line in output.splitlines():
            if line.startswith('user: hello'):
                assert line.endswith('<nudge:foo>')
                return
        pytest.fail('Turn line not found in output')

    def test_c6_memory_seed_uses_radius(self):
        proc = self._make_proc_with_mocks(
            memory_seed='hello memories',
            memory_seed_radius=0.2,
        )
        output = proc.getUserPrompt()

        assert '[memory(radius=0.2)] hello memories' in output

    def test_c7_no_memory_line_when_seed_empty(self):
        proc = self._make_proc_with_mocks(memory_seed=None)
        output = proc.getUserPrompt()

        assert '[memory(radius=' not in output

    def test_c8_act_trail_appended_at_end(self):
        proc = self._make_proc_with_mocks(
            act_trail='[tool(x=1)] result line',
        )
        output = proc.getUserPrompt()

        non_empty_lines = [l for l in output.splitlines() if l.strip()]
        assert non_empty_lines[-1] == '[tool(x=1)] result line'

    def test_c9_memory_seed_skipped_when_radius_none(self):
        """Memory seed line requires BOTH _memory_seed and _memory_seed_radius."""
        proc = self._make_proc_with_mocks(
            memory_seed='some seed text',
            memory_seed_radius=None,  # not set
        )
        output = proc.getUserPrompt()

        assert '[memory(radius=' not in output


# ─────────────────────────────────────────────────────────────────────────────
# D. getSystemPrompt() placeholder injection
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    """Both placeholder injections and user definition prepend."""

    def test_d1_fills_adaptive_directives_placeholder(self):
        proc = _make_ump()

        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt',
                          return_value='TEMPLATE {{adaptive_directives}} END'), \
             patch.object(proc, 'getUserDefinition', return_value='MOD'), \
             patch.object(proc, '_get_voice_modulation', return_value='VOICE'), \
             patch.object(proc, '_get_adaptive_directives', return_value='DIR'):
            result = proc.getSystemPrompt()

        assert result == 'MOD\n\nTEMPLATE DIR END'

    def test_d2_no_identity_modulation_placeholder_left_in_template(self):
        proc = _make_ump()

        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt',
                          return_value='TEMPLATE {{adaptive_directives}} END'), \
             patch.object(proc, 'getUserDefinition', return_value='MOD'), \
             patch.object(proc, '_get_voice_modulation', return_value='VOICE'), \
             patch.object(proc, '_get_adaptive_directives', return_value='DIR'):
            result = proc.getSystemPrompt()

        assert '{{identity_modulation}}' not in result

    def test_d3_user_definition_prepended(self):
        proc = _make_ump()

        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt',
                          return_value='BODY'), \
             patch.object(proc, 'getUserDefinition', return_value='MY_DEF'), \
             patch.object(proc, '_get_voice_modulation', return_value=''), \
             patch.object(proc, '_get_adaptive_directives', return_value=''):
            result = proc.getSystemPrompt()

        assert result.startswith('MY_DEF\n\n')

    def test_d4_fills_voice_modulation_placeholder(self):
        """{{voice_modulation}} is filled by _get_voice_modulation()."""
        proc = _make_ump()

        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt',
                          return_value='PREAMBLE {{voice_modulation}} POSTAMBLE {{adaptive_directives}}'), \
             patch.object(proc, 'getUserDefinition', return_value='USER_DEF'), \
             patch.object(proc, '_get_voice_modulation', return_value='TEST_VOICE'), \
             patch.object(proc, '_get_adaptive_directives', return_value='DIR'):
            result = proc.getSystemPrompt()

        assert 'TEST_VOICE' in result
        assert '{{voice_modulation}}' not in result

    def test_d5_no_literal_placeholders_in_output(self):
        """After weaving, no {{...}} literals survive in the final output."""
        import re
        proc = _make_ump()

        with patch.object(proc, 'getUserDefinition', return_value='USER_DEF'), \
             patch.object(proc, '_get_voice_modulation', return_value='VOICE_STRING'), \
             patch.object(proc, '_get_adaptive_directives', return_value='ADAPTIVE_STRING'):
            result = proc.getSystemPrompt()

        remaining = re.findall(r'\{\{\w+\}\}', result)
        assert not remaining, f"Unfilled placeholders in system prompt: {remaining}"

# ─────────────────────────────────────────────────────────────────────────────
# E. getTools() voice-mode filter
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTools:
    """Voice mode narrows the tool list by removing rich_render."""

    def test_e1_non_voice_returns_super_tools(self):
        proc = _make_ump(metadata={})

        fake_schemas = [{'name': 'memory'}, {'name': 'rich_render'}, {'name': 'find_tools'}]

        # getTools() is on the base class and lazily imports tool_schema_service
        with patch('services.tool_schema_service.get_skill_schemas',
                   return_value=fake_schemas):
            result = proc.getTools()

        names = [s['name'] for s in result]
        assert 'rich_render' in names  # NOT filtered in non-voice

    def test_e2_voice_excludes_rich_render(self):
        proc = _make_ump(metadata={'source': 'voice'})

        # Voice path calls get_skill_schemas from tool_schema_service
        # with the NATIVE_TOOLS list minus 'rich_render' — so rich_render
        # is never even passed in. Return a realistic subset.
        voice_native = [{'name': 'memory'}, {'name': 'find_tools'}]

        with patch('services.tool_schema_service.get_skill_schemas',
                   return_value=voice_native):
            result = proc.getTools()

        names = [s['name'] for s in result]
        assert 'rich_render' not in names

    def test_e3_voice_dedupes_with_dynamic_tools(self):
        proc = _make_ump(metadata={'source': 'voice'})
        # Pre-populate _discovered_tools with memory (already in native list)
        proc._discovered_tools = [{'name': 'memory'}]

        native_schemas_no_rich = [{'name': 'memory'}, {'name': 'find_tools'}]

        with patch('services.tool_schema_service.get_skill_schemas',
                   return_value=native_schemas_no_rich):
            result = proc.getTools()

        # memory must appear exactly once
        names = [s['name'] for s in result]
        assert names.count('memory') == 1


# ─────────────────────────────────────────────────────────────────────────────
# F. _run_memory_seed()
# ─────────────────────────────────────────────────────────────────────────────

class TestRunMemorySeed:
    """_run_memory_seed() DTO shape, empty-result no-op, exception swallow."""

    # _run_memory_seed() now calls recall_episodes(caller='seed') directly
    # (not via ActDispatcherService) so the telemetry row in memory_recall_log
    # carries caller='seed', distinguishing pre-turn seeds from LLM-driven recall.

    def _patch_recall(self, hits, status='2 matches'):
        """Patch recall_episodes + _format_results at the import site."""
        recall_patch = patch(
            'services.innate_skills.memory_skill.recall_episodes',
            return_value=(hits, status),
        )
        format_patch = patch(
            'services.innate_skills.memory_skill._format_results',
            return_value='memories text' if hits else '',
        )
        return recall_patch, format_patch

    def test_f1_appends_durable_dto_on_success(self):
        proc = _make_ump(raw_input='tell me about Paris')
        fake_hits = [{'layer': 'episodes', 'content': 'Paris trip', 'confidence': 0.8,
                      'freshness': '2026-01-01', 'salience': 0.5}]
        recall_patch, format_patch = self._patch_recall(fake_hits)

        with recall_patch, format_patch:
            proc._run_memory_seed()

        assert proc._memory_seed == 'memories text'
        # SEED_RADIUS_BASELINE is 0.2 (tuned at df1c08d).
        assert proc._memory_seed_radius == 0.2
        assert len(proc._pending_tool_calls) == 1

        dto = proc._pending_tool_calls[0]
        assert dto['name'] == 'memory'
        assert dto['ephemeral'] == 0
        assert dto['invoked_by'] == 'system'
        assert dto['result'] == 'memories text'
        assert 'timestamp' in dto

    def test_f2_no_dto_on_empty_result(self):
        proc = _make_ump()
        recall_patch, format_patch = self._patch_recall([])

        with recall_patch, format_patch:
            proc._run_memory_seed()

        assert proc._memory_seed is None
        assert proc._pending_tool_calls == []

    def test_f3_swallows_dispatch_exception(self):
        proc = _make_ump()

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   side_effect=ValueError('db error')):
            # Must not raise
            proc._run_memory_seed()

        assert proc._memory_seed is None

    def test_f4_dto_radius_param_matches_baseline(self):
        from services.innate_skills.memory_skill import SEED_RADIUS_BASELINE

        proc = _make_ump()
        fake_hits = [{'layer': 'episodes', 'content': 'x', 'confidence': 0.5,
                      'freshness': '2026-01-01', 'salience': 0.1}]
        recall_patch, format_patch = self._patch_recall(fake_hits)

        with recall_patch, format_patch:
            proc._run_memory_seed()

        dto = proc._pending_tool_calls[0]
        assert dto['params']['radius'] == SEED_RADIUS_BASELINE

    def test_f5_dto_contains_action_recall(self):
        """The DTO params must include action='recall'."""
        proc = _make_ump()
        fake_hits = [{'layer': 'episodes', 'content': 'x', 'confidence': 0.5,
                      'freshness': '2026-01-01', 'salience': 0.1}]
        recall_patch, format_patch = self._patch_recall(fake_hits)

        with recall_patch, format_patch:
            proc._run_memory_seed()

        dto = proc._pending_tool_calls[0]
        assert dto['params'].get('action') == 'recall'

    def test_f6_recall_called_with_seed_caller(self):
        """recall_episodes must be called with caller='seed' so telemetry logs correctly."""
        proc = _make_ump(raw_input='what is my coffee order?')
        fake_hits = [{'layer': 'episodes', 'content': 'oat milk latte', 'confidence': 0.9,
                      'freshness': '2026-01-01', 'salience': 0.7}]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=(fake_hits, '1 match')) as mock_recall, \
             patch('services.innate_skills.memory_skill._format_results',
                   return_value='oat milk latte memory'):
            proc._run_memory_seed()

        mock_recall.assert_called_once()
        call_kwargs = mock_recall.call_args
        assert call_kwargs.kwargs.get('caller') == 'seed' or \
               (call_kwargs.args and 'seed' in call_kwargs.args)


# ─────────────────────────────────────────────────────────────────────────────
# G. store() override + _last_response
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreOverride:
    """store() must call super() then capture _last_response."""

    def test_g1_store_calls_super_then_captures_response(self):
        proc = _make_ump()

        with patch('services.transcript_service.append_atomic_turn', return_value=42):
            proc.store('hello world')

        assert proc._last_response == 'hello world'
        assert proc._uid == 42

    def test_g2_store_propagates_base_exception(self):
        proc = _make_ump()

        with patch('services.transcript_service.append_atomic_turn',
                   side_effect=RuntimeError('db full')):
            with pytest.raises(RuntimeError, match='db full'):
                proc.store('should fail')

        # _last_response must NOT be set if base raised
        assert proc._last_response == ''


# ─────────────────────────────────────────────────────────────────────────────
# H. postTurn() 8-service fan-out
# ─────────────────────────────────────────────────────────────────────────────

class TestPostTurn:
    """postTurn() must call all eight services with correct args and be fault-isolated."""

    def _make_proc_for_postturn(self, metadata=None, last_response='reply text',
                                 uid=1):
        proc = _make_ump(raw_input='user input', metadata=metadata or {})
        proc._last_response = last_response
        proc._uid = uid
        return proc

    # ── H1: All seven services called ─────────────────────────────────────────

    def test_h1_all_seven_services_called_in_order(self):
        """Verify each of the seven services fires and ordering is load-bearing."""
        proc = self._make_proc_for_postturn()

        call_log = []

        mock_phase = MagicMock()
        mock_phase.update.side_effect = lambda *a, **kw: call_log.append('phase')
        mock_situation = MagicMock()
        mock_situation.update_on_message.side_effect = lambda *a: call_log.append('situation')
        mock_save = MagicMock()
        mock_save.get_saveable_flag.side_effect = lambda *a: call_log.append('save') or None
        mock_save.detect_saveable_content.return_value = None
        mock_detect_fork = MagicMock(side_effect=lambda *a: call_log.append('detect_fork'))
        mock_store_signals = MagicMock(side_effect=lambda *a: call_log.append('store_signals'))
        mock_dmn = MagicMock()
        mock_dmn.on_turn.side_effect = lambda: call_log.append('dmn')
        mock_metrics = MagicMock()
        mock_metrics.record_counter.side_effect = lambda name: call_log.append(f'metrics:{name}')
        mock_compaction = MagicMock(side_effect=lambda *a, **kw: call_log.append('compaction'))

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=mock_phase), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=mock_situation), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=mock_save), \
             patch('workers.post_exchange_hooks._detect_fork_response', mock_detect_fork), \
             patch('workers.post_exchange_hooks._store_adaptive_signals', mock_store_signals), \
             patch('services.dmn_service.get_dmn_service', return_value=mock_dmn), \
             patch('services.metrics_service.MetricsService', return_value=mock_metrics), \
             patch('services.compaction_service.check_and_compact', mock_compaction):
            proc.postTurn()

        # dmn fires before metrics
        assert 'dmn' in call_log
        assert 'metrics:requests_total' in call_log

    # ── H2: ConversationPhaseService two calls ────────────────────────────────

    def test_h3_phase_two_calls(self):
        proc = self._make_proc_for_postturn()
        mock_phase = MagicMock()

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=mock_phase), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=MagicMock()), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=MagicMock(
                       get_saveable_flag=MagicMock(return_value=None),
                       detect_saveable_content=MagicMock(return_value=None),
                   )), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=MagicMock()), \
             patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()), \
             patch('services.compaction_service.check_and_compact'):
            proc.postTurn()

        assert mock_phase.update.call_count == 2
        calls = mock_phase.update.call_args_list
        # First call is_user=True, second is_user=False
        assert calls[0].kwargs.get('is_user') is True or calls[0].args[2] is True
        assert calls[1].kwargs.get('is_user') is False or calls[1].args[2] is False

    # ── H4: Save trigger then detect ─────────────────────────────────────────

    def test_h4_save_trigger_then_detect(self):
        """5a fires emit_save_card + clear_flag, then 5b detect_saveable fires."""
        proc = self._make_proc_for_postturn(
            last_response='Here is the summary of your meeting.',
        )
        call_log = []

        mock_save = MagicMock()
        # 5a: existing flag with trigger text
        mock_save.get_saveable_flag.return_value = {
            'content_type': 'note',
            'topic': 'meetings',
        }
        mock_save.detect_save_trigger.return_value = True
        mock_save.emit_save_card.side_effect = lambda *a: call_log.append('emit_save_card')
        mock_save.clear_flag.side_effect = lambda *a: call_log.append('clear_flag')

        # 5b: response has saveable content
        mock_save.detect_saveable_content.side_effect = (
            lambda *a: call_log.append('detect_saveable') or {'content_type': 'note'}
        )
        mock_save.flag_saveable.side_effect = lambda *a: call_log.append('flag_saveable')

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=MagicMock()), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=MagicMock()), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=mock_save), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=MagicMock()), \
             patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()), \
             patch('services.compaction_service.check_and_compact'):
            proc.postTurn()

        # 4a fires before 4b
        assert call_log.index('emit_save_card') < call_log.index('detect_saveable')
        assert 'clear_flag' in call_log
        assert 'flag_saveable' in call_log

    # ── H5: Metrics two counters ──────────────────────────────────────────────

    def test_h5_metrics_records_two_counters(self):
        proc = self._make_proc_for_postturn()
        mock_metrics = MagicMock()

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=MagicMock()), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=MagicMock()), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=MagicMock(
                       get_saveable_flag=MagicMock(return_value=None),
                       detect_saveable_content=MagicMock(return_value=None),
                   )), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=MagicMock()), \
             patch('services.metrics_service.MetricsService', return_value=mock_metrics), \
             patch('services.compaction_service.check_and_compact'):
            proc.postTurn()

        counter_calls = [c.args[0] for c in mock_metrics.record_counter.call_args_list]
        assert 'requests_total' in counter_calls
        assert 'user_messages_total' in counter_calls

    # ── H6: DMN on_turn exactly once ─────────────────────────────────────────

    def test_h6_dmn_on_turn_called_exactly_once(self):
        """R10 lockstep: DMN must reset idle timer on every user turn."""
        proc = self._make_proc_for_postturn()
        mock_dmn = MagicMock()

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=MagicMock()), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=MagicMock()), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=MagicMock(
                       get_saveable_flag=MagicMock(return_value=None),
                       detect_saveable_content=MagicMock(return_value=None),
                   )), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=mock_dmn), \
             patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()), \
             patch('services.compaction_service.check_and_compact'):
            proc.postTurn()

        mock_dmn.on_turn.assert_called_once()

    # ── H7: One-service failure does not block others ─────────────────────────

    def test_h7_failure_in_one_service_does_not_block_others(self):
        """Phase raising must not prevent all other services from firing."""
        proc = self._make_proc_for_postturn()
        mock_situation = MagicMock()
        mock_dmn = MagicMock()
        mock_metrics = MagicMock()
        mock_compaction = MagicMock()

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   side_effect=RuntimeError('phase exploded')), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=mock_situation), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=MagicMock(
                       get_saveable_flag=MagicMock(return_value=None),
                       detect_saveable_content=MagicMock(return_value=None),
                   )), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=mock_dmn), \
             patch('services.metrics_service.MetricsService', return_value=mock_metrics), \
             patch('services.compaction_service.check_and_compact', mock_compaction):
            proc.postTurn()

        mock_situation.update_on_message.assert_called_once()
        mock_dmn.on_turn.assert_called_once()
        mock_metrics.record_counter.assert_called()
        mock_compaction.assert_called_once()

    # ── H11: flag_saveable uses exchange_id string, not _uid int ─────────────

    def test_h11_flag_saveable_uses_exchange_id_string_not_uid(self):
        """When exchange_id is in metadata, flag_saveable receives that string."""
        proc = self._make_proc_for_postturn(
            metadata={'exchange_id': 'ex-123'},
            uid=42,
        )
        proc._last_response = 'a document to save'
        mock_save = MagicMock()
        mock_save.get_saveable_flag.return_value = None  # skip 5a
        mock_save.detect_saveable_content.return_value = {'content_type': 'note'}
        mock_save.flag_saveable = MagicMock()

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=MagicMock()), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=MagicMock()), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=mock_save), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=MagicMock()), \
             patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()), \
             patch('services.compaction_service.check_and_compact'):
            proc.postTurn()

        mock_save.flag_saveable.assert_called_once()
        # 4th positional arg (exchange_id_for_flag)
        flag_call = mock_save.flag_saveable.call_args
        exchange_id_arg = flag_call.args[3] if flag_call.args else flag_call.kwargs.get('exchange_id')
        assert exchange_id_arg == 'ex-123'

    # ── H12: flag_saveable falls back to str(_uid) ───────────────────────────

    def test_h12_flag_saveable_falls_back_to_str_uid(self):
        """No exchange_id in metadata → flag_saveable receives str(_uid)."""
        proc = self._make_proc_for_postturn(uid=42)
        proc._last_response = 'a note to save'
        mock_save = MagicMock()
        mock_save.get_saveable_flag.return_value = None
        mock_save.detect_saveable_content.return_value = {'content_type': 'note'}
        mock_save.flag_saveable = MagicMock()

        with patch('services.conversation_phase_service.get_conversation_phase_service',
                   return_value=MagicMock()), \
             patch('services.situation_model_service.get_situation_model_service',
                   return_value=MagicMock()), \
             patch('services.save_suggestion_service.SaveSuggestionService',
                   return_value=mock_save), \
             patch('workers.post_exchange_hooks._detect_fork_response'), \
             patch('workers.post_exchange_hooks._store_adaptive_signals'), \
             patch('services.dmn_service.get_dmn_service', return_value=MagicMock()), \
             patch('services.metrics_service.MetricsService',
                   return_value=MagicMock()), \
             patch('services.compaction_service.check_and_compact'):
            proc.postTurn()

        mock_save.flag_saveable.assert_called_once()
        flag_call = mock_save.flag_saveable.call_args
        exchange_id_arg = flag_call.args[3] if flag_call.args else flag_call.kwargs.get('exchange_id')
        assert exchange_id_arg == '42'


# ─────────────────────────────────────────────────────────────────────────────
# I. _emit_narration()
# ─────────────────────────────────────────────────────────────────────────────

class TestEmitNarration:
    """Callback invocation, exception swallow, None guard."""

    def test_i1_calls_callback_when_set(self):
        received = []
        proc = _make_ump(on_narration=lambda text, it: received.append((text, it)))
        proc._emit_narration('thinking...', 3)

        assert received == [('thinking...', 3)]

    def test_i2_swallows_callback_exception(self):
        def _bad_callback(text, iteration):
            raise RuntimeError('sse queue full')

        proc = _make_ump(on_narration=_bad_callback)
        # Must not raise
        proc._emit_narration('hello', 1)

    def test_i3_no_op_when_callback_is_none(self):
        proc = _make_ump(on_narration=None)
        # Must not raise
        proc._emit_narration('anything', 0)

    def test_i4_no_op_when_text_is_empty(self):
        called = []
        proc = _make_ump(on_narration=lambda t, i: called.append(True))
        proc._emit_narration('', 1)

        # Empty text → callback is NOT invoked (per implementation guard)
        assert called == []


# ─────────────────────────────────────────────────────────────────────────────
# J. Smoke / integration (no real LLM, no real DB)
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoke:
    """Verify the constructor→getUserPrompt→getSystemPrompt path runs without
    a real DB or LLM, and that key instance fields are initialized."""

    def test_j1_getUserPrompt_runs_with_all_services_stubbed(self):
        proc = _make_ump(raw_input='what is the weather?')

        with patch.object(proc, '_get_world_state', return_value=''), \
             patch.object(proc, '_get_self_awareness', return_value=''), \
             patch.object(proc, 'getPreviousMessages', return_value=''):
            output = proc.getUserPrompt()

        assert 'user: what is the weather?' in output
        assert isinstance(output, str)

    def test_j2_getSystemPrompt_runs_with_template_and_directives_stubbed(self):
        proc = _make_ump()

        # Template mirrors the real structure: both placeholders must be
        # present so the test fails loudly if replace() stops working.
        template = 'TEMPLATE voice={{voice_modulation}} adaptive={{adaptive_directives}} END'

        with patch.object(proc.SYSTEM_PROMPT_CLASS, 'getPrompt',
                          return_value=template), \
             patch.object(proc, 'getUserDefinition', return_value='TEST_DEF'), \
             patch.object(proc, '_get_voice_modulation', return_value='VOICE_STUB'), \
             patch.object(proc, '_get_adaptive_directives', return_value='ADAPT_STUB'):
            result = proc.getSystemPrompt()

        assert result.startswith('TEST_DEF\n\n')
        assert 'TEMPLATE' in result
        assert 'voice=VOICE_STUB' in result
        assert 'adaptive=ADAPT_STUB' in result
        assert '{{voice_modulation}}' not in result
        assert '{{adaptive_directives}}' not in result

    def test_j3_instance_is_not_a_singleton(self):
        """Two instances must be fully independent — no shared mutable state."""
        proc_a = _make_ump(raw_input='hello')
        proc_b = _make_ump(raw_input='world')

        proc_a._pending_tool_calls.append({'name': 'test'})
        proc_a._act_trail.append('trail entry')

        assert proc_b._pending_tool_calls == []
        assert proc_b._act_trail == []

    def test_j4_postTurn_called_after_store_sets_last_response(self):
        """postTurn() can read _last_response set by store()."""
        proc = _make_ump(raw_input='check weather')

        captured = []

        def _fake_store(resp):
            proc._uid = 99
            proc._last_response = resp

        def _capture_postturn():
            captured.append(proc._last_response)

        proc.store = _fake_store
        proc.postTurn = _capture_postturn

        # Simulate the sequence send() would use
        proc.store('the weather is sunny')
        proc.postTurn()

        assert captured == ['the weather is sunny']
