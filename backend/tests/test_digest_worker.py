"""Tests for digest_worker — calculate_context_warmth, NLP signal patterns, ignore branch triage."""

import pytest
from unittest.mock import MagicMock, patch

from workers.digest_worker import calculate_context_warmth, _handle_ignore_branch, _is_innate_skill_only
from services.cognitive_triage_service import TriageResult
from services.mode_router_service import (
    GREETING_PATTERNS,
    INTERROGATIVE_WORDS,
    IMPLICIT_REFERENCE,
    POSITIVE_FEEDBACK,
    NEGATIVE_FEEDBACK,
)


pytestmark = pytest.mark.unit


# ── calculate_context_warmth ─────────────────────────────────────────

class TestCalculateContextWarmth:
    """
    warmth = (wm_score + world_score) / 2
    wm_score    = min(working_memory_len / 4, 1.0)
    world_score = 1.0 if world_state_nonempty else 0.0

    Gist score removed in Stream 1 (memory chunker killed).
    """

    def test_all_zeros_returns_zero(self):
        assert calculate_context_warmth(0, False) == 0.0

    def test_all_maxed_returns_one(self):
        result = calculate_context_warmth(8, True)
        assert result == pytest.approx(1.0)

    def test_wm_caps_at_one(self):
        # 8 turns → min(8/4, 1.0) = 1.0, world=False
        result = calculate_context_warmth(8, False)
        expected = (1.0 + 0.0) / 2
        assert result == pytest.approx(expected)

    def test_world_state_true_contributes_half(self):
        result = calculate_context_warmth(0, True)
        expected = (0.0 + 1.0) / 2
        assert result == pytest.approx(expected)

    def test_world_state_false_contributes_zero(self):
        result = calculate_context_warmth(0, False)
        assert result == 0.0

    def test_mixed_inputs(self):
        # wm=2 → 0.5, world=True → 1.0
        result = calculate_context_warmth(2, True)
        expected = (0.5 + 1.0) / 2
        assert result == pytest.approx(expected, abs=0.001)


# ── NLP signal patterns ──────────────────────────────────────────────

class TestNlpSignalPatterns:

    def test_greeting_match_on_hey(self):
        assert GREETING_PATTERNS.match("hey there") is not None

    def test_greeting_match_on_good_morning(self):
        assert GREETING_PATTERNS.match("good morning") is not None

    def test_greeting_no_match_on_normal_text(self):
        assert GREETING_PATTERNS.match("the weather is nice") is None

    def test_interrogative_match_on_what(self):
        assert INTERROGATIVE_WORDS.search("what is this") is not None

    def test_interrogative_no_match_on_plain_sentence(self):
        assert INTERROGATIVE_WORDS.search("the cat sat") is None

    def test_implicit_reference_match(self):
        assert IMPLICIT_REFERENCE.search("you remember that?") is not None

    def test_implicit_reference_no_match(self):
        assert IMPLICIT_REFERENCE.search("the sky is blue") is None

    def test_question_mark_detection(self):
        assert '?' in "What time is it?"
        assert '?' not in "Tell me the time"

    def test_token_count_via_split(self):
        tokens = "hello world foo".split()
        assert len(tokens) == 3

    def test_positive_feedback_match(self):
        assert POSITIVE_FEEDBACK.search("thanks a lot") is not None

    def test_negative_feedback_match(self):
        assert NEGATIVE_FEEDBACK.search("that's not what I meant") is not None

    def test_information_density_calculation(self):
        tokens = "the the the cat".split()
        unique = len(set(t.lower() for t in tokens))
        density = unique / max(len(tokens), 1)
        # 2 unique / 4 total = 0.5
        assert density == pytest.approx(0.5)


def _make_triage(mode):
    return TriageResult(
        branch='ignore', mode=mode, tools=[], skills=[],
        confidence_internal=1.0, confidence_tool_need=0.0,
        freshness_risk=0.0, decision_entropy=0.0,
        reasoning='test', triage_time_ms=0.0,
        fast_filtered=False, self_eval_override=False, self_eval_reason=None,
    )


class TestHandleIgnoreBranch:
    """
    _handle_ignore_branch must only fast-exit for CANCEL/IGNORE.
    Any other mode (e.g. RESPOND) must return None so callers route
    through generate_for_mode — preventing the NoneType crash that
    caused 'No response received' for ambiguous requests like 'Schedule it'.
    """

    def test_cancel_returns_empty_response(self):
        result = _handle_ignore_branch(
            _make_triage('CANCEL'), 'never mind', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is not None
        assert result['mode'] == 'CANCEL'
        assert result['response'] == ''

    def test_ignore_returns_empty_response(self):
        result = _handle_ignore_branch(
            _make_triage('IGNORE'), '', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is not None
        assert result['mode'] == 'IGNORE'
        assert result['response'] == ''

    def test_respond_mode_returns_none(self):
        """RESPOND must not be handled here — callers route to generate_for_mode."""
        result = _handle_ignore_branch(
            _make_triage('RESPOND'), 'Schedule it', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is None, (
            "RESPOND mode in ignore branch should return None so the dispatch "
            "condition (mode in CANCEL/IGNORE) prevents this path from being reached, "
            "not silently produce an empty response."
        )

    def test_clarify_mode_returns_none(self):
        result = _handle_ignore_branch(
            _make_triage('CLARIFY'), 'Schedule it', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is None


# ── _is_innate_skill_only / contextual_skills dispatch ───────────

def _make_act_triage(skills):
    return TriageResult(
        branch='act', mode='ACT', tools=[], skills=skills,
        confidence_internal=0.9, confidence_tool_need=0.1,
        freshness_risk=0.0, decision_entropy=0.0,
        reasoning='test', triage_time_ms=0.0,
        fast_filtered=False, self_eval_override=False, self_eval_reason=None,
    )


class TestInnateSkillOnly:
    """
    _is_innate_skill_only gates _handle_innate_skill_dispatch.
    Ensures only contextual skills (not primitives) trigger direct dispatch,
    and that the dispatch path passes only contextual_skills to the LLM
    (preventing introspect/recall from crowding out the intended skill).
    """

    def test_schedule_skill_is_innate_only(self):
        """schedule in skills + no tools → direct dispatch path."""
        triage = _make_act_triage(['recall', 'memorize', 'introspect', 'schedule'])
        assert _is_innate_skill_only(triage) is True

    def test_primitives_only_is_not_innate_only(self):
        """Only recall/memorize/introspect → not an innate-only dispatch (no contextual skill)."""
        triage = _make_act_triage(['recall', 'memorize', 'introspect'])
        assert _is_innate_skill_only(triage) is False

    def test_empty_skills_is_not_innate_only(self):
        triage = _make_act_triage([])
        assert _is_innate_skill_only(triage) is False

    def test_external_tool_present_is_not_innate_only(self):
        """External tools present → full ACT loop, not direct dispatch."""
        triage = _make_act_triage(['schedule'])
        triage.tools = ['duckduckgo_search']
        assert _is_innate_skill_only(triage) is False

    def test_list_skill_is_innate_only(self):
        triage = _make_act_triage(['recall', 'list'])
        assert _is_innate_skill_only(triage) is True


# ── Trait extraction in digest_worker ────────────────────────────────

MODULE = 'workers.digest_worker'

import contextlib


def _make_pipeline_patches(enqueue_side_effect=None):
    """
    Return a context manager that activates all mocks required to run
    ``digest_worker()`` through its full main pipeline without real I/O.

    The heavy dependencies (LLM, Redis, SQLite, classifiers, triage) are all
    replaced with lightweight ``MagicMock`` objects so that the test can reach
    Phase D — where ``enqueue_trait_extraction`` is called — without spawning
    network connections or requiring on-disk models.

    Args:
        enqueue_side_effect: Optional side-effect to assign to the
            ``enqueue_trait_extraction`` mock (e.g. an exception class to
            simulate a failure).

    Returns:
        contextlib.ExitStack: An active context manager whose ``__exit__``
            tears down all patches, plus a ``mock_enqueue`` attribute set
            once the stack is entered (see usage in tests).
    """

    @contextlib.contextmanager
    def _cm():
        # ── Redis / MemoryClientService ──────────────────────────────────
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = True

        # ── Thread service ───────────────────────────────────────────────
        mock_thread_res = MagicMock()
        mock_thread_res.thread_id = 'thread-123'
        mock_thread_svc = MagicMock()
        mock_thread_svc.resolve_thread.return_value = mock_thread_res

        # ── Working memory ───────────────────────────────────────────────
        mock_wm = MagicMock()
        mock_wm.get_recent_turns.return_value = []

        # ── Topic classifier ─────────────────────────────────────────────
        mock_topic_clf = MagicMock()
        mock_topic_clf.classify.return_value = {
            'topic': 'health',
            'confidence': 0.8,
            'classification_time': 0.01,
            'message_embedding': None,
            'just_reset_from_silence': False,
            'is_new_topic': False,
        }

        # ── Intent classifier ────────────────────────────────────────────
        mock_intent_clf = MagicMock()
        mock_intent_clf.classify.return_value = {
            'intent_type': 'conversational',
            'complexity': 'low',
            'confidence': 0.9,
            'is_cancel': False,
            'is_self_resolved': False,
        }

        # ── Metrics ──────────────────────────────────────────────────────
        mock_metrics = MagicMock()
        mock_metrics.start_trace.return_value = 'trace-123'

        # ── Thread conversation service ──────────────────────────────────
        mock_thread_conv = MagicMock()
        mock_thread_conv.add_exchange.return_value = 'exchange-123'

        # ── Session service ──────────────────────────────────────────────
        mock_session = MagicMock()
        mock_session.is_returning_from_silence.return_value = 0
        mock_session.check_topic_switch.return_value = False
        mock_session.topic_exchange_count = 0

        # ── Cognitive triage result (RESPOND branch, no tool dispatch) ───
        mock_triage_result = MagicMock()
        mock_triage_result.branch = 'respond'
        mock_triage_result.mode = 'RESPOND'
        mock_triage_result.tools = []
        mock_triage_result.skills = []
        mock_triage_result.confidence_internal = 0.9
        mock_triage_result.confidence_tool_need = 0.1
        mock_triage_result.triage_time_ms = 5.0
        mock_triage_result.effort_estimate = 'low'
        mock_triage_result.self_eval_override = False
        mock_triage_result.fast_filtered = False

        mock_triage_svc = MagicMock()
        mock_triage_svc.triage.return_value = mock_triage_result

        mock_tool_profile = MagicMock()
        mock_tool_profile.get_triage_summaries.return_value = {}

        # ── Synthesised LLM response ─────────────────────────────────────
        fake_response_data = {
            'response': 'I am here to help!',
            'mode': 'RESPOND',
            'confidence': 0.9,
            'generation_time': 0.1,
        }
        fake_routing_result = {
            'mode': 'RESPOND',
            'router_confidence': 0.9,
            'routing_source': 'triage',
            'routing_time_ms': 5.0,
        }

        # ── Minimal frontal-cortex config ────────────────────────────────
        minimal_configs = {
            'cortex': {
                'config': {'max_working_memory_turns': 10},
                'prompt_map': {
                    'RESPOND': 'test prompt',
                    'CLARIFY': 'test prompt',
                    'ACT': 'test prompt',
                },
            }
        }

        # ── enqueue_trait_extraction (the subject under test) ────────────
        mock_enqueue = MagicMock()
        if enqueue_side_effect is not None:
            mock_enqueue.side_effect = enqueue_side_effect

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f'{MODULE}.load_configs',
                                      return_value=minimal_configs))
            stack.enter_context(patch(f'{MODULE}.get_thread_service',
                                      return_value=mock_thread_svc))
            stack.enter_context(patch(
                'services.memory_client.MemoryClientService.create_connection',
                return_value=mock_redis,
            ))
            stack.enter_context(patch(f'{MODULE}.ThreadConversationService',
                                      return_value=mock_thread_conv))
            stack.enter_context(patch(f'{MODULE}.RecentTopicService',
                                      return_value=MagicMock(
                                          get_recent_topic=MagicMock(return_value=None)
                                      )))
            stack.enter_context(patch(f'{MODULE}.WorldStateService',
                                      return_value=MagicMock(
                                          get_world_state=MagicMock(return_value='')
                                      )))
            stack.enter_context(patch(f'{MODULE}.WorkingMemoryService',
                                      return_value=mock_wm))
            stack.enter_context(patch(f'{MODULE}.EventBusService',
                                      return_value=MagicMock()))
            stack.enter_context(patch(f'{MODULE}.MetricsService',
                                      return_value=mock_metrics))
            stack.enter_context(patch(f'{MODULE}.get_mode_router',
                                      return_value=MagicMock()))
            stack.enter_context(patch(f'{MODULE}.get_existing_topics_from_db',
                                      return_value=[]))
            stack.enter_context(patch(f'{MODULE}.get_topic_classifier',
                                      return_value=mock_topic_clf))
            stack.enter_context(patch(f'{MODULE}.get_session_service',
                                      return_value=mock_session))
            stack.enter_context(patch(f'{MODULE}.get_intent_classifier',
                                      return_value=mock_intent_clf))
            stack.enter_context(patch(f'{MODULE}.compute_nlp_signals',
                                      return_value={}))
            stack.enter_context(patch(f'{MODULE}._run_iip_hook'))
            stack.enter_context(patch(f'{MODULE}._run_belief_correction_hook'))
            stack.enter_context(patch(f'{MODULE}._detect_fork_response'))
            stack.enter_context(patch(f'{MODULE}._store_adaptive_signals'))
            stack.enter_context(patch(f'{MODULE}._check_active_tool_work',
                                      return_value=None))
            stack.enter_context(patch(
                'services.cognitive_triage_service.CognitiveTriageService',
                return_value=mock_triage_svc,
            ))
            stack.enter_context(patch(
                'services.tool_profile_service.ToolProfileService',
                return_value=mock_tool_profile,
            ))
            stack.enter_context(patch(f'{MODULE}.route_and_generate',
                                      return_value=(fake_response_data,
                                                    fake_routing_result)))
            stack.enter_context(patch(f'{MODULE}.enqueue_trait_extraction',
                                      mock_enqueue))
            yield mock_enqueue

    return _cm()


class TestTraitExtractionEnqueue:
    """
    Verify that enqueue_trait_extraction is called for regular chat messages
    flowing through digest_worker, but NOT duplicated for early-return paths.
    """

    def test_tool_result_path_does_not_call_trait_extraction_from_main_path(self):
        """Messages with type=tool_result use the early return, not the main path."""
        with patch(f'{MODULE}._handle_tool_result', return_value='ok') as mock_handler, \
             patch(f'{MODULE}.enqueue_trait_extraction') as mock_enqueue:
            from workers.digest_worker import digest_worker
            result = digest_worker('tool output here', {'type': 'tool_result'})
            mock_handler.assert_called_once()
            mock_enqueue.assert_not_called()

    def test_proactive_drift_path_does_not_call_trait_extraction_from_main_path(self):
        """Messages with type=proactive_drift use the early return, not the main path."""
        with patch(f'{MODULE}._handle_proactive_drift', return_value='ok') as mock_handler, \
             patch(f'{MODULE}.enqueue_trait_extraction') as mock_enqueue:
            from workers.digest_worker import digest_worker
            result = digest_worker('drift message', {'type': 'proactive_drift'})
            mock_handler.assert_called_once()
            mock_enqueue.assert_not_called()

    def test_cron_tool_path_does_not_call_trait_extraction_from_main_path(self):
        """Messages with cron_tool source use the early return, not the main path."""
        with patch(f'{MODULE}._handle_cron_tool_result', return_value='ok') as mock_handler, \
             patch(f'{MODULE}.enqueue_trait_extraction') as mock_enqueue:
            from workers.digest_worker import digest_worker
            result = digest_worker('cron result', {'source': 'cron_tool:weather'})
            mock_handler.assert_called_once()
            mock_enqueue.assert_not_called()

    def test_regular_chat_enqueues_trait_extraction(self):
        """
        A regular chat message flowing through the full pipeline must call
        ``enqueue_trait_extraction`` exactly once with the user message
        (truncated to 1000 chars), a ``source`` key prefixed with 'chat:',
        the resolved topic, and the correct ``thread_id``.

        This test mocks all heavy infrastructure (LLM, Redis, SQLite,
        classifiers, triage) so the assertion is a pure behavioural check
        that the call site in Phase D is reached and wired correctly.
        """
        text = 'My name is Alice and I work as a nurse.'

        with _make_pipeline_patches() as mock_enqueue:
            from workers.digest_worker import digest_worker
            result = digest_worker(text, {'source': 'web'})

        # The function must still return a valid result string
        assert isinstance(result, str)
        assert len(result) > 0

        # enqueue_trait_extraction must be called exactly once …
        mock_enqueue.assert_called_once()

        # … with the expected keyword arguments
        _, kwargs = mock_enqueue.call_args
        assert kwargs['prompt_message'] == text[:1000]
        assert kwargs['thread_id'] == 'thread-123'
        assert kwargs['metadata']['source'] == 'chat:web'
        # topic comes from the mocked classifier's return value
        assert kwargs['metadata']['topic'] == 'health'

    def test_trait_extraction_failure_does_not_propagate(self):
        """
        If ``enqueue_trait_extraction`` raises at runtime, ``digest_worker``
        must swallow the exception (try/except wrapper) and still return a
        valid response string — the user must never see a 500 because of a
        background trait-extraction failure.
        """
        text = 'Tell me about machine learning.'

        with _make_pipeline_patches(
            enqueue_side_effect=RuntimeError('simulated extraction failure')
        ) as mock_enqueue:
            from workers.digest_worker import digest_worker
            # Must not raise — the try/except in Phase D must absorb the error
            result = digest_worker(text, {'source': 'web'})

        # enqueue was called (and raised), yet a valid result was returned
        mock_enqueue.assert_called_once()
        assert isinstance(result, str)
        assert len(result) > 0
