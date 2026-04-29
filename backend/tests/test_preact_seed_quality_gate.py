"""Unit tests — pre_act() quality gate on relevance:high.

Verifies that UserMessageProcessor._memory_seed is only set when the recall
block contains at least one `relevance:high` result, and that
ToolRenderAndRecordService is called unconditionally.
"""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

_HIGH_BLOCK = (
    "[memory(query=foo,results=2)]\n"
    "[id:e1,relevance:high] body1\n"
    "[id:e2,relevance:medium] body2\n"
    "[end:memory]"
)

_MEDIUM_ONLY_BLOCK = (
    "[memory(query=foo,results=2)]\n"
    "[id:e1,relevance:medium] body1\n"
    "[id:e2,relevance:low] body2\n"
    "[end:memory]"
)

_EMPTY_BLOCK = "[memory(query=foo,results=0)]"

_ERROR_BLOCK = "[memory(query=foo,error=timeout)]"


def _make_proc(raw_input="oven reminder"):
    """Construct a UserMessageProcessor with _uid pre-set so pre_act runs fully."""
    from services.user_message_processor import UserMessageProcessor

    proc = UserMessageProcessor(raw_input=raw_input)
    proc._uid = "test-uid-001"
    return proc


def _run_pre_act_with_block(block_text, monkeypatch):
    """Patch AbilityRegistry and ToolRenderAndRecordService, run pre_act, return (proc, mock_trrs)."""
    mock_ability = MagicMock()
    mock_ability.execute.return_value = {'text': block_text}

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_ability

    mock_trrs_instance = MagicMock()
    mock_trrs_cls = MagicMock(return_value=mock_trrs_instance)

    # AbilityRegistry and ToolRenderAndRecordService are imported locally inside
    # pre_act() so we patch them at their source modules.
    with patch('abilities._registry.AbilityRegistry', mock_registry), \
         patch('services.tool_render_and_record_service.ToolRenderAndRecordService', mock_trrs_cls):
        proc = _make_proc()
        proc.pre_act()

    return proc, mock_trrs_cls, mock_trrs_instance


@pytest.mark.unit
class TestPreActSeedQualityGate:
    """pre_act() injects seed only when relevance:high is present in the block."""

    def test_high_relevance_result_injects_seed(self, monkeypatch):
        """Block with at least one relevance:high hit → _memory_seed is set to the full block."""
        proc, _, _ = _run_pre_act_with_block(_HIGH_BLOCK, monkeypatch)
        assert proc._memory_seed == _HIGH_BLOCK

    def test_medium_and_low_only_does_not_inject_seed(self, monkeypatch):
        """Block with only relevance:medium and relevance:low → _memory_seed remains None."""
        proc, _, _ = _run_pre_act_with_block(_MEDIUM_ONLY_BLOCK, monkeypatch)
        assert proc._memory_seed is None

    def test_empty_results_does_not_inject_seed(self, monkeypatch):
        """Block with results=0 in header → _memory_seed remains None (pre-existing behavior)."""
        proc, _, _ = _run_pre_act_with_block(_EMPTY_BLOCK, monkeypatch)
        assert proc._memory_seed is None

    def test_error_does_not_inject_seed(self, monkeypatch):
        """Block with error= in header → _memory_seed remains None (pre-existing behavior)."""
        proc, _, _ = _run_pre_act_with_block(_ERROR_BLOCK, monkeypatch)
        assert proc._memory_seed is None

    def test_recording_happens_regardless_of_gate(self, monkeypatch):
        """ToolRenderAndRecordService is called even when the seed is gated out."""
        proc, mock_trrs_cls, mock_trrs_instance = _run_pre_act_with_block(
            _MEDIUM_ONLY_BLOCK, monkeypatch
        )
        assert proc._memory_seed is None
        mock_trrs_cls.assert_called_once()
        mock_trrs_instance.renderAndRecord.assert_called_once()

    def test_recording_happens_when_seed_is_injected(self, monkeypatch):
        """ToolRenderAndRecordService is called when seed passes the gate too."""
        proc, mock_trrs_cls, mock_trrs_instance = _run_pre_act_with_block(
            _HIGH_BLOCK, monkeypatch
        )
        assert proc._memory_seed == _HIGH_BLOCK
        mock_trrs_cls.assert_called_once()
        mock_trrs_instance.renderAndRecord.assert_called_once()
