"""
Unit tests verifying that _run_analysis publishes an 'image_ready' event to the
``output:events`` pub/sub channel after analysis completes (Step 3 / B5 fix).

The existing ``_drift_sender`` thread in websocket.py subscribes to
``output:events`` and forwards every message it receives to the connected
WebSocket client.  These tests verify the *publishing* side — that
``_run_analysis`` calls ``store.publish('output:events', ...)`` with the correct
payload on both the success and failure paths.

All tests are marked @pytest.mark.unit and require no real Redis or vision
provider.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_publish_calls(mock_store):
    """
    Return a list of (channel, payload_dict) tuples from all ``store.publish``
    calls recorded on *mock_store*.

    Args:
        mock_store: A ``MagicMock`` whose ``publish`` attribute has been called.

    Returns:
        list[tuple[str, dict]]: Decoded publish arguments, one entry per call.
    """
    calls = []
    for call in mock_store.publish.call_args_list:
        channel, payload_str = call[0]
        calls.append((channel, json.loads(payload_str)))
    return calls


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_run_analysis_publishes_image_ready_on_success():
    """
    After successful vision analysis, ``_run_analysis`` must publish a message to
    ``output:events`` with ``{"type": "image_ready", "image_id": ..., "status": "ready"}``.

    This verifies Step 3 of the plan: the frontend receives the event via the
    existing ``_drift_sender`` thread and removes the upload spinner (B4/B5 fix).
    """
    from api.chat_image import _run_analysis

    mock_store = MagicMock()
    mock_result = {
        'description': 'A test image.',
        'ocr_text': '',
        'has_text': False,
        'error': None,
        'analysis_time_ms': 120,
    }

    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('services.image_context_service.analyze', return_value=mock_result):
        _run_analysis('testid001testid0', b'fake-png-bytes', 'image/png')

    publish_calls = _extract_publish_calls(mock_store)
    assert publish_calls, "store.publish() should be called at least once on success"

    channel, payload = publish_calls[0]
    assert channel == 'output:events', (
        f"Expected publish to 'output:events', got '{channel}'"
    )
    assert payload['type'] == 'image_ready', (
        f"Expected type='image_ready', got '{payload.get('type')}'"
    )
    assert payload['image_id'] == 'testid001testid0'
    assert payload['status'] == 'ready', (
        f"Expected status='ready' on success, got '{payload.get('status')}'"
    )


@pytest.mark.unit
def test_run_analysis_publishes_image_ready_on_failure():
    """
    When the vision analysis raises an exception, ``_run_analysis`` must still
    publish ``{"type": "image_ready", ..., "status": "failed"}`` to
    ``output:events`` so the frontend can surface the error instead of leaving
    the spinner spinning indefinitely (B5 fix).
    """
    from api.chat_image import _run_analysis

    mock_store = MagicMock()

    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('services.image_context_service.analyze',
               side_effect=RuntimeError('No vision provider configured')):
        _run_analysis('failid002failid0', b'fake-png-bytes', 'image/png')

    publish_calls = _extract_publish_calls(mock_store)
    assert publish_calls, "store.publish() should be called even when analysis fails"

    channel, payload = publish_calls[0]
    assert channel == 'output:events', (
        f"Expected publish to 'output:events', got '{channel}'"
    )
    assert payload['type'] == 'image_ready'
    assert payload['image_id'] == 'failid002failid0'
    assert payload['status'] == 'failed', (
        f"Expected status='failed' on exception, got '{payload.get('status')}'"
    )


@pytest.mark.unit
def test_run_analysis_stores_result_before_publishing():
    """
    The result must be written to MemoryStore **before** ``output:events`` is
    published, so that any WebSocket client that immediately calls
    ``store.get(result_key)`` upon receiving the event finds the data.

    This is verified by inspecting the order of calls on the mock store.
    """
    from api.chat_image import _run_analysis

    call_order = []
    mock_store = MagicMock()
    mock_store.set.side_effect = lambda *a, **kw: call_order.append('set')
    mock_store.publish.side_effect = lambda *a, **kw: call_order.append('publish')

    mock_result = {
        'description': 'Ordered call test.',
        'ocr_text': '',
        'has_text': False,
        'error': None,
        'analysis_time_ms': 50,
    }

    with patch('api.chat_image._get_store', return_value=mock_store), \
         patch('services.image_context_service.analyze', return_value=mock_result):
        _run_analysis('orderid03orderid', b'bytes', 'image/jpeg')

    # At least one 'set' must appear before the first 'publish'
    assert 'set' in call_order and 'publish' in call_order, (
        "Both set() and publish() must be called"
    )
    first_set = call_order.index('set')
    first_publish = call_order.index('publish')
    assert first_set < first_publish, (
        "store.set() (result storage) must happen before store.publish() "
        f"(event push); got order: {call_order}"
    )
