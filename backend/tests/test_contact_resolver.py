"""Unit tests for contact_resolver — DataGraphService-backed people index."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _mock_dgs(recall_return=None, store_return=None):
    dgs = MagicMock()
    dgs.recall.return_value = recall_return or []
    dgs.store.return_value = store_return or {'id': 1, 'key': 'contact:x@y.com', 'value': 'X'}
    return dgs


@pytest.mark.unit
def test_index_stores_person():
    from capabilities.contact_resolver import index_person

    mock = _mock_dgs()
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        index_person("alice@example.com", "Alice Smith", source="imap")

    mock.store.assert_called_once()
    assert mock.store.call_args.kwargs["key"] == "contact:alice@example.com"
    assert mock.store.call_args.kwargs["value"] == "Alice Smith"
    assert mock.store.call_args.kwargs["kind"] == "user_specific"
    assert mock.store.call_args.kwargs["source"] == "imap"


@pytest.mark.unit
def test_index_normalizes_email():
    from capabilities.contact_resolver import index_person

    mock = _mock_dgs()
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        index_person("  Bob@Corp.COM  ", "Bob Jones")

    assert mock.store.call_args.kwargs["key"] == "contact:bob@corp.com"


@pytest.mark.unit
@pytest.mark.parametrize("email,name", [
    ("not-an-email", "Name"),
    (None, "Name"),
])
def test_index_skips_invalid(email, name):
    from capabilities.contact_resolver import index_person

    mock = _mock_dgs()
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        index_person(email, name)

    mock.store.assert_not_called()


@pytest.mark.unit
def test_index_survives_exception():
    from capabilities.contact_resolver import index_person

    mock = _mock_dgs()
    mock.store.side_effect = RuntimeError("db error")
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        # Must not raise
        index_person("alice@example.com", "Alice", source="imap")


@pytest.mark.unit
def test_resolve_returns_matches():
    from capabilities.contact_resolver import resolve

    mock = _mock_dgs(recall_return=[
        {"id": 1, "key": "contact:sarah@corp.com", "value": "Sarah Chen",
         "retrieval_weight": 0.9, "evidence_count": 1, "composite_score": 2.0},
    ])
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        result = resolve("Sarah")

    assert result == [{"email": "sarah@corp.com", "name": "Sarah Chen"}]


@pytest.mark.unit
def test_resolve_filters_non_contact_keys():
    """Only rows with 'contact:' prefix are returned."""
    from capabilities.contact_resolver import resolve

    mock = _mock_dgs(recall_return=[
        {"id": 1, "key": "contact:alice@corp.com", "value": "Alice",
         "retrieval_weight": 0.9, "evidence_count": 1, "composite_score": 2.0},
        {"id": 2, "key": "user_name", "value": "Sam",
         "retrieval_weight": 0.8, "evidence_count": 1, "composite_score": 1.5},
    ])
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        result = resolve("Alice")

    assert len(result) == 1
    assert result[0]["email"] == "alice@corp.com"




# --- resolve_contact tool tests ---




@pytest.mark.unit
def test_tool_resolves_by_name():
    from capabilities.contact_resolver import get_tool

    mock = _mock_dgs(recall_return=[
        {"id": 1, "key": "contact:sarah@corp.com", "value": "Sarah Chen",
         "retrieval_weight": 0.9, "evidence_count": 1, "composite_score": 2.0},
    ])
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        handler = get_tool()["handler"]
        result = handler("topic1", {"query": "Sarah"})

    assert result["count"] == 1
    assert result["contacts"][0]["email"] == "sarah@corp.com"
    assert result["contacts"][0]["name"] == "Sarah Chen"




@pytest.mark.unit
def test_tool_empty_query_returns_empty():
    from capabilities.contact_resolver import get_tool

    handler = get_tool()["handler"]
    result = handler("topic1", {"query": ""})

    assert result == {"contacts": [], "count": 0}




@pytest.mark.unit
def test_tool_respects_limit():
    from capabilities.contact_resolver import get_tool

    mock = _mock_dgs(recall_return=[])
    with patch("capabilities.contact_resolver._dgs", return_value=mock):
        handler = get_tool()["handler"]
        handler("topic1", {"query": "test", "limit": 3})

    _, kwargs = mock.recall.call_args
    assert kwargs["limit"] == 3


