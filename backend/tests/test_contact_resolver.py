"""Unit tests for ContactResolver — people index and identity resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _patch_ks():
    """Patch _get_knowledge_service so no real DB is needed."""
    ks = MagicMock()
    ks.recall.return_value = []
    with patch("capabilities.contact_resolver._get_knowledge_service", return_value=ks):
        yield ks


@pytest.mark.unit
def test_index_stores_person(_patch_ks):
    from capabilities.contact_resolver import index_person

    index_person("alice@example.com", "Alice Smith", source="imap")

    _patch_ks.store.assert_called_once()
    assert _patch_ks.store.call_args.kwargs["key"] == "alice@example.com"


@pytest.mark.unit
def test_index_normalizes_email(_patch_ks):
    from capabilities.contact_resolver import index_person

    index_person("  Bob@Corp.COM  ", "Bob Jones")

    assert _patch_ks.store.call_args.kwargs["key"] == "bob@corp.com"


@pytest.mark.unit
@pytest.mark.parametrize("email,name", [
    ("anon@example.com", None),
    ("anon@example.com", "   "),
    ("not-an-email", "Name"),
    ("", "Name"),
    (None, "Name"),
])
def test_index_skips_invalid(_patch_ks, email, name):
    from capabilities.contact_resolver import index_person

    index_person(email, name)

    _patch_ks.store.assert_not_called()


@pytest.mark.unit
def test_index_survives_exception(_patch_ks):
    from capabilities.contact_resolver import index_person
    _patch_ks.store.side_effect = RuntimeError("db error")

    index_person("alice@example.com", "Alice", source="imap")


@pytest.mark.unit
def test_resolve_returns_matches(_patch_ks):
    from capabilities.contact_resolver import resolve
    _patch_ks.recall.return_value = [{"key": "sarah@corp.com", "value": "Sarah Chen"}]

    assert resolve("Sarah") == [{"email": "sarah@corp.com", "name": "Sarah Chen"}]


@pytest.mark.unit
def test_resolve_empty_or_no_match(_patch_ks):
    from capabilities.contact_resolver import resolve

    assert resolve("") == []
    assert resolve(None) == []


@pytest.mark.unit
def test_resolve_survives_exception(_patch_ks):
    from capabilities.contact_resolver import resolve
    _patch_ks.recall.side_effect = RuntimeError("db error")

    assert resolve("Alice") == []
