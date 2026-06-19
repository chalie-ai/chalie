# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Contacts-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
contacts ability's genuine behaviour tests (list/get precision, ambiguous-match,
not-found with hint) that have no coverage elsewhere."""

import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig, UserConfig
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="find John's number")


def _seed_contact(
    *,
    fn: str,
    given_name: str = "",
    family_name: str = "",
    emails: list[object] | None = None,
    phones: list[object] | None = None,
    org: str = "",
    title: str = "",
) -> dict[str, object]:
    """Must use ``contact_resolver.index_contact_profile`` (CardDAV ingest entry
    point) so the contact lands as a ``user_specific`` data_graph row resolvable by
    the ability."""
    from capabilities.contact_resolver import index_contact_profile

    profile: dict[str, object] = {
        "fn": fn,
        "given_name": given_name,
        "family_name": family_name,
        "emails": emails or [],
        "phones": phones or [],
        "org": org,
        "title": title,
    }
    index_contact_profile(profile, source="carddav")
    return profile


@pytest.fixture
def dmn_mp(db: sqlite3.Connection) -> _MP:
    """A real non-user-broadcast mp (``broadcast_to is None``): contacts.list /
    contacts.get are seeded ``allow`` on the subconscious channel so the real
    policy gate passes, and no live WebSocket is needed (the act-event emitter is
    a real no-op). The dispatcher drops the rich card here."""
    return _MP(_seed_transcript(db, "subconscious"), DmnConfig())


@pytest.fixture
def user_mp(db: sqlite3.Connection) -> _MP:
    """A real user-broadcast mp (``broadcast_to == 'user'``). On this channel the
    dispatcher assigns a rich-media ordinal and renders the contacts card
    trailer. contacts.list / contacts.get are seeded ``allow`` on chat."""
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "contacts") -> object:
    return parse_body(rendered, tool, rich=True)


def test_list_returns_structured_rows_with_count(db: sqlite3.Connection, dmn_mp: _MP) -> None:
    _seed_contact(
        fn="John Smith", given_name="John", family_name="Smith",
        emails=[{"value": "john.smith@example.com", "type": "home"}],
        phones=[{"value": "+35699111222", "type": "cell"}],
    )
    _seed_contact(
        fn="Mike Borg", given_name="Mike", family_name="Borg",
        emails=[{"value": "mike@borg.mt", "type": "home"}],
        phones=[{"value": "+35679000000", "type": "cell"}],
    )

    out = ToolDispatcher(dmn_mp).dispatch("contacts", {"action": "list"})

    assert "[contacts(status=success" in out
    assert "action=list" in out
    from typing import cast
    body = cast("dict[str, object]", _parse_body(out))
    assert body["action_performed"] == "list"
    assert body["count"] == 2
    names = {c.get("fn") for c in cast("list[dict[str, object]]", body["contacts"])}
    assert names == {"John Smith", "Mike Borg"}
    john = next(c for c in cast("list[dict[str, object]]", body["contacts"]) if c["fn"] == "John Smith")
    assert cast("list[dict[str, str]]", john["emails"])[0]["value"] == "john.smith@example.com"
    assert cast("list[dict[str, str]]", john["phones"])[0]["value"] == "+35699111222"


def test_list_with_query_filters_through_resolve(db: sqlite3.Connection, dmn_mp: _MP) -> None:
    _seed_contact(fn="Mike Borg", given_name="Mike", family_name="Borg")
    _seed_contact(fn="Sarah Vella", given_name="Sarah", family_name="Vella")

    out = ToolDispatcher(dmn_mp).dispatch(
        "contacts", {"action": "list", "query": "Mike"}
    )

    assert "[contacts(status=success" in out
    from typing import cast
    body = cast("dict[str, object]", _parse_body(out))
    names = {c.get("fn") for c in cast("list[dict[str, object]]", body["contacts"])}
    assert "Mike Borg" in names
    assert "Sarah Vella" not in names


def test_get_exact_name_returns_single_contact(db: sqlite3.Connection, user_mp: _MP) -> None:
    """Pairs a rich card on the user-broadcast channel."""
    _seed_contact(
        fn="John Smith", given_name="John", family_name="Smith",
        emails=[{"value": "john.smith@example.com", "type": "home"}],
    )
    _seed_contact(fn="John Doe", given_name="John", family_name="Doe")

    out = ToolDispatcher(user_mp).dispatch(
        "contacts", {"action": "get", "identifier": "John Smith", "act_summary": "x"}
    )

    assert "[contacts(status=success" in out
    assert "action=get" in out
    assert "<span id='contacts_1'>" in out
    from typing import cast
    payload = cast("dict[str, object]", _parse_body(out))
    assert payload["action_performed"] == "get"
    assert cast("dict[str, str]", payload["contact"])["fn"] == "John Smith"


def test_get_ambiguous_lists_candidates_picks_nothing(db: sqlite3.Connection, dmn_mp: _MP) -> None:
    """Returns ``code=ambiguous-match`` with BOTH candidate names — never a
    silent first-hit pick."""
    _seed_contact(fn="John Smith", given_name="John", family_name="Smith")
    _seed_contact(fn="John Doe", given_name="John", family_name="Doe")

    out = ToolDispatcher(dmn_mp).dispatch(
        "contacts", {"action": "get", "identifier": "John"}
    )

    assert "[contacts(status=error, code=ambiguous-match" in out
    assert "code=error]" not in out
    assert "John Smith" in out
    assert "John Doe" in out
    assert "hint:" in out


def test_get_total_miss_returns_not_found_with_closest_match(db: sqlite3.Connection, dmn_mp: _MP) -> None:
    """Errors ``code=not-found`` with a closest-match hint naming a real candidate."""
    _seed_contact(fn="John Smith", given_name="John", family_name="Smith")
    _seed_contact(fn="Mike Borg", given_name="Mike", family_name="Borg")

    out = ToolDispatcher(dmn_mp).dispatch(
        "contacts", {"action": "get", "identifier": "Smith Family"}
    )

    assert "[contacts(status=error, code=not-found" in out
    assert "code=error]" not in out
    assert "hint:" in out
    assert "closest" in out.lower()
    # The hint names a real candidate the model can re-issue against.
    assert "John Smith" in out
