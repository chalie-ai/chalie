# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the contacts tool's ToolResult contract (TKT-905).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution of the production
``ContactsAbility`` (a ``CapabilityAbility`` subclass), the real
``ToolDispatcher._prevalidate`` ACTION_REQUIRED pre-gate, the real
``PolicyManager.wrap`` gate reading the real ``policy`` table (contacts.list /
contacts.get are seeded ``allow``), the real ``ContactsAbility.run`` reading the
real data graph, and the real ``ActTrail`` write.

Contacts are seeded the production way — through
``contact_resolver.index_contact_profile`` (the same entry point the CardDAV
ingest calls) into the real ``data_graph`` ``user_specific`` rows — and read back
through the ability, so the whole index → resolve → contract chain runs for real.

What TKT-905 changes, exercised end to end:

* **Reads run inline against the local index** — ``list`` / ``get`` no longer
  hit the base's connected gate when the index has data, so local contacts work
  offline (the old base flow refused every read with not-connected).
* **get precision contract:** a single relevant candidate is returned; ≥2
  relevant candidates → ``code=ambiguous-match`` with candidate rows in the
  body (never a silent first-hit pick); 0 relevant + fuzzy candidates →
  ``code=not-found`` with a closest-match hint naming real candidates.
* **Contract + rich:** every action returns a ``ToolResult``; success bodies are
  JSON rows (uid/fn/emails/phones for ``list``, the contact dict for ``get``)
  and pair a rich card via ``ToolResult(rich=…)`` (the dispatcher owns the
  ordinal + the single span instruction); errors carry stable kebab codes
  (NOT the ``code="error"`` placeholder) + hints.
* **Not-connected preservation:** an empty index with no connected mail
  capability still surfaces the base's ``code=not-connected`` remediation.

RED-before-change: get-miss shipped as ``status=success`` with a
``{"error": "Contact not found"}`` body (no precision, no ambiguity gate); list
refused with not-connected even when the local index had data; no rich card ever
travelled. These tests fail against the pre-TKT-905 ability.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig, UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP as _MP
from tests._tool_result_harness import parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    return seed_transcript(db, channel=channel, content="find John's number")


def _seed_contact(
    *,
    fn: str,
    given_name: str = "",
    family_name: str = "",
    emails=None,
    phones=None,
    org: str = "",
    title: str = "",
) -> dict:
    """Index a contact the production way — through
    ``contact_resolver.index_contact_profile`` (the CardDAV ingest entry point) —
    so it lands as a ``user_specific`` data_graph row resolvable by the ability."""
    from capabilities.contact_resolver import index_contact_profile

    profile = {
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
def dmn_mp(db):
    """A real non-user-broadcast mp (``broadcast_to is None``): contacts.list /
    contacts.get are seeded ``allow`` on the subconscious channel so the real
    policy gate passes, and no live WebSocket is needed (the act-event emitter is
    a real no-op). The dispatcher drops the rich card here."""
    return _MP(_seed_transcript(db, "subconscious"), DmnConfig())


@pytest.fixture
def user_mp(db):
    """A real user-broadcast mp (``broadcast_to == 'user'``). On this channel the
    dispatcher assigns a rich-media ordinal and renders the contacts card
    trailer. contacts.list / contacts.get are seeded ``allow`` on chat."""
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "contacts") -> object:
    """Extract and JSON-parse the body (the JSON head before any card trailer)."""
    return parse_body(rendered, tool, rich=True)


# ── Foundation ──────────────────────────────────────────────────────────────────


def test_contacts_is_a_capability_ability():
    """The ability stays a ``CapabilityAbility`` subclass keyed to the mail
    capability — the precision logic is built on the shared base, not bespoke."""
    from abilities._capability import CapabilityAbility
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("contacts")
    assert isinstance(ability, CapabilityAbility)
    assert ability.CAPABILITY_KEY == "mail"


# ── 1. list with seeded contacts → structured rows + count meta ─────────────────


def test_list_returns_structured_rows_with_count(db, dmn_mp):
    """``list`` with seeded contacts serves them from the local index (no
    connected gate) as JSON rows carrying uid/fn/emails/phones, with a count
    meta — offline, regardless of mail-capability connection state."""
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
    body = _parse_body(out)
    assert body["action_performed"] == "list"
    assert body["count"] == 2
    names = {c.get("fn") for c in body["contacts"]}
    assert names == {"John Smith", "Mike Borg"}
    john = next(c for c in body["contacts"] if c["fn"] == "John Smith")
    assert john["emails"][0]["value"] == "john.smith@example.com"
    assert john["phones"][0]["value"] == "+35699111222"


def test_list_renders_rich_card_on_user_broadcast(db, user_mp):
    """On a user-broadcasting channel ``list`` pairs a rich card: the dispatcher
    injects the ordinal-keyed span instruction and the card payload (JSON head
    before the blank line) carries the contacts + action_performed."""
    assert getattr(user_mp.config, "broadcast_to", None) == "user"
    _seed_contact(
        fn="Sarah Vella", given_name="Sarah", family_name="Vella",
        emails=[{"value": "sarah@vella.mt", "type": "work"}],
    )
    _seed_contact(fn="Alex Camilleri", given_name="Alex", family_name="Camilleri")

    out = ToolDispatcher(user_mp).dispatch(
        "contacts", {"action": "list", "act_summary": "x"}
    )

    assert "[contacts(status=success" in out
    assert "<span id='contacts_1'>" in out
    payload = _parse_body(out)
    assert payload["action_performed"] == "list"
    assert {c["fn"] for c in payload["contacts"]} == {"Sarah Vella", "Alex Camilleri"}


# ── 2. list with a query filters through real resolve() ─────────────────────────


def test_list_with_query_filters_through_resolve(db, dmn_mp):
    """A ``query`` routes ``list`` through the real ``resolve()`` RRF lookup so the
    rows are the matched subset, not the whole book."""
    _seed_contact(fn="Mike Borg", given_name="Mike", family_name="Borg")
    _seed_contact(fn="Sarah Vella", given_name="Sarah", family_name="Vella")

    out = ToolDispatcher(dmn_mp).dispatch(
        "contacts", {"action": "list", "query": "Mike"}
    )

    assert "[contacts(status=success" in out
    body = _parse_body(out)
    names = {c.get("fn") for c in body["contacts"]}
    assert "Mike Borg" in names
    assert "Sarah Vella" not in names


# ── 3. get exact name → single contact body + rich ──────────────────────────────


def test_get_exact_name_returns_single_contact(db, user_mp):
    """``get`` for a name that ci-equals exactly one candidate's fn returns that
    one contact (even when fuzzy resolve also surfaces a same-first-name sibling),
    and pairs a rich card on the user-broadcast channel."""
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
    payload = _parse_body(out)
    assert payload["action_performed"] == "get"
    assert payload["contact"]["fn"] == "John Smith"


# ── 4. get ambiguous → code=ambiguous-match, nothing chosen ─────────────────────


def test_get_ambiguous_lists_candidates_picks_nothing(db, dmn_mp):
    """``get "John"`` with two relevant John candidates returns
    ``code=ambiguous-match`` with BOTH candidate names in the body — never a
    silent first-hit pick, and no single ``contact`` payload."""
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


# ── 5. get total miss on a non-empty index → not-found + closest-match hint ──────


def test_get_total_miss_returns_not_found_with_closest_match(db, dmn_mp):
    """``get`` for an identifier with NO relevant candidate but fuzzy hits on a
    non-empty index errors ``code=not-found`` with a closest-match hint naming a
    real candidate — the spec's closest-match signal."""
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


# ── 6. get without identifier → pre-gate missing-params ──────────────────────────


def test_get_without_identifier_reports_missing_params(db, dmn_mp):
    """``get`` with no ``identifier`` is pre-gated by ACTION_REQUIRED into a
    ``code=missing-params`` error BEFORE run() — never an empty/None lookup."""
    out = ToolDispatcher(dmn_mp).dispatch("contacts", {"action": "get"})

    assert "[contacts(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "identifier" in out


# ── 7. unknown action → unknown-action with valid ladder ────────────────────────


def test_unknown_action_lists_valid_actions(db, dmn_mp):
    """An unknown action errors ``code=unknown-action`` whose ``valid:`` line names
    the real actions list / get."""
    out = ToolDispatcher(dmn_mp).dispatch("contacts", {"action": "teleport"})

    assert "[contacts(status=error, code=unknown-action" in out
    assert "valid:" in out
    assert "list" in out
    assert "get" in out


# ── 8. empty index + not connected → not-connected remediation preserved ────────


def test_empty_index_not_connected_preserves_not_connected_error(db, dmn_mp):
    """With NO seeded contacts and no connected mail capability, ``list`` surfaces
    the base's ``code=not-connected`` remediation (naming the mail integration) —
    the connect signal survives the inline-reads change."""
    out = ToolDispatcher(dmn_mp).dispatch("contacts", {"action": "list"})

    assert "[contacts(status=error, code=not-connected" in out
    assert "code=error]" not in out
    assert "not connected" in out.lower()
    assert "mail integration" in out.lower()


# ── Act-trail records the rendered envelope ─────────────────────────────────────


def test_act_trail_records_the_envelope(db, dmn_mp):
    """The act-trail records the same non-empty contacts envelope against the
    transcript anchor — the cross-step write really lands."""
    _seed_contact(fn="John Smith", given_name="John", family_name="Smith")
    ToolDispatcher(dmn_mp).dispatch("contacts", {"action": "list"})

    trail = ActTrail().fetch_by_transcript_id(dmn_mp.uid)
    assert any("[contacts(status=" in row["result"] for row in trail)
