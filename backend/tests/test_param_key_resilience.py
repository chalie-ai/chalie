# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — argument-key resilience at the dispatch seam.

A weak model mangles the argument KEYS it emits: a stray escaped quote
(``source"``), a capital (``URL``, ``MAX_CHARS``), or a synonym (``url`` for the
read target, ``source`` for a download). Before the seam, ``json.loads`` decoded
the corrupt key faithfully, it never matched the tool's schema, and the call
bounced on a spurious required-field error (``read`` returning
``source-required`` for every URL/file a model passed under ``url``/``path``).

These are ZERO-MOCK feature tests: each drives the live
``ToolDispatcher(mp).dispatch()`` chokepoint exactly as ``MessageProcessor._step``
does — real ``AbilityRegistry`` resolution, the real ``PolicyManager.wrap`` gate
(reading the real ``policy`` table the ``db`` fixture binds), the real
``Ability.run`` with real file I/O / the real SSRF guard, and the real
``ActTrail`` write. The healing under test happens at the seam
(``abilities._params.KeyHealer.heal``), so a corrupt key is canonicalised
BEFORE the ACTION_REQUIRED pre-gate, the policy gate, or ``run()`` ever sees it.

The discriminator in every case is a downstream effect that can ONLY be reached
if the key healed: ``read`` reaching its URL/file branch instead of
``source-required``, ``web_download`` reaching its SSRF guard
(``blocked-url``) instead of the ``missing-params`` pre-gate, ``list`` resolving
a real service-backed list by its ``id`` alias. The registry-level tests assert
the structural invariants the design rests on against the REAL registry.
"""

import sqlite3
from typing import cast

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._params import Keys, KeyHealer
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from configs.channels import DmnConfig, UserConfig
from services.act_trail import ActTrail
from services.list_service import ListService
from tests._registry_invariants import RegistryInvariant, RegistryOverlapError
from tests._tool_result_harness import MP, allow_policy, parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _chat_mp(db: sqlite3.Connection, *, allow: tuple[str, ...] = ()) -> MP:
    # user-channel ``MessageProcessor`` bound to the test db, with each *allow*
    # permission flipped to ``allow`` in the real policy table.
    for permission in allow:
        allow_policy(db, permission, "chat")
    return MP(seed_transcript(db, "chat", "do a thing"), UserConfig({}))


def _list_service() -> ListService:
    from services.database_service import get_shared_db_service

    return ListService(get_shared_db_service())


# ── read: the  regression — a mangled/aliased key heals to ``source`` ─────


def test_read_mangled_source_key_heals_not_source_required(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db, allow=("read",))

    result = ToolDispatcher(mp).dispatch("read", {'source"': "http://localhost", "act_summary": "x"})

    assert "source-required" not in result, result
    assert "private-or-internal-url-blocked" in result, result

    # The real trail recorded the healed read against the anchor.
    rows = ActTrail().fetch_by_transcript_id(mp.uid)
    assert [r["tool_name"] for r in rows] == ["read"], rows
    assert "private-or-internal-url-blocked" in cast(str, rows[0]["result"]), rows


def test_read_uppercase_url_composes_normalize_and_variant(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db, allow=("read",))

    result = ToolDispatcher(mp).dispatch("read", {"URL": "http://localhost", "act_summary": "x"})

    assert "source-required" not in result, result
    assert "private-or-internal-url-blocked" in result, result


def test_read_mangled_max_chars_heals_and_truncates_downstream(db: sqlite3.Connection, tmp_path: object) -> None:
    mp = _chat_mp(db, allow=("read",))
    from pathlib import Path
    target = cast(Path, tmp_path) / "big.txt"
    target.write_text("A" * 200, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch(
        "read", {"path": str(target), "MAX_CHARS": 100, "act_summary": "x"}
    )

    assert "source-required" not in result, result
    assert "[read(status=success" in result, result
    # The healed cap took effect: the body was clipped, so the meta flags it.
    assert "truncated=true" in result, result
    assert "AAAA" in result, result


def test_read_genuinely_unknown_key_passes_through_and_still_bounces(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db, allow=("read",))

    result = ToolDispatcher(mp).dispatch(
        "read", {"foobar": "https://example.com", "act_summary": "x"}
    )

    assert "source-required" in result, result
    assert "foobar" in result, result


# ── web_download: the ladder, scoped — ``source`` heals to ``url`` ───────────────


def test_web_download_source_alias_heals_to_url(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db, allow=("web_download",))

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"source": "http://localhost/file.pdf", "act_summary": "x"}
    )

    assert "missing-params" not in result, result
    assert "blocked-url" in result, result


def test_web_download_url_stays_canonical(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db, allow=("web_download",))

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"url": "http://localhost/file.pdf", "act_summary": "x"}
    )

    assert "missing-params" not in result, result
    assert "blocked-url" in result, result


# ── list: the historical ``id`` alias heals to the ``list`` selector ─────────────


def test_list_id_alias_heals_to_list_selector_and_returns_contents(db: sqlite3.Connection) -> None:
    mp = MP(seed_transcript(db, "subconscious", "show my list"), DmnConfig())
    service = _list_service()
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["milk", "eggs"])

    result = ToolDispatcher(mp).dispatch(
        "list", {"action": "view", "id": list_id, "act_summary": "x"}
    )

    assert "[list(status=success" in result, result
    assert "missing-target" not in result and "list-not-found" not in result, result
    body = cast(dict[str, object], parse_body(result, "list", rich=True))
    texts = [cast(dict[str, object], row)["text"] for row in cast(list[object], body["items"])]
    assert "milk" in texts and "eggs" in texts, body


def test_list_mangled_action_key_heals_before_action_required_pre_gate(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db)

    result = ToolDispatcher(mp).dispatch("list", {'action"': "list_all", "act_summary": "x"})

    assert "unknown-action" not in result, result
    assert "[list(status=success" in result, result
    assert "count=0" in result, result


# ── Registry-level structural invariants (driven against the REAL registry) ──────


def _shipping_abilities() -> list[Ability]:
    # ``AbilityRegistry.all()`` walks EVERY ``Ability`` subclass, including throwaway
    # test abilities. The production registry only holds ``abilities.*`` classes — no
    # test module is imported in prod — so we filter to that real, shipping set.
    return [a for a in AbilityRegistry.all() if type(a).__module__.startswith("abilities.")]


def test_real_registry_has_no_variant_overlaps() -> None:
    RegistryInvariant().check_no_overlaps(_shipping_abilities())


def test_every_native_param_is_a_keys_constant() -> None:
    allowed = RegistryInvariant().canonical_keys()
    offenders = []
    for ability in _shipping_abilities():
        properties = (ability.get_parameters() or {}).get("properties") or {}
        offenders += [
            (ability.get_name(), key) for key in cast(dict[str, object], properties) if key not in allowed
        ]
    assert not offenders, f"non-Keys param keys declared: {offenders}"


def test_overlap_validator_rejects_a_tool_declaring_two_aliased_params() -> None:

    class _OverlapProbe(Ability):
        _SYNTHETIC = True

        def get_name(self) -> str:
            return "overlap_probe"

        def get_summary(self) -> str:
            return "probe"

        def get_examples(self) -> list[str]:
            return []

        def get_search_tooltip(self) -> str:
            return "probe"

        def get_parameters(self) -> dict[str, object]:
            return {
                "type": "object",
                "properties": {
                    Keys.source: {"type": "string"},
                    Keys.url: {"type": "string"},
                    Keys.list: {"type": "string"},
                },
                "required": [],
            }

        def run(self, params: dict[str, object]) -> ToolResult:
            return ToolResult.ok("x")

    with pytest.raises(RegistryOverlapError) as excinfo:
        RegistryInvariant().check_no_overlaps([_OverlapProbe()])

    message = str(excinfo.value)
    assert "source" in message and "url" in message, message
    assert "collides" in message, message


# ── First-write-wins collision: both ``source`` and ``url`` present for read ──────


def test_read_collision_first_key_wins_and_warning_emitted(db: sqlite3.Connection, tmp_path: object, caplog: object) -> None:
    mp = _chat_mp(db, allow=("read",))
    from pathlib import Path
    real_file = cast(Path, tmp_path) / "winner.txt"
    real_file.write_text("content if the wrong key won", encoding="utf-8")

    import logging
    import pytest as _pytest

    with cast(_pytest.LogCaptureFixture, caplog).at_level(logging.WARNING, logger="abilities._params"):
        result = ToolDispatcher(mp).dispatch(
            "read",
            {
                "source": "http://localhost",   # first → wins → SSRF block
                "url": str(real_file),          # second → dropped
                "act_summary": "x",
            },
        )

    # The SSRF winner reached run() — not a source-required bounce, not a file read.
    assert "private-or-internal-url-blocked" in result, result
    assert "[read(status=success" not in result, result

    # The warning must name both original keys and the resolved canonical.
    warn_messages = [r.message for r in cast(_pytest.LogCaptureFixture, caplog).records if r.levelno == logging.WARNING]
    assert warn_messages, "expected a KeyHealer collision warning, got none"
    combined = " ".join(str(m) for m in warn_messages)
    assert "source" in combined and "url" in combined, combined
    assert "source" in combined, combined   # the canonical that both map to


def test_read_collision_order_decides_winner_not_key_identity(db: sqlite3.Connection, tmp_path: object, caplog: object) -> None:
    mp = _chat_mp(db, allow=("read",))
    from pathlib import Path
    real_file = cast(Path, tmp_path) / "order_proof.txt"
    real_file.write_text("MARKER_CONTENT", encoding="utf-8")

    import logging
    import pytest as _pytest

    with cast(_pytest.LogCaptureFixture, caplog).at_level(logging.WARNING, logger="abilities._params"):
        result = ToolDispatcher(mp).dispatch(
            "read",
            {
                "url": str(real_file),          # first → wins → file read succeeds
                "source": "http://localhost",   # second → dropped
                "act_summary": "x",
            },
        )

    # The file-read winner reached run() — not a blocked URL.
    assert "[read(status=success" in result, result
    assert "private-or-internal-url-blocked" not in result, result
    assert "MARKER_CONTENT" in result, result

    # Collision warning fires regardless of which key wins.
    warn_messages = [r.message for r in cast(_pytest.LogCaptureFixture, caplog).records if r.levelno == logging.WARNING]
    assert warn_messages, "expected a KeyHealer collision warning, got none"
    combined = " ".join(str(m) for m in warn_messages)
    assert "source" in combined and "url" in combined, combined


# ── web_browse: the goal←query ladder — a model fluent in web_search emits query ──


def _shipping_ability(name: str) -> Ability:
    """The single REAL shipping ability with this tool name, from the live registry —
    the same instance the dispatcher resolves and the same ``get_parameters()`` the
    healer reads at the seam."""
    return next(a for a in _shipping_abilities() if a.get_name() == name)


def test_web_browse_query_alias_heals_to_goal() -> None:
    """web_search names its task slot ``query``; web_browse names the SAME slot
    ``goal``. A model fluent in web_search reaches for ``query`` on web_browse. The
    goal←query ladder heals it at the dispatch seam — byte-identical to the heal
    ``ToolDispatcher.dispatch`` performs (``KeyHealer.heal(params,
    ability.get_parameters())``) — so the call clears the ACTION_REQUIRED ``goal``
    pre-gate instead of bouncing on ``missing-params``. Asserted at the seam, not
    end-to-end, because ``web_browse.run()`` spawns a live browser delegate a unit
    test must never start (see ``test_web_delegates`` for the same reason)."""
    browse = _shipping_ability("web_browse")

    healed = KeyHealer().heal(
        {"query": "book the earliest appointment"}, browse.get_parameters()
    )

    assert healed == {Keys.goal: "book the earliest appointment"}


def test_web_browse_goal_canonical_is_untouched() -> None:
    """The canonical ``goal`` still passes the exact-match layer verbatim — the new
    ladder never disturbs a model that already emits ``goal``."""
    browse = _shipping_ability("web_browse")

    healed = KeyHealer().heal({"goal": "read the page"}, browse.get_parameters())

    assert healed == {Keys.goal: "read the page"}


def test_web_browse_unknown_goal_synonym_still_bounces(db: sqlite3.Connection) -> None:
    """Only the REGISTERED ``query`` synonym bridges to ``goal``. A genuinely unknown
    key (``task``) passes through verbatim, so the real ACTION_REQUIRED pre-gate
    still bounces with ``missing-params`` naming ``goal``. Drives the real
    ``ToolDispatcher.dispatch`` chokepoint; it bounces at the pre-gate, so no
    delegate spawns and it cannot hang."""
    mp = _chat_mp(db)

    result = ToolDispatcher(mp).dispatch(
        "web_browse", {"task": "book an appointment", "act_summary": "x"}
    )

    assert "code=missing-params" in result, result
    assert "valid: goal" in result, result


def test_web_search_query_canonical_untouched_by_goal_ladder() -> None:
    """web_search declares ``query`` as its OWN canonical, so the exact-match layer
    keeps it verbatim. The goal←query ladder is scoped to web_browse (the only tool
    that declares ``goal``), so it never rewrites web_search's ``query`` to ``goal``
    — the cross-tool safety the no-overlap invariant guarantees, proven at the seam."""
    search = _shipping_ability("web_search")

    healed = KeyHealer().heal(
        {"query": "latest fusion energy developments"}, search.get_parameters()
    )

    assert healed == {Keys.query: "latest fusion energy developments"}
