# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — argument-key resilience at the dispatch seam (TKT-963 / TKT-964).

A weak model mangles the argument KEYS it emits: a stray escaped quote
(``source"``), a capital (``URL``, ``MAX_CHARS``), or a synonym (``url`` for the
read target, ``source`` for a download). Before the seam, ``json.loads`` decoded
the corrupt key faithfully, it never matched the tool's schema, and the call
bounced on a spurious required-field error — TKT-963 (``read`` returning
``source-required`` for every URL/file a model passed under ``url``/``path``).

These are ZERO-MOCK feature tests: each drives the live
``ToolDispatcher(mp).dispatch()`` chokepoint exactly as ``MessageProcessor._loop``
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

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._params import Keys
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from configs.channels import DmnConfig, UserConfig
from services.act_trail import ActTrail
from tests._registry_invariants import RegistryInvariant, RegistryOverlapError
from tests._tool_result_harness import MP, allow_policy, parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _chat_mp(db, *, allow: "tuple[str, ...]" = ()) -> MP:
    """A real user-channel mp bound to the test db, with each *allow* permission
    flipped to ``allow`` in the real policy table (so a gate that ships ``ask`` on
    a headless channel passes through to the production ``run()``)."""
    for permission in allow:
        allow_policy(db, permission, "chat")
    return MP(seed_transcript(db, "chat", "do a thing"), UserConfig({}))


def _list_service():
    """A real ``ListService`` bound to the test db — built off the same
    ``get_shared_db_service()`` singleton ``ListAbility.run`` builds it from."""
    from services.database_service import get_shared_db_service
    from services.list_service import ListService

    return ListService(get_shared_db_service())


# ── read: the TKT-963 regression — a mangled/aliased key heals to ``source`` ─────


def test_read_mangled_source_key_heals_not_source_required(db):
    """The exact TKT-963 defect: a model stored ``{'source"': <url>}`` (a stray
    escaped quote on the KEY). The seam strips the junk char so the key matches the
    declared ``source`` and the call reaches the URL branch (a deterministic,
    network-free SSRF block) instead of bouncing on ``source-required``."""
    mp = _chat_mp(db, allow=("read",))

    result = ToolDispatcher(mp).dispatch("read", {'source"': "http://localhost", "act_summary": "x"})

    assert "source-required" not in result, result
    assert "private-or-internal-url-blocked" in result, result

    # The real trail recorded the healed read against the anchor.
    rows = ActTrail().fetch_by_transcript_id(mp.uid)
    assert [r["tool_name"] for r in rows] == ["read"], rows
    assert "private-or-internal-url-blocked" in rows[0]["result"], rows


def test_read_uppercase_url_composes_normalize_and_variant(db):
    """``{'URL': …}`` exercises BOTH layers in one key: normalise (lower-case
    ``URL`` → ``url``) THEN variant resolution (``url`` is read's ``source`` alias).
    It must reach the URL branch, not ``source-required``."""
    mp = _chat_mp(db, allow=("read",))

    result = ToolDispatcher(mp).dispatch("read", {"URL": "http://localhost", "act_summary": "x"})

    assert "source-required" not in result, result
    assert "private-or-internal-url-blocked" in result, result


def test_read_mangled_max_chars_heals_and_truncates_downstream(db, tmp_path):
    """``MAX_CHARS`` (capitalised) must heal to the declared ``max_chars`` — proven
    by a DOWNSTREAM effect: the file is clipped to the healed cap (100) and the
    envelope carries ``truncated=true``. If the key did NOT heal, ``max_chars``
    would fall back to its 20000 default, the 200-char file would NOT be clipped,
    and ``truncated`` would be absent. The compound call also heals ``path`` →
    ``source`` in the same dict."""
    mp = _chat_mp(db, allow=("read",))
    target = tmp_path / "big.txt"
    target.write_text("A" * 200, encoding="utf-8")

    result = ToolDispatcher(mp).dispatch(
        "read", {"path": str(target), "MAX_CHARS": 100, "act_summary": "x"}
    )

    assert "source-required" not in result, result
    assert "[read(status=success" in result, result
    # The healed cap took effect: the body was clipped, so the meta flags it.
    assert "truncated=true" in result, result
    assert "AAAA" in result, result


def test_read_genuinely_unknown_key_passes_through_and_still_bounces(db):
    """Healing must NOT over-reach: a key that is neither a declared param nor a
    known variant (``foobar``) passes through VERBATIM, so ``read`` still bounces on
    ``source-required`` and echoes the received key for the model to self-correct.
    This guards against a future over-broad variant ladder silently absorbing junk."""
    mp = _chat_mp(db, allow=("read",))

    result = ToolDispatcher(mp).dispatch(
        "read", {"foobar": "https://example.com", "act_summary": "x"}
    )

    assert "source-required" in result, result
    assert "foobar" in result, result


# ── web_download: the ladder, scoped — ``source`` heals to ``url`` ───────────────


def test_web_download_source_alias_heals_to_url(db):
    """``web_download({'source': …})`` — a model reusing read's key — must heal to
    the canonical ``url`` and reach the SSRF guard (``blocked-url``), NOT bounce on
    the ACTION_REQUIRED ``missing-params`` pre-gate (which fires when ``url`` is
    absent). ``localhost`` makes the SSRF block deterministic and network-free."""
    mp = _chat_mp(db, allow=("web_download",))

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"source": "http://localhost/file.pdf", "act_summary": "x"}
    )

    assert "missing-params" not in result, result
    assert "blocked-url" in result, result


def test_web_download_url_stays_canonical(db):
    """The same call under the canonical ``url`` resolves identically — proving the
    variant resolution is per-tool scoped (``url`` is canonical for ``web_download``
    even though it is a ``source`` ALIAS for ``read``)."""
    mp = _chat_mp(db, allow=("web_download",))

    result = ToolDispatcher(mp).dispatch(
        "web_download", {"url": "http://localhost/file.pdf", "act_summary": "x"}
    )

    assert "missing-params" not in result, result
    assert "blocked-url" in result, result


# ── list: the historical ``id`` alias heals to the ``list`` selector ─────────────


def test_list_id_alias_heals_to_list_selector_and_returns_contents(db):
    """``list({'action':'view', 'id': <id>})`` — the frontend silent-action key —
    must heal ``id`` → the declared ``list`` selector and resolve the REAL
    service-backed list, returning its actual contents (not ``missing-target`` /
    ``list-not-found``). A non-broadcast channel drops the card so the body head is
    the model-facing rows."""
    mp = MP(seed_transcript(db, "subconscious", "show my list"), DmnConfig())
    service = _list_service()
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["milk", "eggs"])

    result = ToolDispatcher(mp).dispatch(
        "list", {"action": "view", "id": list_id, "act_summary": "x"}
    )

    assert "[list(status=success" in result, result
    assert "missing-target" not in result and "list-not-found" not in result, result
    body = parse_body(result, "list", rich=True)
    texts = [row["text"] for row in body["items"]]
    assert "milk" in texts and "eggs" in texts, body


def test_list_mangled_action_key_heals_before_action_required_pre_gate(db):
    """A mangled ACTION key (``{'action"': 'list_all'}``) must heal to ``action``
    BEFORE the ACTION_REQUIRED pre-gate reads it. Without the heal the pre-gate
    sees no ``action`` and returns ``unknown-action`` (\"requires an 'action'\");
    with it, ``list_all`` runs and returns a ``count=0`` success."""
    mp = _chat_mp(db)

    result = ToolDispatcher(mp).dispatch("list", {'action"': "list_all", "act_summary": "x"})

    assert "unknown-action" not in result, result
    assert "[list(status=success" in result, result
    assert "count=0" in result, result


# ── Registry-level structural invariants (driven against the REAL registry) ──────


def _shipping_abilities() -> list:
    """The abilities that actually ship — those defined in the ``abilities``
    package. ``AbilityRegistry.all()`` walks EVERY ``Ability`` subclass, which
    during a full-suite pytest run also sweeps up the throwaway test abilities
    other modules define (``_tr_param`` & friends). The production registry only
    ever holds the ``abilities.*`` classes — no test module is imported in prod —
    so we assert the invariants against that real, shipping set, not the
    test-contaminated global subclass walk."""
    return [a for a in AbilityRegistry.all() if type(a).__module__.startswith("abilities.")]


def test_real_registry_has_no_variant_overlaps():
    """The whole design rests on this: within every shipping tool, no variant of
    one declared parameter may collide with another declared parameter or a
    framework key. ``check_no_overlaps`` over the real registry must not raise."""
    RegistryInvariant().check_no_overlaps(_shipping_abilities())


def test_every_native_param_is_a_keys_constant():
    """Schema-completeness — the single-source-of-truth invariant the sweep
    establishes: every property key every shipping ability declares MUST be a
    :class:`Keys` constant, so the wire keys and the variant registry can never
    drift from the schema. A new ability that hard-codes a bare string key fails
    here."""
    allowed = RegistryInvariant().canonical_keys()
    offenders = []
    for ability in _shipping_abilities():
        properties = (ability.get_parameters() or {}).get("properties") or {}
        offenders += [
            (ability.get_name(), key) for key in properties if key not in allowed
        ]
    assert not offenders, f"non-Keys param keys declared: {offenders}"


def test_overlap_validator_rejects_a_tool_declaring_two_aliased_params():
    """The validator must actually FIRE when an overlap exists, not vacuously pass.
    A synthetic probe declaring BOTH ``source`` and ``url`` — whose real VARIANTS
    ladders name each other — is the canonical overlap: ``source``'s ``url`` alias
    collides with the declared ``url`` (and vice-versa). ``_SYNTHETIC`` keeps the
    probe out of the real registry while still exercising the real validator."""

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

        def get_parameters(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    Keys.source: {"type": "string"},
                    Keys.url: {"type": "string"},
                    Keys.list: {"type": "string"},
                },
                "required": [],
            }

        def run(self, params: dict) -> ToolResult:
            return ToolResult.ok("x")

    with pytest.raises(RegistryOverlapError) as excinfo:
        RegistryInvariant().check_no_overlaps([_OverlapProbe()])

    message = str(excinfo.value)
    assert "source" in message and "url" in message, message
    assert "collides" in message, message


# ── First-write-wins collision: both ``source`` and ``url`` present for read ──────


def test_read_collision_first_key_wins_and_warning_emitted(db, tmp_path, caplog):
    """When a model sends BOTH ``source`` and ``url`` to ``read``, both resolve to
    the same canonical ``source``.  First dict-iteration entry wins; the loser's
    VALUE is dropped entirely.  Proven by a DOWNSTREAM discriminator: the first key
    carries ``http://localhost`` (SSRF block → ``private-or-internal-url-blocked``)
    and the second carries a real file path that would succeed.  The SSRF error
    proves the first value survived; ``[read(status=success`` NOT in result proves
    the file value was dropped.  The warning log proves the collision was surfaced."""
    mp = _chat_mp(db, allow=("read",))
    real_file = tmp_path / "winner.txt"
    real_file.write_text("content if the wrong key won", encoding="utf-8")

    import logging

    with caplog.at_level(logging.WARNING, logger="abilities._params"):
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
    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert warn_messages, "expected a KeyHealer collision warning, got none"
    combined = " ".join(str(m) for m in warn_messages)
    assert "source" in combined and "url" in combined, combined
    assert "source" in combined, combined   # the canonical that both map to


def test_read_collision_order_decides_winner_not_key_identity(db, tmp_path, caplog):
    """Flipping the dict order reverses the outcome — the file path is now FIRST so
    its value wins, and the ``http://localhost`` SSRF value is dropped.  The
    ``[read(status=success`` in the result (the file read succeeds) proves the
    iteration-order rule, not key identity, decides which value survives."""
    mp = _chat_mp(db, allow=("read",))
    real_file = tmp_path / "order_proof.txt"
    real_file.write_text("MARKER_CONTENT", encoding="utf-8")

    import logging

    with caplog.at_level(logging.WARNING, logger="abilities._params"):
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
    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert warn_messages, "expected a KeyHealer collision warning, got none"
    combined = " ".join(str(m) for m in warn_messages)
    assert "source" in combined and "url" in combined, combined
