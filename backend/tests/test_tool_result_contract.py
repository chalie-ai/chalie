# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the ToolResult contract (TKT-882).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``mp``-shaped
context, the real ``AbilityRegistry`` resolution, the real ``PolicyManager.wrap``
gate, the real ``Ability.run``, and the real ``ActTrail`` write (the ``db``
fixture binds ``ActTrail()`` to a real SQLite database).

The contract under test:
  * ``run()`` returning anything that is not a ``ToolResult`` hard-fails as
    ``code=non-canonical-result`` (legacy dict / plain str / None).
  * ``ToolResult.ok``/``err`` render the sealed tag envelope (dict body → compact
    JSON; str body → verbatim; meta in the open tag; hint/valid lines on errors).
  * ``ACTION_REQUIRED`` pre-validation reports an unknown action with ``valid=``
    and ALL missing params in ONE ``missing-params`` error.
  * ``Ability.param`` resolves aliases / validates choices / clamps numerics and
    raises ``ToolParamError`` which the dispatcher renders canonically.
  * Rich-media: an ``ok(rich=…)`` result dispatched on a ``broadcast_to=='user'``
    mp carries the ordinal + the card instruction; on a non-user channel it does
    NOT (the entire card path is gated on user broadcast).
"""

import threading

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._result import ToolResult
from configs.channels import DmnConfig, UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "do a thing"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (policy channel + emitter gate + broadcast_to) and
    ``uid`` (the transcript anchor the trail records against)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


# ── Throwaway abilities — they self-register via Ability.__init_subclass__ +
#    AbilityRegistry's subclass walk.  The fixture resets the registry cache so
#    the live registry picks them up, then resets again to drop them. ──


class _BadDictAbility(Ability):
    def get_name(self): return "_tr_baddict"
    def get_summary(self): return "throwaway: returns a legacy dict"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params): return {"status": "ok", "text": "legacy"}


class _BadStrAbility(Ability):
    def get_name(self): return "_tr_badstr"
    def get_summary(self): return "throwaway: returns a plain str"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params): return "just a string"


class _BadNoneAbility(Ability):
    def get_name(self): return "_tr_badnone"
    def get_summary(self): return "throwaway: returns None"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params): return None


class _OkDictAbility(Ability):
    def get_name(self): return "_tr_okdict"
    def get_summary(self): return "throwaway: ok with dict body + meta"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params):
        return ToolResult.ok({"location": "London", "temp_c": 18}, count=1)


class _OkStrAbility(Ability):
    def get_name(self): return "_tr_okstr"
    def get_summary(self): return "throwaway: ok with prose body"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params):
        return ToolResult.ok("plain prose verbatim")


class _ErrAbility(Ability):
    def get_name(self): return "_tr_err"
    def get_summary(self): return "throwaway: err with code/hint/valid"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params):
        return ToolResult.err(
            "dtstart '2026-13-40' is not a valid datetime.",
            code="invalid-param",
            hint="pass ISO-8601 e.g. 2026-06-10T15:00:00+02:00.",
            valid=("create", "update", "delete", "list"),
        )


class _ActionMapAbility(Ability):
    ACTION_REQUIRED = {"create": ("name", "value"), "delete": ("id",)}

    def get_name(self): return "_tr_actionmap"
    def get_summary(self): return "throwaway: ACTION_REQUIRED pre-validation"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params):
        return ToolResult.ok("ran")


class _ParamAbility(Ability):
    def get_name(self): return "_tr_param"
    def get_summary(self): return "throwaway: exercises Ability.param"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params):
        source = self.param(params, "source", aliases=("url", "path"), required=True)
        limit = self.param(params, "limit", default=10, clamp=(1, 100))
        mode = self.param(params, "mode", default="fast", choices=("fast", "slow"))
        return ToolResult.ok({"source": source, "limit": limit, "mode": mode})


class _RichAbility(Ability):
    def get_name(self): return "_tr_rich"
    def get_summary(self): return "throwaway: sets rich payload"
    def get_examples(self): return ["a", "b", "c", "d", "e", "f"]
    def get_search_tooltip(self): return "x"
    def get_parameters(self): return {"type": "object", "properties": {}, "required": []}
    def run(self, params):
        # Mirrors the production exemplar (weather.py): the rich card payload IS
        # the structured body — on a user turn the renderer surfaces that payload
        # as the card head, then the span instruction trailer.
        payload = {"city": "Paris", "temp_c": 14}
        return ToolResult.ok(payload, rich=payload)


# Bare tool names plus the action-qualified permissions the suite dispatches —
# the gate keys on ``<tool>.<action>`` when an action is present, so the
# ACTION_REQUIRED cases must allow those specific permissions to reach run().
_THROWAWAY_TOOLS = (
    "_tr_baddict", "_tr_badstr", "_tr_badnone", "_tr_okdict", "_tr_okstr",
    "_tr_err", "_tr_actionmap", "_tr_actionmap.bogus", "_tr_actionmap.create",
    "_tr_param", "_tr_rich",
)


@pytest.fixture
def registered(db):
    """Rebuild the registry so it picks up the throwaway abilities defined in
    this module (they self-register as Ability subclasses), then reset it again
    so the live registry is left exactly as found.

    The throwaway tools carry no seed row, so on a no-human channel
    (SUBCONSCIOUS) the gate would default-deny them before ``run()`` is reached —
    masking the very contract under test. We grant them ``allow`` through the
    real ``PolicyManager.upsert`` (the production policy-write entry point, no
    mock) on every channel the suite dispatches them on, so the dispatcher's
    render/hard-fail path is what's exercised, not the policy gate."""
    from services.database_service import get_shared_db_service
    from services.policy_manager import PolicyManager
    pm = PolicyManager(get_shared_db_service())
    for tool in _THROWAWAY_TOOLS:
        for channel in ("subconscious", "chat"):
            pm.upsert(channel, tool, "allow")

    from abilities import _registry as _reg_mod
    _reg_mod._reset_for_tests()  # noqa: SLF001
    _reg_mod._get_registry()     # noqa: SLF001  — force rebuild now (with our classes)
    try:
        yield
    finally:
        _reg_mod._reset_for_tests()  # noqa: SLF001


# ── (a) hard-fail chokepoint: legacy dict / str / None → non-canonical-result ──

@pytest.mark.parametrize("tool", ["_tr_baddict", "_tr_badstr", "_tr_badnone"])
def test_non_toolresult_return_hard_fails(db, registered, tool):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch(tool, {})
    assert out.startswith(f"[{tool}(status=error, code=non-canonical-result")
    assert tool in out
    assert out.rstrip().endswith(f"[end:{tool}]")
    # The act-trail recorded the same hard-fail envelope (ok flag is False).
    rows = ActTrail().fetch_by_transcript_id(mp.uid)
    assert "non-canonical-result" in rows[0]["result"]


# ── (b) success render: dict body → compact JSON envelope; str body verbatim ──

def test_ok_dict_body_renders_compact_json_with_meta(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_okdict", {})
    assert out == (
        '[_tr_okdict(status=success, count=1)]\n'
        '{"location":"London","temp_c":18}\n'
        '[end:_tr_okdict]'
    )


def test_ok_str_body_renders_verbatim(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_okstr", {})
    assert out == (
        '[_tr_okstr(status=success)]\n'
        'plain prose verbatim\n'
        '[end:_tr_okstr]'
    )


# ── (c) error render with code/hint/valid lines ──────────────────────────────

def test_err_render_carries_code_hint_valid(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_err", {})
    assert out == (
        "[_tr_err(status=error, code=invalid-param)]\n"
        "dtstart '2026-13-40' is not a valid datetime.\n"
        "hint: pass ISO-8601 e.g. 2026-06-10T15:00:00+02:00.\n"
        "valid: create | update | delete | list\n"
        "[end:_tr_err]"
    )


# ── (d) ACTION_REQUIRED pre-validation ───────────────────────────────────────

def test_action_required_unknown_action_lists_valid(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_actionmap", {"action": "bogus"})
    assert "status=error" in out
    assert "code=unknown-action" in out
    assert "valid: create | delete" in out


def test_action_required_missing_params_reported_in_one_error(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_actionmap", {"action": "create"})
    assert "code=missing-params" in out
    # BOTH missing params named in the single error.
    assert "name" in out and "value" in out


# ── (e) Ability.param aliases / choices / clamp through the real chokepoint ───

def test_param_alias_clamp_choice_happy_path(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_param", {"url": "x", "limit": 999})
    # alias url→source; limit clamped 999→100; mode default 'fast'.
    assert '"source":"x"' in out
    assert '"limit":100' in out
    assert '"mode":"fast"' in out


def test_param_required_missing_raises_canonical_error(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_param", {})
    assert "status=error" in out
    assert "code=invalid-param" in out


def test_param_bad_choice_raises_canonical_error(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_param", {"url": "x", "mode": "warp"})
    assert "status=error" in out
    assert "valid: fast | slow" in out


# ── (f) rich exemplar: ordinal + card instruction on user; absent off-user ────

def test_rich_payload_injected_on_user_broadcast(db, registered):
    mp = _MP(_seed_transcript(db, "user"), UserConfig())
    out = ToolDispatcher(mp).dispatch("_tr_rich", {})
    # The rich block round-trips: JSON head, blank line, then the card
    # instruction carrying the ordinal-keyed span tag.
    assert '{"city":"Paris","temp_c":14}' in out
    assert "<span id='_tr_rich_1'>" in out


def test_rich_payload_absent_on_non_user_broadcast(db, registered):
    mp = _MP(_seed_transcript(db, "dmn"), DmnConfig())
    out = ToolDispatcher(mp).dispatch("_tr_rich", {})
    assert "_tr_rich_1" not in out
    assert "<span" not in out
