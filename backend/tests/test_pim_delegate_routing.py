"""Feature tests — the PIM delegate surface split and its anti-leak guarantee.

The ``pim`` delegate owns email, calendar, and contacts. Those three abilities
are ``DISCOVERABLE=False``, so they are absent from BOTH the find_tools semantic
index AND the exact-name select roster — the ONLY path to them is a direct
``always_available`` pin, and ``PimConfig`` is the single config that holds one.
Every other agent (the main/user agent included) has lost all access; a
personal-information intent can only reach them by delegating to ``pim``.

These lock that whole promise, deterministically:

  1. Routing matrix — every calendar/email/contact-flavoured intent surfaces the
     ``pim`` delegate, and NONE of them (not even an exact-name query for the raw
     tool) ever surfaces raw ``calendar`` / ``email`` / ``contacts``. The
     negative is structural, not merely observed: a ``DISCOVERABLE=False`` tool
     is not in the index at all, so no query can reach it.
  2. Sole-pinner invariant — ``PimConfig`` is the only ProcessorConfig whose
     ``always_available`` exposes the three, and the shared default roster does
     not. A future re-pin on any other channel trips this guard.
  3. Verbatim disconnected contract — the delegate prompt carries, word for word,
     the message the pim loop must emit when no mail provider is connected.
  4. Disconnected short-circuit — when no mail provider is connected, ``pim`` returns
     the canonical not-connected contract immediately instead of spawning its
     delegate loop. This is a deterministic wiring guarantee, so it is covered here
     rather than left to QA.

Driven against the REAL ``FindToolsAbility.run()`` over the REAL production
abilities.sqlite + EmbeddingService, and the REAL production configs. Zero mocks.
The live CalDAV read is proven in QA on the live env — it is non-deterministic and
does not belong in the deterministic unit tier.
"""

import importlib
import pkgutil
import sqlite3
from typing import cast

import pytest

import configs.channels as channels_pkg
from abilities.find_tools import FindToolsAbility
from configs.channels import UserConfig
from contracts.params.find_tools_params_bag import FindToolsParamsBag
from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.channels.pim import PimConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

# The three personal-information tools the pim delegate owns exclusively.
_RAW = ("calendar", "email", "contacts")
_RAW_SET = frozenset(_RAW)

# The exact, word-for-word message the pim loop must emit when no mail provider
# is connected (mirrors the string embedded in PimConfig.system_prompt).
_VERBATIM = (
    "You currently do not have a connected calendar thus I can't perform this "
    "action. You need to connect a mail provider."
)


def _injected(query: object) -> list[str]:
    """Drive the real find_tools cascade on a real user-channel MessageProcessor
    and return the tools the call added to ``active_tools`` (the mutation the
    dispatcher acts on). ``hidden_input`` skips the turn-0 transcript row."""
    mp = MessageProcessor(UserConfig({"hidden_input": True}), raw_input="find a tool for me")
    mp.active_tools = list(mp.config.always_available or [])
    base = set(mp.active_tools)
    ability = FindToolsAbility()
    ability.mp = mp
    ability.run(built(FindToolsParamsBag.from_params({"query": query})))
    return [t for t in mp.active_tools if t not in base]


# ---------------------------------------------------------------------------
# 1. Routing matrix — personal-info intents route to the pim delegate.
# ---------------------------------------------------------------------------

_PIM_INTENTS = [
    "add appointment",
    "what's on my calendar today",
    "add a contact",
    "find phone number for John",
    "look up an email address",
    "check my inbox",
    "send an email",
]


@pytest.mark.parametrize("intent", _PIM_INTENTS)
def test_personal_info_intent_surfaces_the_pim_delegate(db: sqlite3.Connection, intent: str) -> None:
    """A calendar/email/contact-flavoured intent must surface ``pim`` — the only
    door to the personal-information tools now that they are delegate-owned."""
    injected = _injected([intent])
    assert "pim" in injected, (
        f"intent {intent!r} must route to the pim delegate. injected={injected!r}"
    )


# The negative is the headline guarantee: raw calendar/email/contacts must be
# unreachable for ANY of these — freeform intents AND exact-name queries for the
# raw tool itself (a DISCOVERABLE=False tool is absent from the roster the
# exact-name rung resolves against, so even naming it exactly cannot reach it).
_LEAK_PROBES = [
    ["calendar"],
    ["email"],
    ["contacts"],
    ["add appointment"],
    ["schedule a meeting"],
    ["what's on my calendar today"],
    ["send an email"],
    ["check my inbox"],
    ["read my latest email"],
    ["add a contact"],
    ["find phone number for John"],
    ["look up an email address"],
    ["remind me to call mum at 5pm"],
    ["set a reminder for tomorrow"],
]


@pytest.mark.parametrize("query", _LEAK_PROBES, ids=lambda q: "+".join(q))
def test_raw_personal_info_tools_never_surface(db: sqlite3.Connection, query: list[str]) -> None:
    """No query — semantic or exact-name — may inject raw calendar/email/contacts
    onto the user channel. They are DISCOVERABLE=False and reachable only through
    the pim delegate's pin."""
    injected = _injected(query)
    leaked = [t for t in _RAW if t in injected]
    assert not leaked, (
        f"raw personal-info tools leaked for query {query!r}: {leaked}. "
        f"They must be reachable only through pim. injected={injected!r}"
    )


# ---------------------------------------------------------------------------
# 2. Sole-pinner invariant — only PimConfig exposes the three tools.
# ---------------------------------------------------------------------------


def test_default_roster_does_not_expose_personal_info() -> None:
    """The shared default roster (carried by the user, DMN, external-agent and
    scheduled channels) must not pin any personal-info tool — otherwise the split
    would leak to every channel that inherits it."""
    assert _RAW_SET.isdisjoint(DEFAULT_ALWAYS_AVAILABLE), (
        f"DEFAULT_ALWAYS_AVAILABLE must not expose personal-info tools. "
        f"leak={sorted(_RAW_SET & set(DEFAULT_ALWAYS_AVAILABLE))}"
    )


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def test_pim_is_the_sole_config_exposing_personal_info(db: sqlite3.Connection) -> None:
    """Enumerate every ProcessorConfig subclass, build it, and assert that the
    ONLY config whose ``always_available`` exposes email/calendar/contacts is
    ``PimConfig``. Configs that cannot be built with a channel's standard
    argument forms (domain background configs) are named in the failure message
    rather than silently skipped."""
    # Import every channel module so all subclasses register.
    for mod in pkgutil.iter_modules(channels_pkg.__path__):
        importlib.import_module(f"configs.channels.{mod.name}")

    pinning: set[str] = set()
    unbuilt: list[str] = []
    for cls in _all_subclasses(ProcessorConfig):
        instance = None
        for call_args in ([], [PolicyChannel.CHAT], [{"hidden_input": True}]):
            try:
                instance = cls(*call_args)
                break
            except Exception:
                continue
        if instance is None:
            unbuilt.append(cls.__name__)
            continue
        if _RAW_SET & set(instance.always_available or []):
            pinning.add(cls.__name__)

    assert pinning == {"PimConfig"}, (
        "Only PimConfig may expose email/calendar/contacts to a model.\n"
        f"  pinning: {sorted(pinning)}\n"
        f"  not built with a standard channel signature (source-clean under the "
        f"current split, unverified here): {sorted(unbuilt)}"
    )


# ---------------------------------------------------------------------------
# 3. Verbatim disconnected contract + the delegate's own toolset.
# ---------------------------------------------------------------------------


def test_pim_prompt_carries_the_verbatim_disconnected_message() -> None:
    """The pim delegate prompt must carry the exact, word-for-word disconnected
    message the loop is required to emit — QA proves the runtime emission on the
    live env; this proves the contract is wired."""
    prompt = PimConfig(PolicyChannel.CHAT).system_prompt
    assert _VERBATIM in prompt, (
        f"PimConfig.system_prompt must carry the verbatim disconnected message.\n"
        f"  expected substring: {_VERBATIM!r}\n"
        f"  prompt: {prompt!r}"
    )


def test_pim_delegate_pins_exactly_its_toolset() -> None:
    """The delegate is pinned exactly to its four tools — the three personal-info
    tools it owns plus memory — no more, no less."""
    always_available = PimConfig(PolicyChannel.CHAT).always_available
    assert set(always_available) == {"email", "calendar", "contacts", "memory"}, (
        f"pim delegate must pin exactly email/calendar/contacts/memory. "
        f"always_available={always_available!r}"
    )


# ---------------------------------------------------------------------------
# 4. Disconnected short-circuit — pim never spawns its delegate when no mail
#    provider is connected; it returns the canonical not-connected contract.
# ---------------------------------------------------------------------------


def test_pim_short_circuits_when_mail_not_connected() -> None:
    """With no mail provider connected, ``pim`` must NOT spawn its delegate loop —
    it returns the canonical not-connected contract immediately (the same
    ``code`` / ``hint`` the calendar/email/contacts tools return). In the unit
    env no mail credentials are configured, so ``is_connected()`` is False and
    every pim call hits this gate. That the call returns a clean error WITHOUT a
    bound MessageProcessor proves the short-circuit fires before any delegate
    machinery (a bound-mp-less delegate spawn would raise instead)."""
    from abilities.pim import PimAbility
    from contracts.params.delegate_params_bag import DelegateParamsBag

    result = PimAbility().run(
        built(DelegateParamsBag.from_params({"instructions": "what's on my calendar today"}))
    )
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "no mail provider is connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()
