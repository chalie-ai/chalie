"""ParamBag — base class of the per-ability typed input bags.

One frozen dataclass per ability (``contracts/params/<tool>_params_bag.py``),
one typed field per parameter. A multi-action ability declares one dataclass
per ACTION under a plain router class that owns the fan-out (see
``from_params``); the router is the ability's boundary type, the leaves are
what actually flow. The dict-facing boundary is the ``from_params``
factory: it validates the params dict — whose KEYS the dispatch seam has
already healed (:class:`~services.key_healer.KeyHealer` runs in
``DispatchService._prepare`` before any bag is built, so a bag never sees
``url`` where it declared ``source``) — and hands proven values to the
generated field-list constructor. A bag that exists is therefore always fully
populated and valid; ``run()`` can never see anything else. A missing or
invalid parameter raises :class:`~exceptions.exception.ToolParamError` inside
``from_params``, which the dispatcher's existing handler renders canonically
(``code`` / ``hint`` / ``valid``).

Each validator below is simultaneously the runtime gate and the static type of
the field it fills: ``source=cls.require_str(params, Keys.source)`` satisfies a
``source: str`` field (non-optional) under mypy strict;
``max_chars=cls.clamp_int(...)`` satisfies ``int``. Validators take
``(params, key)`` — not a pre-extracted value — so a failure can echo the keys
the model DID send: the self-correction diagnostic a weak model needs to stop
looping on the same bad call.

Bags are ``@dataclass(frozen=True, slots=True)``: immutable, closed attribute
set, and the class-level field declarations stay runtime-readable
(``__annotations__``) for the schema slice to come. The empty ``__slots__``
here is load-bearing — a slotted subclass of a dict-carrying base silently
regains ``__dict__`` and the boundary evaporates.

Deliberate scope — the INPUT side only. The LLM-facing schema
(``get_parameters()``) and the ``ACTION_REQUIRED`` pre-gate of unmigrated
abilities are separate concerns and stay where they are.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from exceptions import ToolParamError


class ParamBag(ABC):
    """Factory contract plus the shared validator vocabulary; subclasses fill
    every field in ``from_params`` through exactly one validator — the
    validator chosen declares both the requirement and the resulting type."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, object]) -> ParamBag:
        """The one dict-facing door: validate ``params`` and return a fully
        populated bag. The dispatcher calls this through ``type[ParamBag]``;
        mypy's ``type-abstract`` check makes a bag that forgot to implement it
        unassignable to ``Ability.PARAMS``.

        Single-shot bags narrow the return to ``Self`` and construct
        themselves. A multi-action ability declares a ROUTER bag instead: its
        ``from_params`` reads ``action`` and fans out to the matching
        per-action subclass's ``from_params`` (which receives the full params
        dict and simply ignores ``action``) — which is why THIS signature
        promises ``ParamBag``, not ``Self``: the router legitimately returns
        its subclasses. The router class stays the ability's ``run()``
        annotation; the instance is always a leaf."""

    @staticmethod
    def require_str(params: dict[str, object], key: str) -> str:
        """Mandatory non-blank string → the stripped value, or ``missing-params``."""
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            received = "|".join(params.keys()) or "none"
            raise ToolParamError(
                f"Required parameter '{key}' is missing.",
                code="missing-params",
                hint=f"pass '{key}' (received: {received})",
                valid=(key,),
            )
        return value.strip()

    @staticmethod
    def opt_str(params: dict[str, object], key: str) -> str | None:
        """Optional string → the value, ``None`` when absent, ``invalid-param``
        on any other type. The typed replacement for the ``cast("str | None",
        params.get(...))`` idiom, which asserted str-ness without checking it."""
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ToolParamError(
                f"'{key}' must be text.",
                code="invalid-param",
                hint=f"pass '{key}' as a string.",
            )
        return value

    @staticmethod
    def str_default(params: dict[str, object], key: str, *, default: str) -> str:
        """Optional string with a default when absent; ``invalid-param`` on any
        other type. An explicit empty string is kept, not defaulted — some
        handlers treat ``""`` as a sentinel and the bag must not decide for them."""
        value = params.get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ToolParamError(
                f"'{key}' must be text.",
                code="invalid-param",
                hint=f"pass '{key}' as a string.",
            )
        return value

    @staticmethod
    def flag(params: dict[str, object], key: str) -> bool:
        """Truthiness of the raw value — for framework-internal marker params
        (``{"_auto": True}``), never model-facing inputs: a model's mistyped
        boolean deserves ``invalid-param``, a framework marker just needs truth."""
        return bool(params.get(key))

    @staticmethod
    def clamp_int(params: dict[str, object], key: str, *, default: int, lo: int, hi: int) -> int:
        """Optional integer with a default, clamped into ``[lo, hi]``.

        A non-numeric value → ``invalid-param``. Booleans land there too (via the
        failing ``int("True")`` parse): a model emitting ``true`` for a count is a
        mistake to correct, not a 1 — the guard timer.py used to hand-roll, held
        here once for every bag."""
        value = params.get(key)
        if value is None:
            return default
        if isinstance(value, int) and not isinstance(value, bool):
            numeric = value
        elif isinstance(value, float):
            numeric = int(value)
        else:
            try:
                numeric = int(str(value))
            except ValueError:
                raise ToolParamError(
                    f"'{key}' must be a number.",
                    code="invalid-param",
                    hint=f"pass a number between {lo} and {hi}.",
                ) from None
        return max(lo, min(hi, numeric))
