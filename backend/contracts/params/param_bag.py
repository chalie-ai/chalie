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
populated and valid; ``run()`` can never see anything else.

A missing or invalid parameter is an error :class:`~abilities._result.ToolResult`
RETURNED from ``from_params`` — never an exception. The bag builds the
model-facing self-correction payload (``code`` / ``hint`` / ``valid``) at the
failure site, where it knows exactly which field of which action broke, and the
dispatch seam passes it to the wire untouched. There is no throw-then-catch
anywhere on this path: value or error, both are return values.

Each validator below returns either the proven, typed value or the error
``ToolResult``. The calling ``from_params`` guards each field once —
``if isinstance(source := cls.require_str(params, Keys.source), ToolResult):
return source`` — and past the guard the name carries the field's exact static
type under mypy strict, so the closing constructor call stays fully
type-checked. Validators take ``(params, key)`` — not a pre-extracted value —
so a failure can echo the keys the model DID send: the self-correction
diagnostic a weak model needs to stop looping on the same bad call.

Bags are ``@dataclass(frozen=True, slots=True)``: immutable, closed attribute
set, and the class-level field declarations stay runtime-readable
(``__annotations__``) for the schema slice to come. The empty ``__slots__``
here is load-bearing — a slotted subclass of a dict-carrying base silently
regains ``__dict__`` and the boundary evaporates.

Deliberate scope — the INPUT side only. The LLM-facing schema
(``get_parameters()``) and the ``ACTION_REQUIRED`` pre-gate are separate
concerns and stay where they are.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from abilities._result import ToolResult


class ParamBag(ABC):
    """Factory contract plus the shared validator vocabulary; subclasses fill
    every field in ``from_params`` through exactly one validator — the
    validator chosen declares both the requirement and the resulting type."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, object]) -> ParamBag | ToolResult:
        """The one dict-facing door: validate ``params`` and return either a
        fully populated bag or the error ``ToolResult`` describing the first
        bad parameter. The dispatcher calls this through ``type[ParamBag]``
        and isinstance-checks the result; mypy's ``type-abstract`` check makes
        a bag that forgot to implement it unassignable to ``Ability.PARAMS``.

        Single-shot bags narrow the bag side of the return to ``Self`` and
        construct themselves. A multi-action ability declares a ROUTER bag
        instead: its ``from_params`` reads ``action`` and fans out to the
        matching per-action subclass's ``from_params`` (which receives the
        full params dict and simply ignores ``action``) — which is why THIS
        signature promises ``ParamBag``, not ``Self``: the router legitimately
        returns its subclasses (or their errors). The router class stays the
        ability's ``run()`` annotation; the instance is always a leaf."""

    @staticmethod
    def require_str(params: dict[str, object], key: str) -> str | ToolResult:
        """Mandatory non-blank string → the stripped value, or ``missing-params``."""
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            received = "|".join(params.keys()) or "none"
            return ToolResult.err(
                f"Required parameter '{key}' is missing.",
                code="missing-params",
                hint=f"pass '{key}' (received: {received})",
                valid=(key,),
            )
        return value.strip()

    @staticmethod
    def require_str_list(params: dict[str, object], key: str) -> list[str] | ToolResult:
        """Mandatory non-empty list of strings → a fresh list, ``missing-params``
        when absent, empty, or not a list; ``invalid-param`` on a non-string
        element. The copy keeps the frozen bag from aliasing the caller's dict."""
        value = params.get(key)
        if not isinstance(value, list) or not value:
            received = "|".join(params.keys()) or "none"
            return ToolResult.err(
                f"Required parameter '{key}' must be a non-empty list.",
                code="missing-params",
                hint=f"pass '{key}' as a list of strings (received: {received})",
                valid=(key,),
            )
        items: list[str] = []
        for i, item in enumerate(value):
            if not isinstance(item, str):
                return ToolResult.err(
                    f"'{key}' item at index {i} must be a string.",
                    code="invalid-param",
                    hint=f"pass '{key}' as a list of strings.",
                )
            items.append(item)
        return items

    @staticmethod
    def opt_str(params: dict[str, object], key: str) -> str | None | ToolResult:
        """Optional string → the value, ``None`` when absent, ``invalid-param``
        on any other type. The typed replacement for the ``cast("str | None",
        params.get(...))`` idiom, which asserted str-ness without checking it."""
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            return ToolResult.err(
                f"'{key}' must be text.",
                code="invalid-param",
                hint=f"pass '{key}' as a string.",
            )
        return value

    @staticmethod
    def opt_int(params: dict[str, object], key: str, *, lo: int) -> int | None | ToolResult:
        """Optional integer with a floor — ``None`` when absent, ``invalid-param``
        on any non-integer value (booleans included), or when the value sits
        below ``lo``. Strict: no float-string coercions here — line counters
        deserve an explicit integer so a stray ``"5"`` produces a useful error
        hint instead of sneaking in as 5."""
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            return ToolResult.err(
                f"'{key}' must be an integer.",
                code="invalid-param",
                hint=f"pass '{key}' as an integer ≥ {lo}.",
            )
        if value < lo:
            return ToolResult.err(
                f"'{key}' must be an integer ≥ {lo}.",
                code="invalid-param",
                hint=f"got {value}; pass at least {lo}.",
            )
        return value

    @staticmethod
    def str_default(params: dict[str, object], key: str, *, default: str) -> str | ToolResult:
        """Optional string with a default when absent; ``invalid-param`` on any
        other type. An explicit empty string is kept, not defaulted — some
        handlers treat ``""`` as a sentinel and the bag must not decide for them."""
        value = params.get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            return ToolResult.err(
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
    def bool_default(params: dict[str, object], key: str, *, default: bool) -> bool | ToolResult:
        """Optional boolean with a default when absent or ``None``.

        A real ``bool`` passes through; the strings ``"true"`` / ``"false"``
        (case-insensitive, whitespace-stripped) parse to ``True`` / ``False``;
        anything else → ``invalid-param``. Pure return — never raises.
        """
        value = params.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped == "true":
                return True
            if stripped == "false":
                return False
        return ToolResult.err(
            f"'{key}' must be a boolean.",
            code="invalid-param",
            hint=f"pass '{key}' as true or false.",
        )

    @staticmethod
    def clamp_int(params: dict[str, object], key: str, *, default: int, lo: int, hi: int) -> int | ToolResult:
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
                return ToolResult.err(
                    f"'{key}' must be a number.",
                    code="invalid-param",
                    hint=f"pass a number between {lo} and {hi}.",
                )
        return max(lo, min(hi, numeric))
