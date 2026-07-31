"""MemoryParamsBag — the typed input contract of the ``memory`` ability.

The multi-action shape: :class:`MemoryParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness (``MemoryStoreParams.key`` is ``str``, not ``str | None``); a
leaf receives the full params dict and simply ignores ``action`` — routing is
the parent's job alone. ``run()`` narrows the router type back to a leaf via
``isinstance``, so a handler handed the wrong leaf fails loudly at the
attribute, never silently.

``auto`` on the recall leaf is the framework-internal ``_auto`` marker the
turn-0 seed sends (``controllers.message_processor._seed_turn_zero``) —
undeclared in the LLM schema, but it crosses the same seam, so the bag
carries it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag

#: The turn-0 seed marker. Framework-internal, so deliberately NOT a
#: :class:`Keys` member — Keys is the model-facing wire vocabulary and this
#: key must never appear in a schema.
_AUTO = "_auto"


class MemoryParamsBag(ParamBag):
    """Router and boundary type: ``MemoryAbility.run()`` is annotated with this
    class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> MemoryParamsBag | ToolResult:
        match params.get(Keys.action):
            case "store":
                return MemoryStoreParams.from_params(params)
            case "recall":
                return MemoryRecallParams.from_params(params)
            case "reflect":
                return MemoryReflectParams.from_params(params)
            case "forget":
                return MemoryForgetParams.from_params(params)
            case unknown:
                return ToolResult.err(
                    f"Unknown memory action: {unknown!r}.",
                    code="unknown-action",
                    valid=("store", "recall", "reflect", "forget"),
                )


@dataclass(frozen=True, slots=True)
class MemoryStoreParams(MemoryParamsBag):
    """``kind`` defaults to ``user_specific``; its allowed values are a store
    business rule (``VALID_KINDS``), so the handler judges it — the bag only
    guarantees a string. ``value`` keeps store's historical contract: any
    non-None value is stored via ``str()`` — a numeric fact ("salary": 50000)
    coerces, never rejects; its required-ness stays the handler's None-check."""

    key: str
    value: str | None
    kind: str
    replaces: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        value = params.get(Keys.value_)
        key = cls.require_str(params, Keys.key)
        if isinstance(key, ToolResult):
            return key
        kind = cls.str_default(params, Keys.kind, default="user_specific")
        if isinstance(kind, ToolResult):
            return kind
        replaces = cls.opt_str(params, Keys.replaces)
        if isinstance(replaces, ToolResult):
            return replaces
        return cls(
            key=key,
            value=None if value is None else str(value),
            kind=kind,
            replaces=replaces,
        )


@dataclass(frozen=True, slots=True)
class MemoryRecallParams(MemoryParamsBag):
    """``query`` and ``location`` are individually optional but at least one
    must be set — an OR the handler enforces (``code=no-query-or-location``):
    moving it here would strip the ``action`` meta from the error envelope."""

    query: str
    location: str
    auto: bool

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.str_default(params, Keys.query, default="")
        if isinstance(query, ToolResult):
            return query
        location = cls.str_default(params, Keys.location, default="")
        if isinstance(location, ToolResult):
            return location
        return cls(query=query, location=location, auto=cls.flag(params, _AUTO))


@dataclass(frozen=True, slots=True)
class MemoryReflectParams(MemoryParamsBag):
    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.require_str(params, Keys.query)
        if isinstance(query, ToolResult):
            return query
        return cls(query=query)


@dataclass(frozen=True, slots=True)
class MemoryForgetParams(MemoryParamsBag):
    """``value`` narrows a forget to one value of a multi-value key; ``kind``
    routes to the vertical — only the migrated two are forgettable, a handler
    rule (``code=kind-not-migrated``)."""

    key: str
    value: str | None
    kind: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        key = cls.require_str(params, Keys.key)
        if isinstance(key, ToolResult):
            return key
        value = cls.opt_str(params, Keys.value_)
        if isinstance(value, ToolResult):
            return value
        kind = cls.str_default(params, Keys.kind, default="user_specific")
        if isinstance(kind, ToolResult):
            return kind
        return cls(key=key, value=value, kind=kind)
