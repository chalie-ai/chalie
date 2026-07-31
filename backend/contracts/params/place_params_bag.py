"""PlaceParamsBag — the typed input contract of the ``place`` ability.

The multi-action shape: :class:`PlaceParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag

_ACTIONS = ("save", "list", "get", "delete")


class PlaceParamsBag(ParamBag):
    """Router and boundary type: ``PlaceAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> PlaceParamsBag | ToolResult:
        match params.get(Keys.action):
            case "save":
                return PlaceSaveParams.from_params(params)
            case "list":
                return PlaceListParams.from_params(params)
            case "get":
                return PlaceGetParams.from_params(params)
            case "delete":
                return PlaceDeleteParams.from_params(params)
            case unknown:
                return ToolResult.err(
                    f"Unknown place action: {unknown!r}.",
                    code="unknown-action",
                    hint="choose one of the valid actions below.",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class PlaceSaveParams(PlaceParamsBag):
    """``name`` is required and stripped; the legacy handler also lowercased it
    before any comparison, so the bag preserves that contract here."""

    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        name = cls.require_str(params, Keys.name_)
        if isinstance(name, ToolResult):
            return name
        return cls(name=name.lower())


@dataclass(frozen=True, slots=True)
class PlaceListParams(PlaceParamsBag):
    """``list`` reads no params at all — the router validates action alone."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()


@dataclass(frozen=True, slots=True)
class PlaceGetParams(PlaceParamsBag):
    """``name`` is required and stripped; lowercased to match the legacy
    comparison path."""

    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        name = cls.require_str(params, Keys.name_)
        if isinstance(name, ToolResult):
            return name
        return cls(name=name.lower())


@dataclass(frozen=True, slots=True)
class PlaceDeleteParams(PlaceParamsBag):
    """``name`` is required and stripped; lowercased to match the legacy
    comparison path."""

    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        name = cls.require_str(params, Keys.name_)
        if isinstance(name, ToolResult):
            return name
        return cls(name=name.lower())
