"""BrowserParamsBag — the typed input contract of the ``browser`` ability.

The multi-action shape: :class:`BrowserParamsBag` is a plain router — the
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

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

_ACTIONS = ("open", "read", "find", "click", "fill", "select", "scroll", "back", "screenshot", "style")


class BrowserParamsBag(ParamBag):
    """Router and boundary type: ``BrowserAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> BrowserParamsBag:
        match params.get(Keys.action):
            case "open":
                return BrowserOpenParams.from_params(params)
            case "read":
                return BrowserReadParams.from_params(params)
            case "find":
                return BrowserFindParams.from_params(params)
            case "click":
                return BrowserClickParams.from_params(params)
            case "fill":
                return BrowserFillParams.from_params(params)
            case "select":
                return BrowserSelectParams.from_params(params)
            case "scroll":
                return BrowserScrollParams.from_params(params)
            case "back":
                return BrowserBackParams.from_params(params)
            case "screenshot":
                return BrowserScreenshotParams.from_params(params)
            case "style":
                return BrowserStyleParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown browser action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class BrowserOpenParams(BrowserParamsBag):
    """``url`` is required; everything else is ignored by 'open'."""

    url: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(url=cls.require_str(params, Keys.url))


@dataclass(frozen=True, slots=True)
class BrowserReadParams(BrowserParamsBag):
    """``section`` is optional — 'read' returns the whole page when absent; a
    blank or whitespace-only section is folded to ``None`` (whole page), the
    same drop the old forwarding loop applied."""

    section: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        raw = cls.opt_str(params, Keys.section)
        section = raw.strip() if raw is not None else None
        return cls(section=section or None)


@dataclass(frozen=True, slots=True)
class BrowserFindParams(BrowserParamsBag):
    """``query`` is required — 'find' searches for text on the page."""

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(query=cls.require_str(params, Keys.query))


@dataclass(frozen=True, slots=True)
class BrowserClickParams(BrowserParamsBag):
    """``target`` is required — the visible text of the element to click."""

    target: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(target=cls.require_str(params, Keys.target))


@dataclass(frozen=True, slots=True)
class BrowserFillParams(BrowserParamsBag):
    """``target`` and ``value`` are required — fill a form field with text."""

    target: str
    value: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            target=cls.require_str(params, Keys.target),
            value=cls.require_str(params, Keys.value_),
        )


@dataclass(frozen=True, slots=True)
class BrowserSelectParams(BrowserParamsBag):
    """``target`` and ``value`` are required — choose an option from a dropdown."""

    target: str
    value: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            target=cls.require_str(params, Keys.target),
            value=cls.require_str(params, Keys.value_),
        )


@dataclass(frozen=True, slots=True)
class BrowserScrollParams(BrowserParamsBag):
    """``direction`` is required — 'up' or 'down'."""

    direction: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(direction=cls.require_str(params, Keys.direction))


@dataclass(frozen=True, slots=True)
class BrowserBackParams(BrowserParamsBag):
    """No fields — 'back' just navigates to the previous page."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls()


@dataclass(frozen=True, slots=True)
class BrowserScreenshotParams(BrowserParamsBag):
    """No fields — 'screenshot' captures the current page."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls()


@dataclass(frozen=True, slots=True)
class BrowserStyleParams(BrowserParamsBag):
    """``target`` is required — the visible text of the element whose styles to inspect."""

    target: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(target=cls.require_str(params, Keys.target))
