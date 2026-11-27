"""BrowserParamsBag — the typed input contract of the ``browser`` ability.

The multi-action shape: :class:`BrowserParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently. Each leaf's ``ACTION`` is the
one place its action's name is written: routing and the list of valid actions
are derived from the leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


class BrowserParamsBag(ParamBag):
    """Router and boundary type: ``BrowserAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    ACTION: ClassVar[str]  # the action a leaf carries; the router itself carries none

    @classmethod
    def from_params(cls, params: dict[str, object]) -> BrowserParamsBag | ToolResult:
        action = params.get(Keys.action)
        leaf = _BY_ACTION.get(action) if isinstance(action, str) else None
        if leaf is None:
            return ToolResult.err(
                f"Unknown browser action: {action!r}.",
                code="unknown-action",
                valid=_ACTIONS,
            )
        return leaf.from_params(params)


@dataclass(frozen=True, slots=True)
class BrowserOpenParams(BrowserParamsBag):
    """``url`` is required; everything else is ignored by 'open'."""

    ACTION: ClassVar[str] = "open"

    url: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        url = cls.require_str(params, Keys.url)
        if isinstance(url, ToolResult):
            return url
        return cls(url=url)


@dataclass(frozen=True, slots=True)
class BrowserReadParams(BrowserParamsBag):
    """``section`` is optional — 'read' returns the whole page when absent; a
    blank or whitespace-only section is folded to ``None`` (whole page), the
    same drop the old forwarding loop applied."""

    ACTION: ClassVar[str] = "read"

    section: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        raw = cls.opt_str(params, Keys.section)
        if isinstance(raw, ToolResult):
            return raw
        section = raw.strip() if raw is not None else None
        return cls(section=section or None)


@dataclass(frozen=True, slots=True)
class BrowserFindParams(BrowserParamsBag):
    """``query`` is required — 'find' searches for text on the page."""

    ACTION: ClassVar[str] = "find"

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.require_str(params, Keys.query)
        if isinstance(query, ToolResult):
            return query
        return cls(query=query)


@dataclass(frozen=True, slots=True)
class BrowserClickParams(BrowserParamsBag):
    """``target`` is required — the visible text of the element to click."""

    ACTION: ClassVar[str] = "click"

    target: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = cls.require_str(params, Keys.target)
        if isinstance(target, ToolResult):
            return target
        return cls(target=target)


@dataclass(frozen=True, slots=True)
class BrowserFillParams(BrowserParamsBag):
    """``target`` and ``value`` are required — fill a form field with text."""

    ACTION: ClassVar[str] = "fill"

    target: str
    value: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = cls.require_str(params, Keys.target)
        if isinstance(target, ToolResult):
            return target
        value = cls.require_str(params, Keys.value_)
        if isinstance(value, ToolResult):
            return value
        return cls(target=target, value=value)


@dataclass(frozen=True, slots=True)
class BrowserSelectParams(BrowserParamsBag):
    """``target`` and ``value`` are required — choose an option from a dropdown."""

    ACTION: ClassVar[str] = "select"

    target: str
    value: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = cls.require_str(params, Keys.target)
        if isinstance(target, ToolResult):
            return target
        value = cls.require_str(params, Keys.value_)
        if isinstance(value, ToolResult):
            return value
        return cls(target=target, value=value)


@dataclass(frozen=True, slots=True)
class BrowserScrollParams(BrowserParamsBag):
    """``direction`` is required — 'up' or 'down'."""

    ACTION: ClassVar[str] = "scroll"

    direction: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        direction = cls.require_str(params, Keys.direction)
        if isinstance(direction, ToolResult):
            return direction
        return cls(direction=direction)


@dataclass(frozen=True, slots=True)
class BrowserBackParams(BrowserParamsBag):
    """No fields — 'back' just navigates to the previous page."""

    ACTION: ClassVar[str] = "back"

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()


@dataclass(frozen=True, slots=True)
class BrowserScreenshotParams(BrowserParamsBag):
    """No fields — 'screenshot' captures the current page."""

    ACTION: ClassVar[str] = "screenshot"

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()


@dataclass(frozen=True, slots=True)
class BrowserStyleParams(BrowserParamsBag):
    """``target`` is required — the visible text of the element whose styles to inspect."""

    ACTION: ClassVar[str] = "style"

    target: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = cls.require_str(params, Keys.target)
        if isinstance(target, ToolResult):
            return target
        return cls(target=target)


@dataclass(frozen=True, slots=True)
class BrowserHandoffParams(BrowserParamsBag):
    """``reason`` is required — what the user needs to do on the page before the run carries on."""

    ACTION: ClassVar[str] = "handoff"

    reason: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        reason = cls.require_str(params, Keys.reason)
        if isinstance(reason, ToolResult):
            return reason
        return cls(reason=reason)


_BY_ACTION: dict[str, type[BrowserParamsBag]] = {leaf.ACTION: leaf for leaf in (
    BrowserOpenParams, BrowserReadParams, BrowserFindParams, BrowserClickParams, BrowserFillParams,
    BrowserSelectParams, BrowserScrollParams, BrowserBackParams, BrowserScreenshotParams, BrowserStyleParams,
    BrowserHandoffParams,
)}
_ACTIONS = tuple(_BY_ACTION)
