"""SearchFilesParamsBag — the typed input contract of the ``search_files`` ability.

The multi-action shape: :class:`SearchFilesParamsBag` is a plain router — the
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

_ACTIONS = ("glob", "grep", "content")

# The grep context window: lines shown above AND below each matched line.
# Owned here because the bag enforces the clamp; the ability imports the
# default for its glob path (where no context is rendered but the gather/cache
# key still carries a width).
DEFAULT_CONTEXT_LINES = 5
MAX_CONTEXT_LINES = 20

# Effectively unbounded — the ability's paginator clamps to the real page
# count itself; the bag only needs the >= 1 floor and the loud non-numeric
# rejection that clamp_int provides.
_MAX_PAGE = 10**9


class SearchFilesParamsBag(ParamBag):
    """Router and boundary type: ``SearchFilesAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> SearchFilesParamsBag | ToolResult:
        match params.get(Keys.action):
            case "glob":
                return SearchFilesGlobParams.from_params(params)
            case "grep":
                return SearchFilesGrepParams.from_params(params)
            case "content":
                return SearchFilesContentParams.from_params(params)
            case unknown:
                return ToolResult.err(
                    f"Unknown search_files action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )

    @classmethod
    def _query(cls, params: dict[str, object]) -> str | ToolResult:
        """The stripped search query, shared by both leaves.

        Hand-rolled instead of ``require_str`` because a blank/missing query
        keeps the handler's historical ``empty-query`` code — ``require_str``
        would report ``missing-params``.
        """
        raw = params.get(Keys.query)
        if raw is not None and not isinstance(raw, str):
            return ToolResult.err(
                f"'{Keys.query}' must be text.",
                code="invalid-param",
                hint=f"pass '{Keys.query}' as a string.",
            )
        if raw is None or not raw.strip():
            return ToolResult.err(
                "query is required and must not be empty.",
                code="empty-query",
                hint="pass a non-empty filename pattern (glob) or search string (grep).",
            )
        return raw.strip()


@dataclass(frozen=True, slots=True)
class SearchFilesGlobParams(SearchFilesParamsBag):
    """``query`` is required and non-blank (``empty-query`` otherwise);
    ``directory`` defaults to the filesystem root; ``page`` is 1-based —
    a non-numeric value is ``invalid-param``, an out-of-range one clamps."""

    query: str
    directory: str
    page: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        directory = cls.str_default(params, Keys.directory, default="/")
        if isinstance(directory, ToolResult):
            return directory
        page = cls.clamp_int(params, Keys.page, default=1, lo=1, hi=_MAX_PAGE)
        if isinstance(page, ToolResult):
            return page
        query = cls._query(params)
        if isinstance(query, ToolResult):
            return query
        return cls(query=query, directory=directory, page=page)


@dataclass(frozen=True, slots=True)
class SearchFilesGrepParams(SearchFilesParamsBag):
    """Everything :class:`SearchFilesGlobParams` carries, plus
    ``context_lines`` clamped into ``[0, MAX_CONTEXT_LINES]`` with
    :data:`DEFAULT_CONTEXT_LINES` when absent."""

    query: str
    directory: str
    page: int
    context_lines: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        directory = cls.str_default(params, Keys.directory, default="/")
        if isinstance(directory, ToolResult):
            return directory
        page = cls.clamp_int(params, Keys.page, default=1, lo=1, hi=_MAX_PAGE)
        if isinstance(page, ToolResult):
            return page
        context_lines = cls.clamp_int(
            params,
            Keys.context_lines,
            default=DEFAULT_CONTEXT_LINES,
            lo=0,
            hi=MAX_CONTEXT_LINES,
        )
        if isinstance(context_lines, ToolResult):
            return context_lines
        query = cls._query(params)
        if isinstance(query, ToolResult):
            return query
        return cls(
            query=query,
            directory=directory,
            page=page,
            context_lines=context_lines,
        )


@dataclass(frozen=True, slots=True)
class SearchFilesContentParams(SearchFilesParamsBag):
    """``query`` is required and non-blank (``empty-query`` otherwise);
    ``page`` is 1-based — a non-numeric value is ``invalid-param``, an
    out-of-range one clamps. No ``directory`` or ``context_lines`` — the
    content index is global."""

    query: str
    page: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        page = cls.clamp_int(params, Keys.page, default=1, lo=1, hi=_MAX_PAGE)
        if isinstance(page, ToolResult):
            return page
        query = cls._query(params)
        if isinstance(query, ToolResult):
            return query
        return cls(query=query, page=page)
