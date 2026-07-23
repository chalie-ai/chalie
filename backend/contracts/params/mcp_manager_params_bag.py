"""McpManagerParamsBag — the typed input contract of the ``mcp_manager`` ability.

The multi-action shape: :class:`McpManagerParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently.

The ``enable`` and ``disable`` leaves carry ``server_id`` and ``name`` as
stripped strings with ``""`` meaning "not provided" — the "at least one" OR is
enforced by the handler, not the bag, because the error code and the hint text
("use action=list to see configured servers" vs. "no server with id/name") are
action-specific business rules the bag cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

_ACTIONS = ("list", "add", "enable", "disable")


class McpManagerParamsBag(ParamBag):
    """Router and boundary type: ``McpManagerAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> McpManagerParamsBag:
        match params.get(Keys.action):
            case "list":
                return McpManagerListParams.from_params(params)
            case "add":
                return McpManagerAddParams.from_params(params)
            case "enable":
                return McpManagerEnableParams.from_params(params)
            case "disable":
                return McpManagerDisableParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown mcp_manager action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class McpManagerListParams(McpManagerParamsBag):
    """``list`` accepts no action-specific params — the handler builds the
    full inventory from the service alone."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls()


@dataclass(frozen=True, slots=True)
class McpManagerAddParams(McpManagerParamsBag):
    """``name`` and ``host`` are required — ``require_str`` strips and rejects
    blank strings. ``headers`` is optional, defaulting to ``{}``; when given it
    must be an object of string key/value pairs (the dispatcher already
    JSON-decodes JSON-string payloads, so a model-supplied object arrives as a
    dict) — anything else is ``invalid-param``."""

    name: str
    host: str
    headers: dict[str, str]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        headers = params.get(Keys.headers) or {}
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise ToolParamError(
                f"'{Keys.headers}' must be a JSON object of string key/value pairs.",
                code="invalid-param",
                hint=f"pass '{Keys.headers}' as an object mapping header names to values.",
            )
        return cls(
            name=cls.require_str(params, Keys.name_),
            host=cls.require_str(params, Keys.host),
            headers=cast("dict[str, str]", headers),
        )


@dataclass(frozen=True, slots=True)
class McpManagerEnableParams(McpManagerParamsBag):
    """``server_id`` and ``name`` are individually optional; ``""`` (after the
    strip) means "not provided". The handler enforces the at-least-one OR and
    produces the per-branch error codes — a judgement the bag does not own."""

    server_id: str
    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            server_id=(cls.opt_str(params, Keys.server_id) or "").strip(),
            name=(cls.opt_str(params, Keys.name_) or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class McpManagerDisableParams(McpManagerParamsBag):
    """Mirror of :class:`McpManagerEnableParams` — same wire shape, same
    handler-owned OR."""

    server_id: str
    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            server_id=(cls.opt_str(params, Keys.server_id) or "").strip(),
            name=(cls.opt_str(params, Keys.name_) or "").strip(),
        )
