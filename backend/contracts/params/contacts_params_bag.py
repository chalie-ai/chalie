"""ContactsParamsBag — the typed input contract of the ``contacts`` ability.

The multi-action shape: :class:`ContactsParamsBag` is a plain router — the
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

_ACTIONS = ("list", "get")


class ContactsParamsBag(ParamBag):
    """Router and boundary type: ``ContactsAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> ContactsParamsBag:
        match params.get(Keys.action):
            case "list":
                return ContactsListParams.from_params(params)
            case "get":
                return ContactsGetParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown contacts action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class ContactsListParams(ContactsParamsBag):
    """``limit`` clamps to ``[1, 50]`` with a default of 20 — the same range
    the old handler enforced via ``self.param(..., clamp=(1, 50), default=20)``.
    ``query`` defaults to ``""``; an empty query lists the whole book."""

    limit: int
    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            limit=cls.clamp_int(params, Keys.limit, default=20, lo=1, hi=50),
            query=cls.str_default(params, Keys.query, default="").strip(),
        )


@dataclass(frozen=True, slots=True)
class ContactsGetParams(ContactsParamsBag):
    """``identifier`` is required by ``ACTION_REQUIRED`` and by the handler's
    strip-or-empty-not-found path — ``require_str`` enforces both."""

    identifier: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(identifier=cls.require_str(params, Keys.identifier))
