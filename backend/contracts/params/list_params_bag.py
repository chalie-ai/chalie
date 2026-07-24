"""ListParamsBag — the typed input contract of the ``list`` ability.

The multi-action shape: :class:`ListParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently.

Two deliberate lenient spots, both inherited contracts:

* the ``list`` target (field ``list_``) is NOT required here — the frontend
  silent-action and old id-addressed transcripts reach the ability through the
  healed ``id`` alias, so the bag only coerces and strips; a blank target
  comes back ``""`` and the ability's resolver raises its historical
  ``missing-target`` error.
* ``items`` accepts a JSON-array STRING and parses it (a weak model often
  serialises the array); element validation stays with each action's handler
  — add/remove filter to non-empty strings, check reads content/checked
  DICTS — so the bag carries the raw ``list[object]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

_ACTIONS = (
    "list_all", "create", "view",
    "add", "check", "remove",
    "clear", "rename", "delete",
)


def _name(params: dict[str, object]) -> str:
    """The create/rename display name: pre-gated present by ACTION_REQUIRED;
    coerced via ``str()`` (a bare number is a legal name) and stripped — the
    blank residue keeps the handlers' historical ``missing-name`` contract."""
    raw = params.get(Keys.name_)
    name = ("" if raw is None else str(raw)).strip()
    if not name:
        raise ToolParamError("'name' is required.", code="missing-name")
    return name


def _target(params: dict[str, object]) -> str:
    """The addressed list — name or 8-char id — coerced via ``str()`` and
    stripped, ``""`` when absent (see the module docstring: the resolver owns
    the missing-target error)."""
    raw = params.get(Keys.list)
    return ("" if raw is None else str(raw)).strip()


def _raw_items(params: dict[str, object]) -> list[object]:
    """``items`` as a raw list — ``[]`` when absent. A string is parsed as a
    JSON array when it is one, else folded to a single-item list (the pre-bag
    normalisation, preserved verbatim); any other type is a bad parameter."""
    raw = params.get(Keys.items, [])
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return [raw]
        return parsed if isinstance(parsed, list) else [raw]
    if not isinstance(raw, list):
        raise ToolParamError(
            "'items' must be a JSON array.",
            code="invalid-items",
            hint="pass 'items' as a JSON array, e.g. [\"milk\",\"eggs\"].",
        )
    return list(raw)  # copy — the frozen bag must not alias the caller's list


class ListParamsBag(ParamBag):
    """Router and boundary type: ``ListAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> ListParamsBag:
        match params.get(Keys.action):
            case "list_all":
                return ListAllParams.from_params(params)
            case "create":
                return ListCreateParams.from_params(params)
            case "view":
                return ListViewParams.from_params(params)
            case "add":
                return ListAddParams.from_params(params)
            case "check":
                return ListCheckParams.from_params(params)
            case "remove":
                return ListRemoveParams.from_params(params)
            case "clear":
                return ListClearParams.from_params(params)
            case "rename":
                return ListRenameParams.from_params(params)
            case "delete":
                return ListDeleteParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown list action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class ListAllParams(ListParamsBag):
    """list_all reads no parameters — the leaf exists so ``run()`` can narrow."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls()


@dataclass(frozen=True, slots=True)
class ListCreateParams(ListParamsBag):
    """``name_`` — the new list's name (non-blank); ``items`` — optional seed
    items, ``[]`` when absent (the handler then creates an empty list)."""

    name_: str
    items: list[object]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(name_=_name(params), items=_raw_items(params))


@dataclass(frozen=True, slots=True)
class ListViewParams(ListParamsBag):
    """``list_`` — the addressed list, resolved (and missing-target-checked)
    by the ability's resolver."""

    list_: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(list_=_target(params))


@dataclass(frozen=True, slots=True)
class ListAddParams(ListParamsBag):
    """``list_`` — the addressed list; ``items`` — the strings to add (the
    handler filters to non-empty strings and errors when none survive)."""

    list_: str
    items: list[object]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(list_=_target(params), items=_raw_items(params))


@dataclass(frozen=True, slots=True)
class ListCheckParams(ListParamsBag):
    """``list_`` — the addressed list; ``items`` — content/checked dicts.
    The non-empty-array guard here preserves the handler's historical
    ``invalid-items`` contract verbatim; per-element shape validation
    (``content`` + ``checked``) stays in the handler's splitter."""

    list_: str
    items: list[object]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        raw = params.get(Keys.items)
        if not isinstance(raw, list) or not raw:
            raise ToolParamError(
                "'items' must be a non-empty array.",
                code="invalid-items",
                hint="each item needs 'content' and 'checked', e.g. [{\"content\":\"milk\",\"checked\":true}].",
            )
        return cls(list_=_target(params), items=list(raw))


@dataclass(frozen=True, slots=True)
class ListRemoveParams(ListParamsBag):
    """``list_`` — the addressed list; ``items`` — the strings to remove
    (fuzzy-resolved against the list's contents by the handler)."""

    list_: str
    items: list[object]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(list_=_target(params), items=_raw_items(params))


@dataclass(frozen=True, slots=True)
class ListClearParams(ListParamsBag):
    """``list_`` — the addressed list to clear (unchecks nothing; removes all
    items). Resolution and missing-target live in the ability's resolver."""

    list_: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(list_=_target(params))


@dataclass(frozen=True, slots=True)
class ListRenameParams(ListParamsBag):
    """``list_`` — the addressed list; ``name_`` — its new name (non-blank,
    same ``missing-name`` contract as create)."""

    list_: str
    name_: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(list_=_target(params), name_=_name(params))


@dataclass(frozen=True, slots=True)
class ListDeleteParams(ListParamsBag):
    """``list_`` — the addressed list to delete (hard delete of the list row
    and its items)."""

    list_: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(list_=_target(params))
