"""DocumentParamsBag — the typed input contract of the ``document`` ability.

The multi-action shape: :class:`DocumentParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently.

The ``upload`` action is mechanical-only (not in the LLM ``INPUT_SCHEMA``): it
is the single file ingest path invoked by code via ``ToolDispatcher.dispatch``.
It takes a file PATH, never bytes — a path is tiny, so the act-trail can never
carry a multi-megabyte base64 blob and blow the context window. The upload leaf
is included in the router so the bag validates its shape before the mechanical
handler runs.

The ``_VALID_ACTIONS`` tuple excludes ``upload`` — it is the set surfaced in
unknown-action errors to the LLM, which never sees the mechanical route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

# LLM-callable actions only — ``upload`` is mechanical (not in INPUT_SCHEMA) and
# never surfaced in unknown-action errors to the model.
_VALID_ACTIONS = ("search", "list", "view", "delete", "restore", "create")


def _address(params: dict[str, object]) -> tuple[str, str]:
    """The optional ``id``/``name`` pair addressing an existing document —
    stripped, ``""`` when absent; the handler raises ``missing-target`` when
    both are blank (either one alone addresses the document)."""
    return (
        (ParamBag.opt_str(params, Keys.id) or "").strip(),
        (ParamBag.opt_str(params, Keys.name_) or "").strip(),
    )


class DocumentParamsBag(ParamBag):
    """Router and boundary type: ``DocumentAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> DocumentParamsBag:
        match params.get(Keys.action):
            case "search":
                return DocumentSearchParams.from_params(params)
            case "list":
                return DocumentListParams.from_params(params)
            case "view":
                return DocumentViewParams.from_params(params)
            case "delete":
                return DocumentDeleteParams.from_params(params)
            case "restore":
                return DocumentRestoreParams.from_params(params)
            case "create":
                return DocumentCreateParams.from_params(params)
            case "upload":
                return DocumentUploadParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown document action: {unknown!r}.",
                    code="unknown-action",
                    valid=_VALID_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class DocumentSearchParams(DocumentParamsBag):
    """``query`` is required; the ACTION_REQUIRED pre-gate enforces presence,
    so the bag uses ``require_str`` which also strips and rejects blank."""

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(query=cls.require_str(params, Keys.query))


@dataclass(frozen=True, slots=True)
class DocumentListParams(DocumentParamsBag):
    """No fields — the list action requires no params."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls()


@dataclass(frozen=True, slots=True)
class DocumentViewParams(DocumentParamsBag):
    """Both ``id`` and ``name`` are optional — the handler's ``_resolve_unique``
    tries id first, then name, then raises ``missing-target`` if neither is set.
    The bag extracts them as stripped strings (``""`` when absent) so the
    handler can use them directly."""

    id: str
    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        doc_id, name = _address(params)
        return cls(id=doc_id, name=name)


@dataclass(frozen=True, slots=True)
class DocumentDeleteParams(DocumentParamsBag):
    """Same shape as view — ``_resolve_unique`` handles the resolution logic."""

    id: str
    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        doc_id, name = _address(params)
        return cls(id=doc_id, name=name)


@dataclass(frozen=True, slots=True)
class DocumentRestoreParams(DocumentParamsBag):
    """Both ``id`` and ``name`` are optional — the handler tries id first, then
    name (searching deleted docs), then raises ``missing-target``."""

    id: str
    name: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        doc_id, name = _address(params)
        return cls(id=doc_id, name=name)


@dataclass(frozen=True, slots=True)
class DocumentCreateParams(DocumentParamsBag):
    """``name_`` and ``content`` are both required — the ACTION_REQUIRED
    pre-gate enforces presence; the bag re-asserts via ``require_str``."""

    name_: str
    content: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            name_=cls.require_str(params, Keys.name_),
            content=cls.require_str(params, Keys.content),
        )


@dataclass(frozen=True, slots=True)
class DocumentUploadParams(DocumentParamsBag):
    """``path`` is required (the mechanical ingest takes a path, never bytes);
    ``name_`` is optional and defaults to ``""`` (the handler derives a
    sanitised name from the basename when absent)."""

    path: str
    name_: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            path=cls.require_str(params, Keys.path),
            name_=cls.opt_str(params, Keys.name_) or "",
        )

