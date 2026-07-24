"""SkillBuilderParamsBag — the typed input contract of the ``skill_builder``
ability (and its ``skill_manager`` SYSTEM variant, which shares ``run()`` and
therefore this bag verbatim).

The multi-action shape: :class:`SkillBuilderParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently.

Edit semantics live in the field types: ``use_for``/``content`` come back
``""`` when omitted or blank (the handler keeps the stored value), while
``tags`` is tri-state — ``None`` means "not given, keep the stored tags" and
``""`` means "explicitly clear them". The bag preserves that distinction; the
merge against the stored row stays in the handler, which is the only place the
existing skill is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

_ACTIONS = ("create", "edit", "delete", "read", "list")


class SkillBuilderParamsBag(ParamBag):
    """Router and boundary type: ``SkillBuilderAbility.run()`` is annotated
    with this class, but every instance that reaches it is one of the leaves
    below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> SkillBuilderParamsBag:
        match params.get(Keys.action):
            case "create":
                return SkillBuilderCreateParams.from_params(params)
            case "edit":
                return SkillBuilderEditParams.from_params(params)
            case "delete":
                return SkillBuilderDeleteParams.from_params(params)
            case "read":
                return SkillBuilderReadParams.from_params(params)
            case "list":
                return SkillBuilderListParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown skill_builder action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class SkillBuilderCreateParams(SkillBuilderParamsBag):
    """The validated create inputs: ``title``, ``use_for``, and ``content``
    (pre-gated present by ACTION_REQUIRED; the bag rejects the whitespace-only
    residue) plus optional ``tags`` — comma-separated keywords, ``""`` when
    omitted."""

    title: str
    use_for: str
    content: str
    tags: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            title=cls.require_str(params, Keys.title_),
            use_for=cls.require_str(params, Keys.use_for),
            content=cls.require_str(params, Keys.content),
            tags=cls.str_default(params, Keys.tags, default="").strip(),
        )


@dataclass(frozen=True, slots=True)
class SkillBuilderEditParams(SkillBuilderParamsBag):
    """``title`` addresses the skill being edited (required); the other fields
    are the partial update — ``use_for``/``content`` come back ``""`` when
    omitted or blank (keep stored value), ``tags`` is ``None`` when omitted
    (keep) but kept verbatim when given, so ``""`` explicitly clears them."""

    title: str
    use_for: str
    content: str
    tags: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        raw_tags = cls.opt_str(params, Keys.tags)
        return cls(
            title=cls.require_str(params, Keys.title_),
            use_for=cls.str_default(params, Keys.use_for, default="").strip(),
            content=cls.str_default(params, Keys.content, default="").strip(),
            tags=raw_tags.strip() if raw_tags is not None else None,
        )


@dataclass(frozen=True, slots=True)
class SkillBuilderDeleteParams(SkillBuilderParamsBag):
    """``title`` of the user skill to delete — the ownership check (only
    ``source='user'`` skills can be deleted) stays in the handler."""

    title: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(title=cls.require_str(params, Keys.title_))


@dataclass(frozen=True, slots=True)
class SkillBuilderReadParams(SkillBuilderParamsBag):
    """``title`` of the skill to read in full — any source; the user-copy
    preference on a title collision stays in the handler."""

    title: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(title=cls.require_str(params, Keys.title_))


@dataclass(frozen=True, slots=True)
class SkillBuilderListParams(SkillBuilderParamsBag):
    """List reads no parameters — the leaf exists so ``run()`` can narrow."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls()
