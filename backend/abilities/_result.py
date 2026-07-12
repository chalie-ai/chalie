"""``ToolResult`` — the single, sealed contract for what every ability returns.

The ability-side equivalent of the frozen ``ProcessorConfig`` contract: a frozen
dataclass + construction-only-via-classmethods + ``__post_init__`` validation, so
that what an ability hands back to the ACT loop cannot diverge from the wire
contract no matter who edits an ability next.

An ability NEVER formats the wire envelope (``[tool(status=…)]\n…\n[end:tool]``)
and NEVER imports the skill-tag formatter.  It returns a ``ToolResult``; the
dispatcher (``abilities/_dispatcher.py``) is the ONE place that renders the
envelope.

Construction is classmethod-only::

    ToolResult.ok(body, *, rich=None, **meta)
    ToolResult.err(message, *, code, hint=None, valid=(), **meta)

``body`` is the payload: a ``str`` is shown to the model verbatim (prose); a
``dict``/``list`` is rendered as compact JSON.  ``meta`` is a FLAT scalar map
rendered into the open tag (``count=3, truncated=true``).  Errors additionally
carry a stable kebab-case ``code``, an optional one-line ``hint``, and a
``valid`` tuple of acceptable actions/values — everything a weak model needs to
self-correct without re-reading the schema.

``ToolParamError`` is the one sanctioned way an ability (via ``Ability.param``)
signals a bad input; the dispatcher catches it and renders it canonically with
its ``code``/``hint``/``valid``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Scalar types permitted in the flat ``meta`` map (rendered into the open tag).
_SCALAR = (str, int, float, bool)


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """The single sanctioned truncation primitive so every tool reports truncation
    the same way (``meta truncated=true``) instead of silently dropping output.
    """
    if limit < 0:
        raise ValueError("truncate limit must be >= 0")
    if len(text) <= limit:
        return text, False
    return text[:limit], True


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The sealed return type of every ``Ability.run``.

    Immutable (frozen) — once built it cannot be mutated mid-loop.  Build it ONLY
    via :meth:`ok` / :meth:`err`; ``__post_init__`` enforces the invariants
    (status enum, meta flatness, ``code`` set iff error).
    """

    status: str
    body: "str | Mapping[str, object] | Sequence[object]"
    meta: "dict[str, object]" = field(default_factory=dict)
    code: str | None = None
    hint: str | None = None
    valid: tuple[str, ...] = ()
    rich: "dict[str, object] | None" = None

    def __post_init__(self) -> None:
        if self.status not in ("success", "error"):
            raise ValueError(
                f"ToolResult.status must be 'success' or 'error', got {self.status!r}"
            )
        if not isinstance(self.body, (str, dict, list)):
            raise TypeError(
                f"ToolResult.body must be str | dict | list, got {type(self.body).__name__}"
            )
        if not isinstance(self.meta, dict):
            raise TypeError("ToolResult.meta must be a dict")
        for k, v in self.meta.items():
            if not isinstance(k, str):
                raise TypeError(f"ToolResult.meta keys must be str, got {k!r}")
            if not isinstance(v, _SCALAR):
                raise TypeError(
                    f"ToolResult.meta[{k!r}] must be a scalar (str/int/float/bool), "
                    f"got {type(v).__name__}"
                )
        if not isinstance(self.valid, tuple):
            raise TypeError("ToolResult.valid must be a tuple")
        if self.rich is not None and not isinstance(self.rich, dict):
            raise TypeError("ToolResult.rich must be a dict or None")
        # code is set iff status == error.
        if self.status == "error" and not self.code:
            raise ValueError("an error ToolResult must carry a non-empty code")
        if self.status == "success" and self.code is not None:
            raise ValueError("a success ToolResult must not carry a code")

    # ── Construction — the only two sanctioned entry points ────────────────────

    @classmethod
    def ok(cls, body: "str | Mapping[str, object] | Sequence[object]", *, rich: "dict[str, object] | None" = None, **meta: object) -> "ToolResult":
        return cls(status="success", body=body, meta=dict(meta), rich=rich)

    @classmethod
    def err(
        cls,
        message: str,
        *,
        code: str,
        hint: str | None = None,
        valid: tuple[str, ...] = (),
        **meta: object,
    ) -> "ToolResult":
        return cls(
            status="error",
            body=message,
            meta=dict(meta),
            code=code,
            hint=hint,
            valid=tuple(valid),
        )
