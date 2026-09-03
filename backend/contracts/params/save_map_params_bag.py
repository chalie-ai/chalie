"""SaveMapParamsBag — the typed input contract of the ``save_map`` ability
(contents + source + cues, plus optional derived_from map ids)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class SaveMapParamsBag(ParamBag):
    """``contents`` — the stripped distillation; ``source`` — the one-sentence
    origin of the episode; ``cues`` — up to five situational tags, the ONLY
    indexed field; ``derived_from`` — map ids this distillation replaces/extends
    (retires them from the searchable pool)."""

    #: A row is reachable only through its cues, and a model asked for an
    #: unbounded tag list writes a summary instead of tags. Overflow is rejected
    #: rather than silently trimmed — a dropped cue is a memory that never
    #: surfaces, and the model is the only thing that can choose which five.
    MAX_CUES: ClassVar[int] = 5

    contents: str
    source: str
    cues: tuple[str, ...]
    derived_from: tuple[int, ...]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        contents = cls.require_str(params, "contents")
        if isinstance(contents, ToolResult):
            return contents
        source = cls.require_str(params, "source")
        if isinstance(source, ToolResult):
            return source
        cues = cls.require_str_list(params, "cues")
        if isinstance(cues, ToolResult):
            return cues
        if len(cues) > cls.MAX_CUES:
            return ToolResult.err(
                f"'cues' accepts at most {cls.MAX_CUES} tags (received {len(cues)}).",
                code="invalid-param",
                hint=f"pick the {cls.MAX_CUES} tags that best identify the situation.",
            )
        raw = params.get("derived_from")
        if raw is None:
            derived: tuple[int, ...] = ()
        elif isinstance(raw, list):
            ids: list[int] = []
            for item in raw:
                if not isinstance(item, int) or isinstance(item, bool):
                    return ToolResult.err(
                        "'derived_from' must be a list of integers (map ids).",
                        code="invalid-param",
                        hint="pass derived_from as a list of existing map ids, or omit it.",
                    )
                ids.append(item)
            derived = tuple(ids)
        else:
            return ToolResult.err(
                "'derived_from' must be a list of integers (map ids).",
                code="invalid-param",
                hint="pass derived_from as a list of existing map ids, or omit it.",
            )
        return cls(
            contents=contents,
            source=source,
            cues=tuple(cues),
            derived_from=derived,
        )
