"""RunScriptParamsBag — the typed input contract of the ``run_script`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class RunScriptParamsBag(ParamBag):
    """``path`` — absolute path to the .ts file to run, required; the model
    must create it first with file_write. ``args`` — optional command-line
    arguments passed to the script (readable inside it as Deno.args); defaults
    to an empty list. A non-list value for ``args`` is rejected here with
    ``code="invalid-args"`` to match the ability's original error contract."""

    path: str
    args: list[str]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        path = cls.require_str(params, Keys.path)
        if isinstance(path, ToolResult):
            return path
        raw_args = params.get(Keys.args)
        if raw_args is None:
            raw_args = []
        if not isinstance(raw_args, list) or any(not isinstance(a, str) for a in raw_args):
            return ToolResult.err(
                "args must be a list of strings.",
                code="invalid-args",
                hint='pass arguments like ["--flag", "value"].',
            )
        return cls(path=path, args=list(raw_args))
