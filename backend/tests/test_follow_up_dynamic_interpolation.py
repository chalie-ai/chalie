# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — result-aware ``get_follow_up`` interpolates LIVE data into the nudge.

``get_follow_up(self, tr: ToolResult)`` lets a tool's success nudge carry the
exact value the model already sees (the downloaded ``path``, the activated tool
``name``, the window ``date_time``, the top result ``url``) — present-in-context
data the model can act on directly. This pins the ONE behavioural guarantee a
contract cannot: that the live value off ``tr`` actually reaches the rendered
nudge. Regressing any override to ignore ``tr`` (revert to static text) turns the
matching case RED — verified by reverting ``web_download`` to static text.

Everything else about the hook is enforced by the import-time probe CONTRACT in
``Ability.__init_subclass__`` (it calls ``get_follow_up(ToolResult.ok(""))`` on
every ability at registry import, so a non-``str`` return or a body-shape crash
fails the import, not a test) — there is no separate conformance sweep to drift.
The drop/keep/reword verdicts are plain text with no dynamic behaviour to break,
so they carry no test. Real hot path, zero mocks: a real ``ToolResult`` through
the real registered ability.
"""

from __future__ import annotations

import pytest

from abilities._registry import AbilityRegistry
from abilities._result import ToolResult

pytestmark = pytest.mark.unit

# name → (representative success ToolResult, the live value the nudge must echo).
_CASES: dict[str, tuple[ToolResult, str]] = {
    "find_tools": (
        ToolResult.ok({"injected": [{"name": "calendar"}], "not_found": []}),
        "`calendar` is now available and can be called.",
    ),
    "mcp_manager": (
        ToolResult.ok({"id": "x", "name": "weather", "url": "h", "status": "online"}),
        "`weather` is now available. Call `find_tools` with `weather`",
    ),
    "web_download": (
        ToolResult.ok({"path": "/tmp/x/file.pdf", "bytes": 1, "content_type": "application/pdf"}),
        "`read(/tmp/x/file.pdf)`",
    ),
    "review_tool_calls": (
        ToolResult.ok([{"iter": 1}], anchor="2026-04-07T14:30:00+00:00"),
        "`review_transcript(date_time=2026-04-07T14:30:00+00:00)`",
    ),
    "programming_docs_search": (
        ToolResult.ok([{"source": "Py", "title": "t", "url": "https://docs.python.org/x", "excerpt": "e"}]),
        "`read(https://docs.python.org/x)`",
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_dynamic_override_echoes_live_value_from_result(name: str) -> None:
    representative, must_contain = _CASES[name]
    ability = next(a for a in AbilityRegistry.all() if a.get_name() == name)
    nudge = ability.get_follow_up(representative)
    assert must_contain in nudge, f"{name} nudge dropped its live value: {nudge!r}"
