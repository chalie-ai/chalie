"""
Rich Render Skill — Just-in-time block protocol reference.

Returns compact documentation on how to produce rich visual blocks
(metrics, cards, charts, progress bars, timelines, etc.) using
fenced-code-block directives in markdown responses.

This is a zero-LLM skill — it returns a static reference string.
The main LLM uses this knowledge to produce rich output in the same
generation pass, avoiding any multi-LLM coordination.
"""

import logging

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "rich_render",
    "description": (
        "Returns block rendering reference. Call BEFORE generating your response "
        "when data would be significantly more readable as metrics, charts, cards, "
        "or structured layouts. Adds context cost — only use when plain markdown "
        "is clearly insufficient for the data density."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

# ── Compact block reference ────────────────────────────────────────────────
# This is what the LLM sees when it invokes the skill.
# Must be complete, extensible (one line per type), and as slim as possible.

_BLOCK_REFERENCE = """\
# Rich Blocks Reference
Write fenced code blocks with the block type as the language tag. Content is JSON.

## Available Types

```metric
{"label":"Name","value":"73%","unit":"optional","change":"+2.1%","trend":"up|down|neutral"}
```

```card
{"title":"Title","body":"Markdown body text","icon":"optional-icon-name","accent":"violet|cyan|magenta|green|red|amber"}
```

```chart
{"title":"optional","type":"bar|line|sparkline","labels":["A","B","C"],"series":[{"label":"S1","values":[10,20,30]}]}
```

```progress
{"label":"Task","value":75,"max":100,"variant":"info|success|warning|error"}
```

```timeline
{"events":[{"label":"Step 1","detail":"optional detail","status":"done|active|pending"}]}
```

## Also Available
- `alert`: {"message":"...","variant":"info|success|warning|error"}
- `badge`: {"text":"...","variant":"info|success|warning|error"}
- `keyvalue`: {"pairs":[{"key":"K","value":"V"}]}
- `columns`: {"columns":[{"width":"1fr","blocks":[...nested blocks...]}]}
- `section`: {"title":"...","collapsible":true,"blocks":[...]}

## Rules
- One JSON object per block, no arrays
- Blocks render between normal markdown — mix freely
- Use metric for single KPIs, card for summaries, chart for trends
- Keep it minimal — one card is better than five\
"""


def handle_rich_render(channel: str, params: dict) -> str:
    """
    Return the block protocol reference for the LLM to use.

    Args:
        channel: Current conversation channel (unused)
        params: Empty dict (no parameters)

    Returns:
        Block reference documentation string.
    """
    return _BLOCK_REFERENCE
