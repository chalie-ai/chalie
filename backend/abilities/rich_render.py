"""
RichRenderAbility — Just-in-time block protocol reference.

Returns compact documentation on how to produce rich visual blocks
(metrics, cards, charts, progress bars, timelines, etc.) using
fenced-code-block directives in markdown responses.

Zero-network, zero-LLM — returns a static reference string instantly.
"""

import logging
from typing import ClassVar

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class RichRenderAbility(Ability):
    NAME = "rich_render"
    SUMMARY = "Return the rich block rendering reference so you can produce metrics, charts, cards, and timelines in your response."
    EXAMPLES = [
        "show me a bar chart of my spending by category",
        "display my goals as a progress bar",
        "give me a dashboard view of my weekly stats",
        "render this data as a card with a title",
        "I want a timeline of the project milestones",
        "show my KPIs as metric blocks",
        "create a visual breakdown of the results",
        "display the comparison as a chart",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    TIMEOUT = 10

    _BLOCK_REFERENCE: ClassVar[str] = """\
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
- Keep it minimal — one card is better than five"""

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        return {"text": _skill_tag("rich_render", self._BLOCK_REFERENCE)}
