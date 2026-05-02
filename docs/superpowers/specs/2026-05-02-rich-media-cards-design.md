# Rich Media Cards — Design

**Date:** 2026-05-02
**Status:** Draft for approval
**Branch target:** rc-0.6.0 (or successor)
**Author / sign-off:** Dylan

## 1. Summary

Chalie's chat surface today renders every assistant turn as a single text bubble. For certain content types — weather, web search, web browser screenshots, etc. — a purpose-built card with structured data, restrained animation, and clean formatting will be a substantially better experience than prose.

Rich media cards introduce an opt-in protocol where:

1. A tool that supports rich-media rendering bakes a small instruction into its own return string telling the LLM to wrap its synthesis in `[<tool>_<N>]…[/<tool>_<N>]` tags.
2. The LLM emits its response with those tags inline.
3. A single backend parser turns the raw LLM text plus the turn's `tool_calls` rows into an ordered list of `{type: text}` and `{type: rich}` segments.
4. The frontend iterates segments, rendering text segments as ordinary chat bubbles and rich segments through a per-tool-type card module.

All rendering happens client-side. The backend only assembles a standard segment array.

The pilot card type is **weather**. The architecture is general; future cards (search, browser, others) follow the same pattern with no framework changes.

## 2. Goals

- Render a rich, animated weather card when the weather tool is used.
- Allow Chalie to mix prose and rich-media in a single response, producing multiple text bubbles and cards interleaved.
- Keep the protocol so simple that adding a new card type later is just: tweak one tool's return string + drop a new frontend module.
- Persist enough information that page refresh fully reconstructs cards from the database, with no client-side replay state.
- Hallucination-proof: the LLM never authors the data values shown in a card.

## 3. Non-goals (v1)

- `search`, `browser`, or any other card beyond weather. They are anticipated by the architecture, but not implemented in this slice.
- Live, ticking, or refreshable cards. A card renders once with the payload it was given.
- Streaming partial cards while the LLM is mid-generation. Cards render after the full assistant turn arrives, matching the existing `message` event timing.
- In-card actions (refresh / pin / expand). Cards are presentational only.
- Schema migrations. The design adds zero columns, zero tables.
- Self-correction signals fed back to the LLM on parser drift (the response is already shipped to the user by the time we'd know; no recovery is possible without re-running the turn).

## 4. End-to-end data flow

```
Tool runs (e.g. weather)
   └─ returns: <data JSON>\n\n<rich-media instruction with [weather_N] tag>
        │
        ▼
tool_calls.result  ←  full string verbatim (data + instruction trailer)
_act_trail         ←  same string, in-memory, shown to LLM
        │
        ▼
LLM emits raw response, possibly containing [weather_1]…[/weather_1]
        │
        ▼
transcript.content  ←  raw LLM response with tags intact (invariant)
        │
        ▼
RichMediaParser.parse(raw_text, this_turn_tool_calls) → segments[]
        │
        ├──────────────────────────────┬──────────────────────────────┐
        │                              │                              │
   WS `message` event              /conversation/recent          (same parser, same shape)
   ships segments live             reruns parser on refresh
        │                              │
        ▼                              ▼
Frontend renderer iterates segments → text bubbles + card modules
```

## 5. Backend protocol

### 5.1 Tool contract

A tool opts into rich-media rendering **at runtime, in its own return string** — not via class metadata, not via a parallel return field, not via a registry. The tool's `execute()` returns one string of the shape:

```
<JSON-encoded structured data>

This tool supports rich-media rendering. To present this result as a card,
wrap your synthesis in [weather_1]your synthesis here[/weather_1]. You may mix
prose and rich-media tags freely; tags will render as cards between text bubbles.
```

The tool obtains its ordinal (`_1`, `_2`, …) from a per-turn, per-tool-name counter exposed by the dispatcher. The counter increments each time a given tool dispatches in the current ACT loop iteration.

A tool that does not want rich-media rendering on a given call simply returns its result without an instruction trailer. There is no static "rich-media-capable" flag on the tool class.

### 5.2 Persistence

| Surface | Holds |
|---|---|
| `transcript.content` | The LLM's verbatim response, **with tags intact**. |
| `tool_calls.result` | The tool's full return string (data JSON + instruction trailer if present). |
| anywhere else | Nothing. No schema change, no new column, no new table. |

This is a strict requirement: refreshing the page must rebuild cards exclusively from these two existing surfaces.

### 5.3 RichMediaParser

A new module `backend/services/rich_media_parser.py` exposing one pure function:

```python
def parse(raw_text: str, tool_calls: list[ToolCallRow]) -> list[Segment]
```

**Algorithm (~80 LOC including helpers):**

```python
TAG_RE = re.compile(r"\[([a-z][a-z0-9_]*)_(\d+)\](.*?)\[/\1_\2\]", re.DOTALL)

def parse(raw_text, tool_calls):
    segments = []
    cursor = 0
    for m in TAG_RE.finditer(raw_text):
        if m.start() > cursor:
            text = raw_text[cursor:m.start()].strip()
            if text:
                segments.append({"type": "text", "content": text})
        tag = f"{m.group(1)}_{m.group(2)}"
        synthesis = m.group(3).strip()
        payload = _find_payload(tag, tool_calls)
        if payload is not None:
            segments.append({
                "type": "rich",
                "tag": tag,
                "payload": payload,
                "synthesis": synthesis,
            })
        else:
            # Orphan tag: degrade gracefully to plain text, log warning.
            if synthesis:
                segments.append({"type": "text", "content": synthesis})
            log.warning("rich_media: orphan tag %s", tag)
        cursor = m.end()
    tail = raw_text[cursor:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})
    return segments
```

`_find_payload(tag, tool_calls)` scans `tool_calls` for a row whose `result` contains the literal substring `[<tag>]`. Pairing therefore happens via the LLM-visible tag name appearing in both the tool's result and the LLM's response — no separate ID system, no framework metadata.

`_extract_data(result)` splits on the first `\n\n` and returns the head as parsed JSON if possible, else the raw head string. Tools must therefore put their structured data on the first JSON segment of their return, with the instruction trailer separated by a blank line.

**Invariant — single data source:** Both the live path and the refresh path read `tool_calls` from the database, not from in-memory state. The `MessageProcessor`'s atomic store of pending tool_calls completes before the WS `message` event is assembled, so by the time the parser runs the rows are durably persisted under the assistant turn's `transcript_id`. This guarantees the live and refresh paths produce byte-identical segment arrays.

### 5.4 Edge cases — Q7 resolution

All of the following result in a **silent strip / passthrough to plain text** (with a warning log; no user-visible error stub, no LLM feedback signal):

| Failure | Behaviour |
|---|---|
| LLM emits `[weather_1]` but no weather tool ran this turn (orphan / hallucinated). | `_find_payload` returns `None` → synthesis becomes a `text` segment. |
| LLM emits `[weather_3]` but only 2 weather calls happened. | Same as above. |
| Mismatched open/close (`[weather_1]…[/weather_2]`). | `TAG_RE`'s backreference prevents a match → entire span passes through as text. |
| Unclosed tag. | No match → text passthrough. |
| Tag prefix with no frontend module registered. | Frontend registry returns plain-text fallback (Section 6.2). |

Rationale for silent over loud: by the time the parser runs, the response has already been generated; loud failures would either show error stubs to end users or leak parser noise into the transcript. The drift is observable in logs.

### 5.5 WebSocket message shape

```json
{
  "type": "message",
  "content": "<raw LLM text with tags intact>",
  "segments": [
    {"type": "text", "content": "Here is some information about..."},
    {"type": "rich", "tag": "weather_1", "payload": {...}, "synthesis": "..."},
    {"type": "text", "content": "I've also checked Tokyo..."},
    {"type": "rich", "tag": "weather_2", "payload": {...}, "synthesis": "..."},
    {"type": "text", "content": "Don't forget your umbrella"}
  ],
  "topic": "...",
  "mode": "...",
  "exchange_id": "...",
  "seq": 42,
  "metrics": {...}
}
```

`content` remains for backwards compatibility (metrics consumers, debug tooling). The frontend renders exclusively from `segments`. For non-rich responses, `segments` is `[{type: "text", content: <raw>}]`.

### 5.6 Refresh path

The existing `/conversation/recent` route reads from `transcript`. The change: also fetch `tool_calls` rows joined to each transcript row, then for each assistant row run `RichMediaParser.parse(content, tool_calls)` and include `segments` on the response. Frontend rendering on refresh becomes byte-identical to live rendering.

## 6. Frontend protocol

### 6.1 Renderer changes

`frontend/interface/renderer.js` switches from rendering `content` directly to iterating `segments`:

```js
function appendChalieForm(msg, ...) {
  const segments = msg.segments || [{type: "text", content: msg.content || ""}];
  for (const seg of segments) {
    if (seg.type === "text") {
      appendTextBubble(seg.content);
    } else if (seg.type === "rich") {
      appendRichCard(seg.tag, seg.payload, seg.synthesis);
    }
  }
}
```

Each segment becomes its own DOM element — multiple text bubbles per assistant turn are explicitly supported and visually equivalent to today's single bubble repeated. Cards render as siblings to bubbles.

If `segments` is missing on the message (legacy clients, non-rich responses), the renderer falls back to a single text bubble using `content` — making this a backward-compatible change.

### 6.2 Card module registry

```
frontend/interface/rich_media/
  registry.js       — tag-prefix → module map
  weather.js        — v1 weather card module
  base_card.css     — shared Radiant card chrome
  weather.css       — weather-specific styles
  icons/weather/    — semantic SVG icons (sunny, rain, cloudy, partly_cloudy, snow, …)
```

`registry.js`:

```js
import * as weather from "./weather.js";

const REGISTRY = {
  weather: weather,
  // search, browser, etc. — future
};

export function renderCard(tag, payload, synthesis, root) {
  const prefix = tag.split("_")[0];
  const mod = REGISTRY[prefix];
  if (!mod) {
    appendTextBubble(synthesis || "");
    return;
  }
  mod.render(payload, synthesis, root);
}
```

Unknown tag prefixes silently fall back to a text bubble using the synthesis (mirrors the backend orphan behaviour).

Each card module exports a single `render(payload, synthesis, root)`. The module owns its DOM, CSS, and entrance animation.

### 6.3 Animation scope (v1)

Per the Radiant design system's "restraint" principle:

- Entrance: card fades in and lifts 8px on mount (200ms ease-out).
- Numeric count-up on first render where appropriate (e.g., temperature 0 → 12°C over 400ms). Per-module opt-in.
- No looping animations, no live ticking, no parallax.

Animation lives entirely inside the card module; the framework imposes nothing.

## 7. Weather card — v1 contract

### 7.1 Tool payload shape

The data portion of `tool_calls.result` (everything before the instruction trailer):

```json
{
  "location": "London, UK",
  "current": {
    "temp_c": 12,
    "condition": "Partly cloudy",
    "icon": "partly_cloudy",
    "feels_like_c": 10,
    "humidity_pct": 78,
    "wind_kph": 14
  },
  "forecast": [
    {"day": "Mon", "high_c": 14, "low_c": 9,  "icon": "rain"},
    {"day": "Tue", "high_c": 16, "low_c": 10, "icon": "cloudy"},
    {"day": "Wed", "high_c": 13, "low_c": 8,  "icon": "rain"}
  ]
}
```

Icons use semantic keys (`sunny`, `cloudy`, `partly_cloudy`, `rain`, `snow`, `storm`, `fog`). Frontend maps to local SVG. No URLs over the wire.

### 7.2 Card visual (sketch)

```
┌─────────────────────────────────────────┐
│ London, UK                              │
│                                         │
│  [icon]  12°C                           │
│          Partly cloudy                  │
│          Feels 10° · 78% · 14 km/h      │
│                                         │
│ ─────────────────────────────────────── │
│ Mon  14°/9°  [icon]                     │
│ Tue  16°/10° [icon]                     │
│ Wed  13°/8°  [icon]                     │
│ ─────────────────────────────────────── │
│  "{LLM synthesis text}"                 │
└─────────────────────────────────────────┘
```

The synthesis renders below the data block in a slightly muted style — visually distinct from the data, so it reads as Chalie's interpretation rather than raw values. Card chrome uses Radiant variables: near-black background, 1px border with subtle violet/cyan accent on the icon, soft glow on hover.

## 8. File-level integration map

### 8.1 Backend — modified

| File | Change |
|---|---|
| `backend/services/rich_media_parser.py` | **NEW.** ~80 LOC. Pure parsing function described in §5.3. Unit-tested in isolation. |
| `backend/services/message_processor.py` | (1) Per-turn `_rich_media_counters: dict[str, int]` on instance init; bumped on each tool dispatch; ordinal exposed to dispatch context. (2) Just before WS `message` event assembly, call `RichMediaParser.parse(llm_response, this_turn_tool_calls)` and stash the result on the outgoing event. |
| `backend/api/websocket.py` | Add `"segments": <parsed segments>` to the `message_evt` dict at the existing assembly site (around line 719). `content` field unchanged. |
| `backend/api/conversation.py` | `/conversation/recent` query gains a join (or follow-up SELECT) for `tool_calls` per transcript row. For each assistant row, run `RichMediaParser.parse(content, tool_calls)` and include `segments` in the response. |
| `backend/services/dispatcher.py` (or equivalent) | New `current_call_index(tool_name)` helper exposed via dispatch context, so a tool's `execute()` can read its ordinal-this-turn. |
| `backend/tools/weather/<module>.py` | `execute()` returns the v1 payload JSON followed by the rich-media instruction trailer with the dispatcher-provided ordinal. Payload shape per §7.1. |
| `backend/schema.sql` | **NO CHANGE.** Confirmed — `transcript.content` and `tool_calls.result` already hold everything needed. |

### 8.2 Backend — tests

| File | Purpose |
|---|---|
| `backend/tests/test_rich_media_parser.py` | **NEW.** Deterministic unit tests: single tag, multiple interleaved tags, orphan tag, unclosed tag, mismatched open/close, empty synthesis, no tags at all. |
| `backend/tests/test_message_processor_rich_media.py` | **NEW.** Feature test: weather tool result + faked LLM response with `[weather_1]` → segment array on outgoing WS event. |
| `backend/tests/test_conversation_recent_rich_media.py` | **NEW.** Feature test: persisted transcript + tool_calls → `/conversation/recent` returns segments byte-identical to what the live path produced. |

### 8.3 Frontend — new

| File | Purpose |
|---|---|
| `frontend/interface/rich_media/registry.js` | Tag-prefix → card module map (~20 LOC). |
| `frontend/interface/rich_media/weather.js` | Weather card module: `render(payload, synthesis, root)`, ~150 LOC including animation hooks. |
| `frontend/interface/rich_media/weather.css` | Card styling using Radiant variables. |
| `frontend/interface/rich_media/base_card.css` | Shared card chrome (border, padding, entrance keyframes, hover glow). |
| `frontend/interface/rich_media/icons/weather/*.svg` | 6–10 semantic SVG icons. |

### 8.4 Frontend — modified

| File | Change |
|---|---|
| `frontend/interface/renderer.js` | `appendChalieForm()` switches from "render `content` once" to "iterate `segments`". Falls back to single-bubble rendering when `segments` is missing. |
| `frontend/interface/chat.js` | No structural change; the existing `appendChalieForm(msg, …)` call already passes the whole message. |
| `frontend/index.html` (importmap injector) | Register the new `rich_media/` modules so they version with `?v=VERSION`. |

### 8.5 Documentation

| File | Change |
|---|---|
| `docs/04-ARCHITECTURE.md` | New short subsection "Rich Media Cards" describing the parser chokepoint, segment shape, and refresh path. |
| `docs/03-WEB-INTERFACE.md` | Append "Rich media cards" to the Radiant design system, referencing the entrance-only animation rule. |
| `backend/tools/weather/<readme or docstring>` | Document the rich-media instruction string and the JSON-on-first-line convention so future rich-media tools follow the same pattern. |

### 8.6 Estimated size

- Backend: ~250 LOC new (parser + helpers + tests), ~30 LOC of edits.
- Frontend: ~300 LOC new (renderer + weather card + CSS), ~20 LOC of edits.
- Schema migrations: zero.

## 9. Acceptance criteria

The feature is complete when all of the following are true:

1. Asking Chalie about the weather (via the existing weather tool) renders a card matching §7.2, with the LLM's synthesis displayed beneath the data.
2. Asking Chalie about the weather in two cities in the same turn renders two distinct cards, each paired with its city's data; the prose between them renders as separate text bubbles.
3. Refreshing the page after a weather conversation rebuilds the same cards from the database; the rebuilt cards are visually indistinguishable from the live-rendered cards.
4. The LLM omitting tags (responding in pure prose) renders normally as one or more text bubbles.
5. The LLM emitting an orphan / mismatched / unclosed tag results in plain-text rendering with a warning logged backend-side. No user-visible error stub.
6. A new tool can be made rich-media-capable by editing only its own `execute()` return string and adding a frontend module + registry entry. No framework or schema changes.
7. All new unit and feature tests pass under `pytest -m unit`.
8. `/conversation/recent` response shape gains `segments` per assistant row; existing consumers reading `content` continue to work unchanged.

## 10. Open questions

None at the time of writing. All Q1–Q9 from the brainstorming session are resolved and reflected in this document.
