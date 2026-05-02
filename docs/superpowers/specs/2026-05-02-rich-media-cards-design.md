# Rich Media Cards — Design

**Date:** 2026-05-02
**Status:** Approved (post-research reconciliation 2026-05-02)
**Branch target:** rc-0.6.0 → `feature/rich-media-cards`
**Author / sign-off:** Dylan

## 1. Summary

Chalie's chat surface today renders every assistant turn as a single text bubble. For certain content types — weather, web search, web browser screenshots, etc. — a purpose-built card with structured data, restrained animation, and clean formatting will be a substantially better experience than prose.

Rich media cards introduce an opt-in protocol where:

1. A tool that supports rich-media rendering bakes a small instruction into its own return string telling the LLM to wrap its synthesis in `<span id='<tool>_<N>'>…</span>` tags.
2. The LLM emits its response with those spans inline.
3. The HTML sanitiser is taught to allow `<span id="…">` so tags survive intact through the existing sanitisation pass.
4. A single backend parser turns the (post-sanitisation) assistant text plus the turn's `tool_calls` rows into an ordered list of `{type: text}` and `{type: rich}` segments at the WS-send / refresh boundary.
5. The frontend iterates segments, rendering text segments as ordinary chat bubbles and rich segments through a per-tool-type card module.

All rendering happens client-side. The backend only assembles a standard segment array.

The pilot card type is **weather**. The architecture is general; future cards (search, browser, others) follow the same pattern with no framework changes.

### 1.1 Why span syntax instead of bracket tags

The first draft used `[weather_1]…[/weather_1]`. Research found that all assistant content is sanitised through `nh3.clean()` inside `OutputService.enqueue_text()` *before* the WS-send boundary, which means by the time the parser would run the raw text is already gone. Three options were considered: (a) move parsing into `MessageProcessor.postTurn()` against `llm_response.text`, (b) thread raw text through metadata, (c) switch to a tag syntax that survives sanitisation.

**Decision: (c).** `<span id='tool_N'>…</span>` is whitelisted in the nh3 config, the assistant text reaches the WS boundary with tags intact, and the parser runs at the boundary as originally intended. No new in-memory plumbing, no duplicate copy of the response, postTurn stays out of this concern.

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
- Restructuring the weather tool's existing payload. Field names, presence, and types stay as they are today; the v1 card adapts to the current shape (§7.1).

## 4. End-to-end data flow

```
Tool runs (e.g. weather)
   └─ returns: <data JSON>\n\n<rich-media instruction with <span id='weather_N'> tag>
        │
        ▼
tool_calls.result  ←  full string verbatim (data + instruction trailer)
_act_trail         ←  same string, in-memory, shown to LLM
        │
        ▼
LLM emits raw response, possibly containing <span id='weather_1'>…</span>
        │
        ▼
OutputService.enqueue_text() → nh3.clean() (span+id whitelisted, tags survive)
        │
        ▼
transcript.content  ←  sanitised LLM response, **tags intact**
        │
        ▼
RichMediaParser.parse(sanitised_text, this_turn_tool_calls) → segments[]
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
wrap your synthesis in <span id='weather_1'>your synthesis here</span>.
You may mix prose and rich-media spans freely; spans render as cards
between text bubbles.
```

The tool obtains its ordinal (`_1`, `_2`, …) from a per-turn, per-tool-name counter exposed by the dispatcher (§5.7). The counter increments each time a given tool dispatches in the current ACT loop iteration.

A tool that does not want rich-media rendering on a given call simply returns its result without an instruction trailer. There is no static "rich-media-capable" flag on the tool class.

The base `Ability.execute()` return-type annotation in `backend/abilities/_base.py` widens from `-> dict` to `-> dict | str`. Existing tools continue to return dicts; rich-media-capable tools return strings.

### 5.2 Sanitisation whitelist

`OutputService` (or wherever `nh3.clean()` is called for assistant text — see §8.1) configures nh3 to allow:

- Tag: `span`
- Attribute on `span`: `id` (only)

The id attribute value is otherwise unconstrained at sanitisation time; the parser regex (§5.3) is the structural validator. Other span attributes (`class`, `style`, `onclick`, …) are stripped exactly as they are today.

Rationale: `<span id="…">` is inert HTML — no scripts, no styles, no event handlers. The id namespace is parser-controlled. This is the smallest sanitiser change that solves the problem.

### 5.3 Persistence

| Surface | Holds |
|---|---|
| `transcript.content` | The sanitised LLM response, **with `<span id='…'>` intact**. |
| `tool_calls.result` | The tool's full return string (data JSON + instruction trailer if present). |
| anywhere else | Nothing. No schema change, no new column, no new table. |

This is a strict requirement: refreshing the page must rebuild cards exclusively from these two existing surfaces.

### 5.4 RichMediaParser

A new module `backend/services/rich_media_parser.py` exposing one pure function:

```python
def parse(content: str, tool_calls: list[ToolCallRow]) -> list[Segment]
```

**Algorithm (~80 LOC including helpers):**

```python
TAG_RE = re.compile(
    r"<span\s+id=['\"]([a-z][a-z0-9_]*)_(\d+)['\"]\s*>(.*?)</span>",
    re.DOTALL | re.IGNORECASE,
)

def parse(content, tool_calls):
    segments = []
    cursor = 0
    for m in TAG_RE.finditer(content):
        if m.start() > cursor:
            head = content[cursor:m.start()].strip()
            if head:
                segments.append({"type": "text", "content": head})
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
            if synthesis:
                segments.append({"type": "text", "content": synthesis})
            log.warning("rich_media: orphan tag %s", tag)
        cursor = m.end()
    tail = content[cursor:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})
    return segments
```

`_find_payload(tag, tool_calls)` scans `tool_calls` for a row whose `result` contains the literal substring `<span id='<tag>'>` (or `id="<tag>"`) — i.e. the same instruction string the tool gave the LLM. Pairing therefore happens via the LLM-visible tag name appearing in both the tool's result and the LLM's response — no separate ID system, no framework metadata.

`_extract_data(result)` splits on the first `\n\n` and returns the head as parsed JSON if possible, else the raw head string. Tools must therefore put their structured data on the first JSON segment of their return, with the instruction trailer separated by a blank line.

**Invariant — single data source:** Both the live path and the refresh path read `tool_calls` from the database, not from in-memory state. The `MessageProcessor`'s atomic store of pending tool_calls completes before the WS `message` event is assembled, so by the time the parser runs the rows are durably persisted under the assistant turn's `transcript_id`. This guarantees the live and refresh paths produce byte-identical segment arrays.

The refresh-path query MUST include rows where `tool_calls.ephemeral = 1` — inline tool_calls (the rows written by `ToolRenderAndRecordService`) carry that flag and are the rows that hold the rich-media instruction trailers. Filtering them out would break refresh rendering.

### 5.5 Edge cases

All of the following result in a **silent strip / passthrough to plain text** (with a warning log; no user-visible error stub, no LLM feedback signal):

| Failure | Behaviour |
|---|---|
| LLM emits `<span id='weather_1'>` but no weather tool ran this turn (orphan / hallucinated). | `_find_payload` returns `None` → synthesis becomes a `text` segment. |
| LLM emits `<span id='weather_3'>` but only 2 weather calls happened. | Same as above. |
| LLM forgets the closing `</span>`. | `TAG_RE` non-greedy `.*?` does not match → entire span passes through as text. |
| LLM nests an unrelated `<span>` inside a rich-media span. | Inner span has no recognised id pattern → the outer `</span>` close still terminates the outer match (sanitiser already balances tags); inner content stays as part of the synthesis. |
| Tag prefix with no frontend module registered. | Frontend registry returns plain-text fallback (§6.2). |

Rationale for silent over loud: by the time the parser runs, the response has already been generated; loud failures would either show error stubs to end users or leak parser noise into the transcript. The drift is observable in logs.

### 5.6 WebSocket message shape

```json
{
  "type": "message",
  "content": "<sanitised LLM text with <span id='weather_1'>…</span> intact>",
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

`content` remains for backwards compatibility (metrics consumers, debug tooling). The frontend renders exclusively from `segments`. For non-rich responses, `segments` is `[{type: "text", content: <sanitised>}]`.

There are **three** assembly sites in `backend/api/websocket.py` where `message_evt` is built today (around lines 415, 719, 760). All three must include `segments`. A small helper (`_attach_segments(message_evt, transcript_id)` or similar) avoids duplicating the parse call across sites.

### 5.7 Per-turn ordinal counter

`backend/services/act_dispatcher_service.py` (the dispatch chokepoint, called as `dispatch_action` per tool call) maintains a per-turn `dict[str, int]` counter scoped to the current `MessageProcessor` instance. Before invoking a tool's `execute()`, it bumps the counter for that tool name and exposes the new value via the dispatch context the tool already receives. Tools that opt into rich-media read `ctx.rich_media_ordinal` (or equivalent name) to fill in their instruction trailer.

The counter resets at the start of each ACT loop iteration's dispatch round, matching the spec's "ordinal per tool, per turn".

### 5.8 Refresh path

The existing `/conversation/recent` route in `backend/api/conversation.py` reads from `transcript`. Two changes:

1. The query filter `role NOT IN ('subagent_return')` stays; the WHERE clause should not exclude any rows that already exist there.
2. For each assistant row returned, fetch its `tool_calls` rows (`WHERE transcript_id = ?`, including `ephemeral = 1`), run `RichMediaParser.parse(content, tool_calls)`, and include `segments` on the response item.

Frontend rendering on refresh becomes byte-identical to live rendering.

## 6. Frontend protocol

### 6.1 Renderer changes

Both append (`_appendMessage`) and prepend (`_prependMessage`, used by scroll-back pagination from `/conversation/recent`) paths in `frontend/interface/chat.js` change. The renderer entry point in `frontend/interface/renderer.js` today is `appendChalieForm(content, meta = {}, opts = {})` — taking a raw content string plus a meta object.

**Decision:** widen `appendChalieForm` (and the matching `prependChalieForm`) to also accept `meta.segments`. When `segments` is present, the renderer iterates segments. When absent, fall back to the existing single-bubble path using `content`. The signature stays `(content, meta, opts)` — `segments` is just a new optional key on `meta`.

```js
function appendChalieForm(content, meta = {}, opts = {}) {
  const segments = meta.segments
    || [{type: "text", content: content || ""}];
  for (const seg of segments) {
    if (seg.type === "text") {
      appendTextBubble(seg.content, meta, opts);
    } else if (seg.type === "rich") {
      appendRichCard(seg.tag, seg.payload, seg.synthesis, meta, opts);
    }
  }
}
```

Each segment becomes its own DOM element — multiple text bubbles per assistant turn are explicitly supported and visually equivalent to today's single bubble repeated. Cards render as siblings to bubbles.

If `segments` is missing on the meta (legacy clients, non-rich responses), the renderer falls back to a single text bubble using `content` — making this a backward-compatible change for any consumer that already calls `appendChalieForm(content, meta, opts)` with a meta object that doesn't contain `segments`.

`chat.js` is updated at both `_appendMessage` and `_prependMessage` to pass `segments` from the WS event / `/conversation/recent` row through `meta`.

> ⚠️ **Nightly-test fallout expected.** The renderer change is structural — turning a single `.message-chalie` bubble into possibly multiple bubbles + cards per turn. Several nightly scenarios assert against single-bubble DOM shapes (e.g. counting `.message-chalie` elements, asserting full assistant text in one node). Acknowledged, accepted, and will be reconciled scenario-by-scenario when they break. The non-rich path should remain stable since it still emits exactly one text segment.

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

### 7.1 Tool payload shape (current — preserved)

The weather tool's existing return shape is preserved verbatim. The card adapts to it; no field renames, no restructuring. The data portion of `tool_calls.result` (everything before the instruction trailer) is the JSON-serialised version of the current dict returned by `WeatherAbility.execute()`:

```json
{
  "location": "London, GB",
  "condition": "Partly cloudy",
  "temperature_c": 12.4,
  "temperature_f": 54.3,
  "feels_like_c": 10.1,
  "humidity_pct": 78,
  "wind_kmh": 14.2,
  "wind_direction": "WSW",
  "visibility_km": null,
  "uv_index": null,
  "precip_mm": 0.0,
  "observation_time": "2026-05-02T14:00",
  "is_raining": false,
  "is_daylight": true,
  "is_hot": false,
  "is_cold": false,
  "is_windy": false,
  "is_clear": false,
  "forecast_tomorrow_condition": "Slight rain",
  "forecast_tomorrow_max_c": 14.0,
  "forecast_tomorrow_min_c": 9.0,
  "forecast_tomorrow_precip_chance_pct": 70,
  "forecast_tomorrow_precip_mm": 3.2
}
```

The card derives a semantic icon key on the **frontend** from `(condition, is_daylight, is_raining, is_clear)`; no extra `icon` field is added to the backend payload. The mapping table lives in `frontend/interface/rich_media/weather.js` and is straightforward: `condition` substring match → key (`sunny` / `cloudy` / `partly_cloudy` / `rain` / `snow` / `storm` / `fog`), with `is_daylight=false` swapping the day variant for night where one exists.

The forecast section is a single day (tomorrow) for v1 — matching what the tool already produces. A future iteration can extend the tool to a 3-day forecast and the card to render it; that is explicitly out of scope here.

### 7.2 Tool change — minimal surface

The only change to `backend/abilities/weather.py`:

1. Return type widens from `dict` to `str`. The dict result is `json.dumps(...)`'d, then concatenated with `\n\n` and the rich-media instruction trailer carrying the dispatcher-provided ordinal.
2. The error-path return (line 169) is also serialised the same way (JSON with `error`/`details`, no instruction trailer — the LLM should not be told to render a card from an error payload).
3. Stale-cache and fresh-cache paths use the same serialisation helper.

A small helper `_serialise(payload: dict, ordinal: int | None) -> str` keeps the body of `execute()` thin. Everything else in the file is untouched.

### 7.3 Card visual (sketch)

```
┌─────────────────────────────────────────┐
│ London, GB                              │
│                                         │
│  [icon]  12°C                           │
│          Partly cloudy                  │
│          Feels 10° · 78% · 14 km/h      │
│                                         │
│ ─────────────────────────────────────── │
│ Tomorrow  14°/9°  Slight rain · 70%     │
│ ─────────────────────────────────────── │
│  "{LLM synthesis text}"                 │
└─────────────────────────────────────────┘
```

The synthesis renders below the data block in a slightly muted style — visually distinct from the data, so it reads as Chalie's interpretation rather than raw values. Card chrome uses Radiant variables: near-black background, 1px border with subtle violet/cyan accent on the icon, soft glow on hover.

## 8. File-level integration map

### 8.1 Backend — modified

| File | Change |
|---|---|
| `backend/services/rich_media_parser.py` | **NEW.** ~80 LOC. Pure parsing function described in §5.4. Unit-tested in isolation. |
| `backend/services/markup.py` | Single sanitiser chokepoint. Add `span` to `LLM_TAGS` and `"span": {"id"}` to `_ATTRIBUTES`. **No other tags / attrs added.** TTS / `extract_plaintext` is unaffected (strips all tags by design). |
| `backend/services/act_dispatcher_service.py` | Per-turn `dict[str, int]` counter on the dispatcher; bumped before each tool dispatch; ordinal exposed on the dispatch context. (§5.7) |
| `backend/abilities/_base.py` | `Ability.execute()` return-type annotation widens from `-> dict` to `-> dict | str`. |
| `backend/api/websocket.py` | All three `message_evt` assembly sites (≈ lines 415, 719, 760) gain `"segments"`. A single helper `_attach_segments(message_evt, transcript_id)` keeps it DRY. |
| `backend/api/conversation.py` | `/conversation/recent` fetches `tool_calls` per assistant transcript row (including `ephemeral = 1`), runs `RichMediaParser.parse(content, tool_calls)`, includes `segments` on each item. |
| `backend/abilities/weather.py` | Minimum-touch: serialise the existing dict via a `_serialise(payload, ordinal)` helper and append the rich-media instruction trailer. No field renames, no shape changes. (§7.2) |
| `backend/schema.sql` | **NO CHANGE.** Confirmed — `transcript.content` and `tool_calls.result` already hold everything needed. |

### 8.2 Backend — tests

| File | Purpose |
|---|---|
| `backend/tests/test_rich_media_parser.py` | **NEW.** Deterministic unit tests: single span, multiple interleaved spans, orphan span, unclosed span, span with mismatched-but-valid id pattern, empty synthesis, no spans at all, span with double-quoted id. |
| `backend/tests/test_message_processor_rich_media.py` | **NEW.** Feature test: weather tool result + faked LLM response with `<span id='weather_1'>` → segment array on outgoing WS event. |
| `backend/tests/test_conversation_recent_rich_media.py` | **NEW.** Feature test: persisted transcript + tool_calls (including `ephemeral=1`) → `/conversation/recent` returns segments byte-identical to what the live path produced. |
| `backend/tests/test_output_service_span_whitelist.py` | **NEW.** Sanitisation test: `<span id='weather_1'>` survives nh3, `<span class='evil' onclick='…'>` does not, raw `<script>` does not. |

### 8.3 Frontend — new

| File | Purpose |
|---|---|
| `frontend/interface/rich_media/registry.js` | Tag-prefix → card module map (~20 LOC). |
| `frontend/interface/rich_media/weather.js` | Weather card module: `render(payload, synthesis, root)`, ~150 LOC including animation hooks and the `(condition, is_daylight, …) → icon_key` mapping. |
| `frontend/interface/rich_media/weather.css` | Card styling using Radiant variables. |
| `frontend/interface/rich_media/base_card.css` | Shared card chrome (border, padding, entrance keyframes, hover glow). |
| `frontend/interface/rich_media/icons/weather/*.svg` | 6–10 semantic SVG icons. |

### 8.4 Frontend — modified

| File | Change |
|---|---|
| `frontend/interface/renderer.js` | `appendChalieForm(content, meta, opts)` and `prependChalieForm(content, meta, opts)` switch from "render `content` once" to "iterate `meta.segments`". Falls back to single-bubble rendering when `meta.segments` is missing. |
| `frontend/interface/chat.js` | Both `_appendMessage` and `_prependMessage` pass `segments` from the WS / `/conversation/recent` payload through `meta`. |
| `backend/api/__init__.py` (HTML serve / asset versioning) | The new `frontend/interface/rich_media/` modules ship through the same `?v=VERSION` mechanism the rest of `frontend/interface/` uses. No registration required — relative imports resolve naturally. |

### 8.5 Documentation

| File | Change |
|---|---|
| `docs/04-ARCHITECTURE.md` | New short subsection "Rich Media Cards" describing the parser chokepoint, segment shape, and refresh path. |
| `docs/03-WEB-INTERFACE.md` | Append "Rich media cards" to the Radiant design system, referencing the entrance-only animation rule. |
| `backend/abilities/weather.py` (docstring) | Document the rich-media instruction string and the JSON-on-first-line convention so future rich-media tools follow the same pattern. |

### 8.6 Estimated size

- Backend: ~250 LOC new (parser + helpers + tests), ~30 LOC of edits.
- Frontend: ~300 LOC new (renderer + weather card + CSS), ~20 LOC of edits.
- Schema migrations: zero.
- Sanitiser config: +1 tag, +1 attribute.

## 9. Acceptance criteria

The feature is complete when all of the following are true:

1. Asking Chalie about the weather (via the existing weather tool) renders a card matching §7.3, with the LLM's synthesis displayed beneath the data.
2. Asking Chalie about the weather in two cities in the same turn renders two distinct cards, each paired with its city's data; the prose between them renders as separate text bubbles.
3. Refreshing the page after a weather conversation rebuilds the same cards from the database; the rebuilt cards are visually indistinguishable from the live-rendered cards.
4. The LLM omitting tags (responding in pure prose) renders normally as one or more text bubbles.
5. The LLM emitting an orphan / mismatched / unclosed span results in plain-text rendering with a warning logged backend-side. No user-visible error stub.
6. A new tool can be made rich-media-capable by editing only its own `execute()` return string and adding a frontend module + registry entry. No framework or schema changes.
7. All new unit and feature tests pass under `pytest -m unit`.
8. `/conversation/recent` response shape gains `segments` per assistant row; existing consumers reading `content` continue to work unchanged.
9. Sanitisation tests confirm `<span id='…'>` survives `nh3.clean()` and that no other span attributes (`class`, `style`, `onclick`, …) leak through.

## 10. Open questions

None. All Q1–Q9 from the brainstorming session and the post-research reconciliation (parser placement, weather minimum-touch, renderer signature, prepend path) are resolved and reflected in this document.

## 11. Revision log

- **2026-05-02 (initial draft, commit 13bdd01):** First version with `[tool_N]…[/tool_N]` bracket syntax, parser at WS-send boundary as if raw text were available there.
- **2026-05-02 (post-research reconciliation, this doc):** Tag syntax switched to `<span id='tool_N'>…</span>` after research found WS `content` is sanitised before assembly. nh3 whitelist for span+id added (§5.2). Parser placement clarified — runs against the post-sanitisation text, which now retains the spans (§4, §5.4). Three WS assembly sites called out (§5.6). Per-turn ordinal counter relocated to `act_dispatcher_service.py` (§5.7). Refresh-path `ephemeral=1` inclusion noted (§5.4, §5.8). Weather tool reduced to minimum-touch — current dict shape preserved, only the return type widens to string with instruction trailer (§7.1, §7.2). Renderer signature reconciled to `(content, meta, opts)` for both `appendChalieForm` and `prependChalieForm`, with `segments` carried on `meta` (§6.1). Nightly-test fallout explicitly acknowledged (§6.1). `Ability._base.py` return-type annotation widened (§5.1, §8.1). New sanitisation test added (§8.2).
- **2026-05-02 (round-1 fixes, commits be0074e + 04f8001):** Pre-critic nightly run 429 surfaced two implementation gaps and one LLM-compliance gap. (a) WS lookup `_fetch_tool_calls_for_recent_user_turn` filtered `role='user'` and used recency, missing subagent rows and racing the DMN daemon — replaced with `_fetch_tool_calls_for_transcript_ids(ids)`. `transcript_ids: list` is now threaded through `OutputService.enqueue_text` metadata from `MessageProcessor._uid`. The deprecated stub was deleted in 04f8001 with regression sentinels added in `test_rich_media_ws_fetch.py` for the channel/role-filter and DMN-race cases. (b) `reset_turn_ordinals()` was unwired (per-turn freshness was incidental to fresh-dispatcher-per-turn) — now called explicitly at the top of `MessageProcessor.send()` to make the contract enforceable. (c) The LLM ignored the inline trailer and responded in plain prose — strengthened the trailer in `weather.py` (imperative "MUST", consequence sentence, concrete example) AND added Operational Principle #9 to `UnifiedSystemMessagePrompt` explicitly directing the LLM to honor rich-media trailers. Run 430 evidence: 058 PASSED end-to-end (LLM emits span, segments correct, refresh returns rich segment). Step-4 `contains` check vs JSON-vs-Python-dict-repr is a harness-level false negative — surfaced separately; scenarios remain LOCKED.
- **2026-05-02 (round-2 fix, commit 14b6618):** Run 430 surfaced a subagent-boundary inheritance gap. In scenario 059 the subagent correctly produced both spans (`weather_1` for Paris, `weather_2` for Tokyo) but the parent paraphrased them away in its final synthesis. Operational Principle #9 was extended with an "Inheritance from subagents and tools" clause: when a tool result or subagent output already contains `<span id='name_N'>…</span>` tags, the LLM MUST preserve those exact tags verbatim — same id, same content — somewhere in its final response. Reword surrounding prose freely; never paraphrase away an existing span.
