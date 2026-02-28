## `emit_card` — Display a Visual Card

Render a rich visual card from the most recent deferred tool result. Combines your synthesis with the tool's structured data (sources, images, etc.) into a visually rich, collapsible card delivered to the user.

Parameters:
- `summary` (required): One-line synthesis of findings — this is the headline the user sees first
- `response` (required): Your full detailed response — exactly what you would say in a text reply. This is shown collapsed under "Read more". **Must be non-empty.** Since the card suppresses the text follow-up, `response` is the only place your analysis appears.
- `invocation_id` (optional): Specific tool invocation ID shown in the card offer. Use this when multiple deferred card offers are available to select the right data.

### Decision Guidance

**Prefer card** when:
- 3+ sources from different domains — a visual card organises them better than prose
- Images are available — they enrich the response visually
- The user asked a broad, research-oriented, or exploratory question
- The topic benefits from scannable sources (news, comparisons, factual overview)

**Prefer text** when:
- Simple factual answer ("What time is it in Tokyo?" / "Who wrote Hamlet?")
- 1–2 sources and the answer is short — text is more direct
- Search returned errors, empty results, or very sparse data
- The user asked something conversational, not research-based

### Behaviour

When called, this skill:
1. Retrieves the cached tool result (by invocation_id, or most recent if omitted)
2. Merges your summary and response with the tool's data (results, images)
3. Renders the tool's card template
4. Delivers the card instantly via the drift stream
5. Suppresses the text follow-up — the card IS the response

**After calling emit_card, return empty actions on the next iteration.**
