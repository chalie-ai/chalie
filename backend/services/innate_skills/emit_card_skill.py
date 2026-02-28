"""
Emit Card Skill — Generic deferred card renderer.

Retrieves cached tool data by invocation_id (or most-recent fallback), merges
with Chalie's summary/response, renders the tool's card template via
CardRendererService, and enqueues via OutputService.

Works for any tool that sets card.mode: "deferred" in its manifest — no tool
names are referenced here.
"""

import json
import logging
import urllib.parse

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EMIT CARD]"


def _extract_domain(url: str) -> str:
    """Extract bare domain from URL, stripping www. prefix."""
    try:
        netloc = urllib.parse.urlparse(url).netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or url
    except Exception:
        return url


def handle_emit_card(topic: str, params: dict) -> dict:
    """
    Render and enqueue a deferred tool card.

    Args:
        topic: Current conversation topic (used for Redis key lookup)
        params: {
            "summary"      (required) — one-line summary from Chalie
            "response"     (required) — detailed analysis from Chalie
            "invocation_id" (optional) — specific cached invocation to render;
                             if omitted, the most recent deferred card is used
        }

    Returns:
        {"card_emitted": True, "tool": tool_name}  on success
        str error message                           on failure
    """
    from services.redis_client import RedisClientService
    from services.tool_registry_service import ToolRegistryService
    from services.card_renderer_service import CardRendererService
    from services.output_service import OutputService

    summary = params.get("summary", "").strip()
    response_text = params.get("response", "").strip()
    invocation_id = params.get("invocation_id", "").strip()

    if not summary:
        return "Error: summary is required"

    redis = RedisClientService.create_connection()

    # ── 1. Resolve the cached tool result ────────────────────────────────────
    entry = None

    if invocation_id:
        raw = redis.get(f"tool_card_cache:{topic}:{invocation_id}")
        if raw:
            try:
                entry = json.loads(raw)
            except Exception:
                logger.warning(f"{LOG_PREFIX} Failed to parse cache entry for {invocation_id}")

    if not entry:
        # Fallback: use the most recent deferred card registered for this topic
        deferred_items = redis.lrange(f"deferred_cards:{topic}", 0, -1)
        if deferred_items:
            try:
                last_deferred = json.loads(deferred_items[-1])
                fallback_id = last_deferred.get("invocation_id", "")
                if fallback_id:
                    raw = redis.get(f"tool_card_cache:{topic}:{fallback_id}")
                    if raw:
                        entry = json.loads(raw)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Deferred-card fallback failed: {e}")

    if not entry:
        logger.warning(f"{LOG_PREFIX} No cached tool result for topic {topic!r}")
        return "Error: no cached tool result found for card rendering"

    tool_name = entry.get("tool", "")
    tool_data = entry.get("data", {})

    # ── 2. Look up the tool manifest ─────────────────────────────────────────
    try:
        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Tool registry unavailable: {e}")
        return f"Error: tool registry unavailable — {e}"

    if not tool:
        return f"Error: tool '{tool_name}' not found in registry"

    card_config = tool["manifest"].get("output", {}).get("card", {})

    # ── 3. Merge Chalie's synthesis with the tool's structured data ───────────
    # Prepare images: add "display" field so the carousel template can set
    # the first slide visible (display:flex) and the rest hidden (display:none)
    # without needing tool-specific JavaScript.
    raw_images = tool_data.get("images", [])
    images_with_display = [
        {**img, "display": "flex" if i == 0 else "none"}
        for i, img in enumerate(raw_images)
    ]

    # Add domain + pipe_before to each result for the sources row in the template.
    # pipe_before is empty for the first item and " | " for all subsequent ones,
    # producing "google.com | amazon.com | reddit.com" without a trailing separator.
    raw_results = tool_data.get("results", [])
    results_with_domain = [
        {
            **r,
            "domain": _extract_domain(r.get("url", "")),
            "pipe_before": "" if i == 0 else " | ",
        }
        for i, r in enumerate(raw_results)
    ]

    # response_lines is a one-item list when response is present, empty otherwise.
    # The template uses {{#response_lines}}...{{/response_lines}} to conditionally
    # render the "Read more" section — the renderer treats any section as a list loop.
    response_lines = [{"text": response_text}] if response_text else []

    merged_data = {
        **tool_data,
        "summary": summary,
        "response": response_text,
        "response_lines": response_lines,
        "source_count": len(raw_results),
        "images": images_with_display,
        "results": results_with_domain,
    }

    # ── 4. Render card via the tool's card/template.html + styles.css ────────
    try:
        renderer = CardRendererService()
        card_data = renderer.render(tool_name, merged_data, card_config, tool["dir"])
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Card render failed for {tool_name}: {e}")
        return f"Error: card rendering failed — {e}"

    if not card_data:
        return "Error: card rendering produced no output (template may be missing)"

    # ── 5. Enqueue card for delivery via drift stream ─────────────────────────
    try:
        OutputService().enqueue_card(topic, card_data, {})
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Card enqueue failed: {e}")
        return f"Error: card delivery failed — {e}"

    # ── 6. Clean up deferred state so subsequent emit_card calls don't recycle
    try:
        redis.delete(f"deferred_cards:{topic}")
    except Exception:
        pass

    used_id = invocation_id or entry.get("invocation_id", "latest")
    logger.info(f"{LOG_PREFIX} Card emitted for {tool_name} (invocation: {used_id})")

    return {"card_emitted": True, "tool": tool_name}
