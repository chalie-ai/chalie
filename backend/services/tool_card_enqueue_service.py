"""
Tool Card Enqueue Service — Renders and delivers cards for card-enabled tools.

Extracted from tool_worker.py to reduce the god function and centralize
card rendering logic.

Fixes:
  B7: Skips deferred tools (handled by emit_card during the loop)
  B8: Per-topic rendered_cards set prevents duplicate on critic retry
"""

import json
import logging

from services import act_redis_keys

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CARD ENQUEUE]"


def enqueue_tool_cards(act_history: list, topic: str, metadata: dict,
                       cycle_id: str = None) -> bool:
    """Render and enqueue cards for card-enabled tools.

    Returns True if any synthesize=false tool was found (suppresses the text
    follow-up since the card was already rendered inline).

    synthesize=false tools: invoke() already rendered the card. Here we just
    close the SSE.

    synthesize=true tools: invoke() deferred card emission here. We render
    exactly once per tool (using the last cached result) even if the tool was
    called multiple times in the ACT loop (deduplication via rendered_tools set).

    B7 fix: Tools with card.mode=="deferred" are skipped here — they are
    rendered by the emit_card skill during the loop. Rendering them again
    from tool_raw_cache would produce duplicates.

    B8 fix: A per-topic Redis set (rendered_cards:{topic}, 60s TTL) prevents
    duplicate cards when the critic retries a synthesize=false tool.
    """
    any_synthesize_false = False
    try:
        from services.redis_client import RedisClientService
        from services.tool_registry_service import ToolRegistryService
        from services.card_renderer_service import CardRendererService
        from services.output_service import OutputService

        redis = RedisClientService.create_connection()
        raw_items = redis.lrange(act_redis_keys.tool_raw_cache(topic), 0, -1)
        redis.delete(act_redis_keys.tool_raw_cache(topic))

        # Build {tool_name: raw_result} map (last result per tool wins)
        raw_map = {}
        for item in raw_items:
            entry = json.loads(item)
            raw_map[entry['tool']] = entry['data']

        registry = ToolRegistryService()
        renderer = CardRendererService()
        output_svc = OutputService()

        # Track tools whose cards have already been enqueued to prevent duplicates.
        rendered_tools: set = set()

        # B7: Also skip tools that emit_card already rendered during the loop.
        for action in act_history:
            if (action.get('action_type') == 'emit_card'
                    and action.get('status') == 'success'
                    and isinstance(action.get('result'), dict)):
                rendered_by_emit = action['result'].get('tool_name')
                if rendered_by_emit:
                    rendered_tools.add(rendered_by_emit)

        # B8: Check per-topic rendered_cards Redis set for cross-invocation dedup
        rendered_key = act_redis_keys.rendered_cards(topic)
        already_rendered = redis.smembers(rendered_key)
        if already_rendered:
            rendered_tools.update(
                m.decode() if isinstance(m, bytes) else m
                for m in already_rendered
            )

        for action in act_history:
            if action.get('status') != 'success':
                continue
            tool_name = action.get('action_type', '')
            tool = registry.tools.get(tool_name)
            if not tool:
                continue
            output_config = tool['manifest'].get('output', {})
            card_config = output_config.get('card', {})
            if not card_config or not card_config.get('enabled'):
                continue

            # B7: Skip deferred-mode tools — emit_card handles them during the loop
            if card_config.get('mode') == 'deferred':
                continue

            # If synthesize is false, invoke() already rendered the card inline.
            # Suppress the text follow-up and close the waiting SSE connection.
            if not output_config.get('synthesize', True):
                any_synthesize_false = True
                sse_uuid = metadata.get('uuid')
                if sse_uuid:
                    try:
                        output_svc.enqueue_close_signal(sse_uuid)
                    except Exception as _re:
                        logger.warning(f"{LOG_PREFIX} Failed to send close signal: {_re}")
                continue

            # synthesize=true: render exactly once per tool name.
            if tool_name in rendered_tools:
                continue

            raw = raw_map.get(tool_name)
            if not raw:
                continue

            # Path A: tool returned inline HTML (formalized contract) — use render_tool_html()
            result_html = raw.get('html') if isinstance(raw, dict) else None
            if result_html:
                result_title = raw.get('title') or card_config.get('title', tool_name)
                card_data = renderer.render_tool_html(tool_name, result_html, result_title, card_config)
            else:
                # Path B: no inline HTML — fall back to template-based render()
                card_data = renderer.render(tool_name, raw, card_config, tool['dir'])

            if card_data:
                output_svc.enqueue_card(topic, card_data, metadata)
                rendered_tools.add(tool_name)
                # B8: Record in Redis so critic retries don't double-render
                redis.sadd(rendered_key, tool_name)
                redis.expire(rendered_key, 60)

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Card enqueue failed: {e}")

    return any_synthesize_false
