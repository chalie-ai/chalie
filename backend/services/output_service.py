import json
import logging
import uuid
from .memory_client import MemoryClientService
from .time_utils import utc_now
from .markup import actions_to_xml, sanitize

logger = logging.getLogger(__name__)

_KEY_NOTIFICATIONS_RECENT = 'notifications:recent'


class OutputService:
    """Service for managing output queue and storage with type-based routing."""

    def __init__(self):
        """Initialize the OutputService with MemoryStore connection."""
        self.store = MemoryClientService.create_connection()

    def enqueue_text(
        self,
        topic: str = None,
        response: str = '',
        mode: str = 'UNIFIED',
        confidence: float = 0.0,
        generation_time: float = 0.0,
        original_metadata: dict = None,
        reply_actions: list = None,
        channel: str = None,
        metrics: dict = None,
        transcript_ids: list = None,
    ) -> str:
        """
        Enqueue a TEXT output for delivery via SSE to the web interface.

        Args:
            topic: Deprecated alias for channel. Use channel instead.
            channel: Conversation channel identifier
            response: The response text to deliver
            mode: Output mode (UNIFIED, ACT)
            confidence: Confidence score of the response
            generation_time: Time taken to generate the response
            original_metadata: Optional original metadata from the request (uuid, source, etc.)
            reply_actions: Optional action buttons for the frontend (only delivered on sync chat, never drift)
            transcript_ids: Optional list of transcript row IDs that generated this response.
                Forwarded to the WS handler so tool_calls can be fetched by exact row IDs
                rather than by recency heuristic. When absent the WS handler falls back to
                no segments (plain text bubble).

        Returns:
            str: UUID of the enqueued output
        """
        # Accept topic as backward-compat alias for channel
        _channel = channel or topic

        output_id = str(uuid.uuid4())
        metadata_dict = {
            "response": response,
            "mode": mode,
            "confidence": confidence,
            "generation_time": generation_time,
            "metadata": original_metadata or {}
        }
        if reply_actions:
            metadata_dict["reply_actions"] = reply_actions
        if metrics is not None:
            metadata_dict["metrics"] = metrics
        if transcript_ids is not None:
            metadata_dict["transcript_ids"] = transcript_ids

        # Single chokepoint: sanitize() accepts mixed plain text + allowlisted
        # HTML and passes both through. Programmatic action buttons are
        # appended in their own XML form (already trusted, model-free).
        content = sanitize(response or "")
        if reply_actions:
            content = content + actions_to_xml(reply_actions)
        metadata_dict["content"] = content

        output = {
            "id": output_id,
            "type": "TEXT",
            "topic": _channel,
            "created_at": utc_now().isoformat(),
            "metadata": metadata_dict
        }

        # Store with 1-hour TTL
        self.store.setex(f"output:{output_id}", 3600, json.dumps(output))

        meta = original_metadata or {}
        source = meta.get('source', '')
        sse_channel = meta.get('uuid')

        # Map source to event type
        source_type_map = {
            'tool_result': 'response',
            'reminder': 'reminder',
            'task': 'task',
            'scheduled_prompt': 'task',
            'critic_escalation': 'escalation',
            'notification': 'notification',
            'plan_action': 'task',
        }
        event_type = source_type_map.get(source, 'response')

        event_payload_dict = {
            'output_id': output_id,
            'type': event_type,
            'topic': _channel,
            'content': content,
            'mode': mode,
            'confidence': confidence,
            'generated_at': output['created_at'],
        }
        if metrics is not None:
            event_payload_dict['metrics'] = metrics

        event_payload = json.dumps(event_payload_dict)

        if sse_channel:
            # Deliver via per-request channel for sync /chat SSE connections.
            # Do NOT publish to output:events — that causes the drift stream to
            # render a duplicate after _isSending resets to false.
            self.store.publish(f"sse:{sse_channel}", output_id)
        else:
            # Background output: publish event, web push, and catch-up buffer.
            self.store.publish('output:events', event_payload)

            try:
                from api.push import send_push_to_all
                send_push_to_all(title='Chalie', body=response[:200])
            except Exception as e:
                logger.warning(f"Web push dispatch failed: {e}")

            # Buffer for catch-up: events published during a brief drift stream
            # reconnect gap are permanently lost from pub/sub. Push to a list so
            # the stream endpoint can drain missed events on next connect.
            try:
                self.store.rpush(_KEY_NOTIFICATIONS_RECENT, event_payload)
                self.store.ltrim(_KEY_NOTIFICATIONS_RECENT, -200, -1)
                self.store.expire(_KEY_NOTIFICATIONS_RECENT, 86400)  # 24h TTL
            except Exception as e:
                logger.warning(f"Notification buffer push failed: {e}")

        logger.info(
            f"Enqueued TEXT output {output_id} for topic '{_channel}' "
            f"(mode={mode}, confidence={confidence:.2f})"
        )

        return output_id

    def enqueue_proactive(
        self,
        topic: str,
        response: str,
        source: str = 'task',
        metrics: dict = None,
    ) -> str:
        """
        Enqueue a proactive/background output (goal pursuits, reminders, etc.).

        Convenience wrapper around enqueue_text() with sensible defaults for
        messages that originate from background workers rather than user requests.

        Args:
            topic: Conversation topic / thread identifier
            response: The message text to deliver
            source: Source identifier for SSE event type mapping
            metrics: Optional per-turn metrics dict (tokens_total, tools,
                response_time_s, …) for DMN / goal-pursuit / scheduled flows.

        Returns:
            str: UUID of the enqueued output
        """
        metadata = {'source': source}
        return self.enqueue_text(
            topic=topic,
            response=response,
            mode='UNIFIED',
            confidence=1.0,
            generation_time=0.0,
            original_metadata=metadata,
            metrics=metrics,
        )


