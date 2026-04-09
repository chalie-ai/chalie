import json
import logging
import uuid
from typing import Dict, Any, Optional, List

from .memory_client import MemoryClientService
from .config_service import ConfigService
from .time_utils import utc_now
from .blocks_render_service import BlocksRenderService

_blocks_svc = BlocksRenderService()

logger = logging.getLogger(__name__)


class OutputService:
    """Service for managing output queue and storage with type-based routing."""

    def __init__(self):
        """Initialize the OutputService with MemoryStore connection and config."""
        self.store = MemoryClientService.create_connection()
        config = ConfigService.connections()
        topics = config.get("memory", {}).get("topics", {})
        self.queue_name = topics.get("output_queue", "output-queue")

        logger.info(f"OutputService initialized with queue: {self.queue_name}")

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

        # Convert response to blocks — the sole output format for all clients
        blocks = _blocks_svc.from_markdown(response)
        if reply_actions:
            blocks.extend(_blocks_svc.from_actions(reply_actions))
        metadata_dict["blocks"] = blocks

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
            'goal_pursuit': 'task',
            'critic_escalation': 'escalation',
            'notification': 'notification',
            'plan_action': 'task',
        }
        event_type = source_type_map.get(source, 'response')

        event_payload_dict = {
            'output_id': output_id,
            'type': event_type,
            'topic': _channel,
            'blocks': blocks,
            'mode': mode,
            'confidence': confidence,
            'generated_at': output['created_at'],
        }

        event_payload = json.dumps(event_payload_dict)

        # Only publish to output:events for background outputs (no sync SSE channel).
        # When a sync /chat SSE connection is open it delivers the response directly;
        # publishing to output:events as well causes the drift stream to render a
        # duplicate after _isSending resets to false.
        if not sse_channel:
            self.store.publish('output:events', event_payload)

        # Deliver via per-request channel for sync /chat SSE connections
        if sse_channel:
            self.store.publish(f"sse:{sse_channel}", output_id)

        # Web push + catch-up buffer for all background output (no sync SSE connection)
        if not sse_channel:
            try:
                from api.push import send_push_to_all
                send_push_to_all(
                    title='Chalie',
                    body=response[:200] if len(response) > 200 else response,
                )
            except Exception as e:
                logger.warning(f"Web push dispatch failed: {e}")

            # Buffer for catch-up: events published during a brief drift stream
            # reconnect gap are permanently lost from pub/sub. Push to a list so
            # the stream endpoint can drain missed events on next connect.
            try:
                self.store.rpush('notifications:recent', event_payload)
                self.store.ltrim('notifications:recent', -200, -1)
                self.store.expire('notifications:recent', 86400)  # 24h TTL
            except Exception as e:
                logger.warning(f"Notification buffer push failed: {e}")

        # Notification tools (Telegram etc.) for drift and all background events
        if source in ('reminder', 'task', 'goal_pursuit') or source.startswith('cron_tool:'):
            self._send_to_notification_tools(response)

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
    ) -> str:
        """
        Enqueue a proactive/background output (persistent tasks, reminders, etc.).

        Convenience wrapper around enqueue_text() with sensible defaults for
        messages that originate from background workers rather than user requests.

        Args:
            topic: Conversation topic / thread identifier
            response: The message text to deliver
            source: Source identifier for SSE event type mapping

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
        )

    def _send_to_notification_tools(self, message: str) -> None:
        """
        Deliver proactive drift to all tools that declare notification support.

        Discovers tools at call time via ToolRegistryService. Any tool with
        "notification": {"default_enabled": true} in its manifest will receive
        the message via registry.invoke() (routed through Docker container).
        """
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            for tool in registry.get_notification_tools():
                try:
                    result_str = registry.invoke(tool["name"], "__notification__", {"message": message})
                    if "Error:" in result_str:
                        logger.warning(f"Notification tool '{tool['name']}' failed: {result_str}")
                except Exception as e:
                    logger.warning(f"Notification tool '{tool['name']}' error: {e}")
        except Exception as e:
            logger.warning(f"Notification dispatch error: {e}")

    def enqueue_act(
        self,
        topic: str = None,
        actions: List[str] = None,
        downstream_mode: str = 'ACT',
        act_history: List[Dict[str, Any]] = None,
        loop_id: str = '',
        generation_time: float = 0.0,
        channel: str = None,
    ) -> str:
        """
        Enqueue an ACT output for action processing.

        Args:
            topic: Conversation topic identifier
            actions: List of actions to execute
            downstream_mode: Mode to use after action processing
            act_history: History of previous actions in this cycle
            loop_id: Identifier for the action loop
            generation_time: Time taken to generate the actions

        Returns:
            str: UUID of the enqueued output
        """
        _channel = channel or topic
        actions = actions or []
        act_history = act_history or []
        output_id = str(uuid.uuid4())
        output = {
            "id": output_id,
            "type": "ACT",
            "topic": _channel,
            "created_at": utc_now().isoformat(),
            "metadata": {
                "actions": actions,
                "downstream_mode": downstream_mode,
                "act_history": act_history,
                "loop_id": loop_id,
                "generation_time": generation_time
            }
        }

        # Store with 1-hour TTL
        self.store.setex(f"output:{output_id}", 3600, json.dumps(output))

        # Add to queue
        self.store.lpush(self.queue_name, output_id)
        self.store.expire(self.queue_name, 3600)

        logger.info(
            f"Enqueued ACT output {output_id} for topic '{_channel}' "
            f"(loop_id={loop_id}, actions={len(actions)})"
        )

        return output_id

    def dequeue(self, output_type: Optional[str] = None, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """
        Dequeue an output from the queue with optional type filtering.

        Args:
            output_type: Optional filter for output type (TEXT, ACT)
            timeout: BRPOP timeout in seconds (0 = block indefinitely)

        Returns:
            Dict containing the output data, or None if timeout
        """
        while True:
            result = self.store.brpop(self.queue_name, timeout)

            if not result:
                return None

            _, output_id = result

            # Handle both bytes and str
            if isinstance(output_id, bytes):
                output_id = output_id.decode()

            output_data = self.store.get(f"output:{output_id}")

            if not output_data:
                logger.warning(f"Output {output_id} expired or deleted, skipping")
                continue

            output = json.loads(output_data)

            # Filter by type if specified
            if output_type and output.get('type') != output_type:
                logger.debug(f"Re-queuing output {output_id} (type={output.get('type')}, want={output_type})")
                self.store.lpush(self.queue_name, output_id)
                self.store.expire(self.queue_name, 3600)
                continue

            logger.info(f"Dequeued {output.get('type')} output {output_id}")
            return output

    def delete_output(self, output_id: str) -> None:
        """
        Delete an output from storage.

        Args:
            output_id: UUID of the output to delete
        """
        deleted = self.store.delete(f"output:{output_id}")

        if deleted:
            logger.info(f"Deleted output {output_id}")
        else:
            logger.warning(f"Output {output_id} not found for deletion")

    def register_consumer_heartbeat(self, consumer_type: str) -> None:
        """
        Update consumer heartbeat with 60-second TTL.

        Args:
            consumer_type: Type of consumer (e.g., "text", "act")
        """
        heartbeat_key = f"consumer:{consumer_type}:heartbeat"
        timestamp = utc_now().isoformat()

        self.store.setex(heartbeat_key, 60, timestamp)
        logger.debug(f"Updated heartbeat for consumer:{consumer_type}")
