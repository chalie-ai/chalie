import logging
from typing import Dict, Any

from services.act_queue_schema import ActQueueMessage, OutputType
from services.output_service import OutputService


logger = logging.getLogger(__name__)


class RespondHandler:
    """Handler for RESPOND path - queues response to output-queue."""

    def __init__(self, output_service):
        """
        Initialize handler.

        Args:
            output_service: OutputService instance
        """
        self.output_service = output_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute RESPOND path.

        Args:
            context: Must contain response, topic, destination, metadata

        Returns:
            dict: {status: "success", queued: True, output_id: str}
        """
        # Enqueue text output (SSE delivery for web; notification tools for proactive drift)
        output_id = self.output_service.enqueue_text(
            topic=context['topic'],
            response=context['response'],
            mode='RESPOND',
            confidence=context.get('confidence', 0.0),
            generation_time=context.get('generation_time', 0.0),
            original_metadata=context.get('metadata'),
            removes=context.get('removes')
        )

        logger.info(f"[RESPOND] Queued response {output_id} for topic '{context['topic']}' to output-queue")

        return {'status': 'success', 'queued': True, 'output_id': output_id}


class ActHandler:
    """Handler for ACT path - executes actions via ActDispatcherService."""

    def __init__(self, act_dispatcher_service):
        """
        Initialize handler.

        Args:
            act_dispatcher_service: ActDispatcherService instance
        """
        self.act_dispatcher = act_dispatcher_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ACT path.

        Args:
            context: Must contain actions, topic

        Returns:
            dict: {status: "success", results: list}
        """
        topic = context['topic']
        actions = context['actions']

        logger.info(f"[ACT] Executing {len(actions)} actions for topic '{topic}'")

        # Execute actions via ActDispatcherService
        results = []
        for action in actions:
            result = self.act_dispatcher.dispatch_action(topic, action)
            results.append(result)

        return {'status': 'success', 'results': results}


class ClarifyHandler:
    """Handler for CLARIFY path - queues clarification question to output-queue."""

    def __init__(self, output_service):
        """
        Initialize handler.

        Args:
            output_service: OutputService instance
        """
        self.output_service = output_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute CLARIFY path.

        Args:
            context: Must contain clarification_question, topic, destination, metadata

        Returns:
            dict: {status: "success", queued: True, output_id: str}
        """
        # Enqueue text output (SSE delivery for web; notification tools for proactive drift)
        output_id = self.output_service.enqueue_text(
            topic=context['topic'],
            response=context['clarification_question'],
            mode='CLARIFY',
            confidence=context.get('confidence', 0.0),
            generation_time=context.get('generation_time', 0.0),
            original_metadata=context.get('metadata'),
            removes=context.get('removes')
        )

        logger.info(f"[CLARIFY] Queued clarification {output_id} for topic '{context['topic']}' to output-queue")

        return {'status': 'success', 'queued': True, 'output_id': output_id}


class AcknowledgeHandler:
    """Handler for ACKNOWLEDGE path - queues acknowledgment to output-queue."""

    def __init__(self, output_service):
        """
        Initialize handler.

        Args:
            output_service: OutputService instance
        """
        self.output_service = output_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ACKNOWLEDGE path.

        Args:
            context: Must contain topic, destination, metadata

        Returns:
            dict: {status: "success", queued: True, output_id: str, temp_id: str}
        """
        import uuid
        from services.redis_client import RedisClientService

        # Acknowledgment message can be empty or minimal
        content = context.get('response', '')

        # Generate temporary ID for placeholder tracking
        temp_id = str(uuid.uuid4())

        # Store mapping for later retrieval when final response arrives
        # Extract cycle_id from metadata if available
        metadata = context.get('metadata', {})
        cycle_id = metadata.get('cycle_id') or metadata.get('root_cycle_id')

        if cycle_id:
            try:
                redis = RedisClientService.create_connection()
                # Store with 1-hour TTL (matches output TTL)
                redis.setex(f"temp_ack:{cycle_id}", 3600, temp_id)
                logger.debug(f"[ACKNOWLEDGE] Stored temp_id {temp_id} for cycle {cycle_id}")
            except Exception as e:
                logger.warning(f"[ACKNOWLEDGE] Failed to store temp_id mapping: {e}")

        # Enqueue text output (SSE delivery for web; notification tools for proactive drift)
        output_id = self.output_service.enqueue_text(
            topic=context['topic'],
            response=content,
            mode='ACKNOWLEDGE',
            confidence=context.get('confidence', 0.0),
            generation_time=context.get('generation_time', 0.0),
            original_metadata=context.get('metadata'),
            removed_by=temp_id
        )

        logger.info(f"[ACKNOWLEDGE] Queued acknowledgment {output_id} for topic '{context['topic']}' (temp_id={temp_id})")

        return {'status': 'success', 'queued': True, 'output_id': output_id, 'temp_id': temp_id}


class IgnoreHandler:
    """Handler for IGNORE path - logs and does nothing."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute IGNORE path.

        Args:
            context: Must contain topic

        Returns:
            dict: {status: "success", ignored: True}
        """
        topic = context['topic']

        logger.info(f"[IGNORE] Ignoring input for topic '{topic}'")

        return {'status': 'success', 'ignored': True}


class ToolSpawnHandler:
    """Handler for TOOL_SPAWN path - sends acknowledgment and spawns background tool work."""

    def __init__(self, output_service):
        self.output_service = output_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TOOL_SPAWN path.

        1. Queue the template acknowledgment for immediate delivery
        2. (Tool work enqueue is handled by the caller in digest_worker)

        Args:
            context: Must contain response (ack text), topic, destination, metadata

        Returns:
            dict: {status: "success", queued: True, output_id: str, temp_id: str}
        """
        import uuid
        from services.redis_client import RedisClientService

        # Generate temporary ID for placeholder tracking
        temp_id = str(uuid.uuid4())

        # Store mapping for later retrieval when final response arrives
        # Extract cycle_id from metadata if available
        metadata = context.get('metadata', {})
        cycle_id = metadata.get('cycle_id') or metadata.get('root_cycle_id')

        if cycle_id:
            try:
                redis = RedisClientService.create_connection()
                # Store with 1-hour TTL (matches output TTL)
                redis.setex(f"temp_ack:{cycle_id}", 3600, temp_id)
                logger.debug(f"[TOOL_SPAWN] Stored temp_id {temp_id} for cycle {cycle_id}")
            except Exception as e:
                logger.warning(f"[TOOL_SPAWN] Failed to store temp_id mapping: {e}")

        output_id = self.output_service.enqueue_text(
            topic=context['topic'],
            response=context['response'],
            mode='TOOL_SPAWN',
            confidence=context.get('confidence', 0.5),
            generation_time=context.get('generation_time', 0.0),
            original_metadata=context.get('metadata'),
            removed_by=temp_id
        )

        logger.info(
            f"[TOOL_SPAWN] Queued ack '{context['response'][:60]}...' "
            f"for topic '{context['topic']}' (temp_id={temp_id})"
        )

        return {'status': 'success', 'queued': True, 'output_id': output_id, 'temp_id': temp_id}
