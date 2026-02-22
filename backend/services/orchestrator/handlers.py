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
            dict: {status: "success", queued: True, output_id: str}
        """
        content = context.get('response', '')

        # Enqueue text output (SSE delivery for web; notification tools for proactive drift)
        output_id = self.output_service.enqueue_text(
            topic=context['topic'],
            response=content,
            mode='ACKNOWLEDGE',
            confidence=context.get('confidence', 0.0),
            generation_time=context.get('generation_time', 0.0),
            original_metadata=context.get('metadata'),
        )

        logger.info(f"[ACKNOWLEDGE] Queued acknowledgment {output_id} for topic '{context['topic']}'")

        return {'status': 'success', 'queued': True, 'output_id': output_id}


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


