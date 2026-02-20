import logging
import time

from services.output_service import OutputService


logger = logging.getLogger(__name__)


class ActionDigestListener:
    """Listener for ACT outputs (stub implementation for Phase 2)."""

    def __init__(self):
        """Initialize the ActionDigestListener."""
        self.output_service = OutputService()

        logger.info("ActionDigestListener initialized (stub)")

    def listen(self) -> None:
        """
        Start blocking loop to consume ACT outputs from the output queue.

        Stub implementation: logs ACT output reception and deletes it.
        Actual action processing will be implemented in future phases.
        """
        logger.info("ActionDigestListener started, waiting for ACT outputs...")

        while True:
            try:
                # Dequeue ACT output with 30-second timeout
                output = self.output_service.dequeue(output_type="ACT", timeout=30)

                if output:
                    self._process_output(output)

                # Update heartbeat
                self.output_service.register_consumer_heartbeat("act")

            except KeyboardInterrupt:
                logger.info("ActionDigestListener shutting down...")
                break
            except Exception as e:
                logger.error(f"[ActionDigestListener] Error: {e}", exc_info=True)
                time.sleep(5)

    def _process_output(self, output: dict) -> None:
        """
        Process an ACT output (stub implementation).

        Args:
            output: Output object containing actions and metadata
        """
        try:
            output_id = output.get('id')
            topic = output.get('topic')
            actions = output.get('metadata', {}).get('actions', [])
            loop_id = output.get('metadata', {}).get('loop_id')

            logger.info(
                f"[ActionDigestListener] Received ACT output {output_id} "
                f"for topic '{topic}' (loop_id={loop_id}, actions={len(actions)})"
            )

            # Stub: Just log and delete (no processing yet)
            logger.info(f"[ActionDigestListener] Actions: {actions}")

            # Delete from queue
            self.output_service.delete_output(output_id)

        except Exception as e:
            logger.error(f"[ActionDigestListener] Failed to process output {output.get('id')}: {e}", exc_info=True)
