import logging
from typing import Dict, Any, List

from .act_dispatcher_service import ActDispatcherService
from .output_service import OutputService
from .orchestrator.path_schemas import ORCHESTRATOR_PATHS
from .orchestrator.handlers import (
    RespondHandler,
    ActHandler,
    ClarifyHandler,
    AcknowledgeHandler,
    IgnoreHandler,
)


logger = logging.getLogger(__name__)


class OrchestratorService:
    """
    Orchestrator service for path-based routing and execution.

    Provides formal contracts for termination paths with validation,
    required fields, and execution routing.
    """

    def __init__(self):
        """Initialize orchestrator with path definitions and handlers."""
        # Initialize services
        self.act_dispatcher_service = ActDispatcherService()
        self.output_service = OutputService()

        # Initialize handlers
        self.handlers = {
            'RESPOND': RespondHandler(self.output_service),
            'ACT': ActHandler(self.act_dispatcher_service),
            'CLARIFY': ClarifyHandler(self.output_service),
            'ACKNOWLEDGE': AcknowledgeHandler(self.output_service),
            'IGNORE': IgnoreHandler(),
        }

        # Link handlers to path definitions
        for path_name, path_def in ORCHESTRATOR_PATHS.items():
            if path_name in self.handlers:
                path_def.handler = self.handlers[path_name]

        logger.info("OrchestratorService initialized with path-based routing")

    def get_available_paths(self) -> List[Dict[str, Any]]:
        """
        Get available paths for FrontalCortex.

        Returns:
            list: Path definitions with name, type, description
        """
        return [
            {
                'name': path_def.name,
                'type': path_def.path_type.value,
                'required_fields': path_def.required_fields,
                'description': path_def.description
            }
            for path_def in ORCHESTRATOR_PATHS.values()
        ]

    def route_path(self, mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and execute a path.

        Args:
            mode: Path name (e.g., "RESPOND", "ACT")
            context: Execution context with required fields

        Returns:
            dict: {status: "success", mode: str, result: dict} or
                  {status: "error", message: str}
        """
        # Check if path exists
        if mode not in ORCHESTRATOR_PATHS:
            logger.error(f"[ORCHESTRATOR] Unknown path: {mode}")
            return {
                'status': 'error',
                'message': f"Unknown path: {mode}"
            }

        path_def = ORCHESTRATOR_PATHS[mode]

        # Validate context
        is_valid, error_message = path_def.validate(context)
        if not is_valid:
            logger.error(f"[ORCHESTRATOR] Validation failed for {mode}: {error_message}")
            return {
                'status': 'error',
                'message': f"Validation failed: {error_message}"
            }

        # Get handler
        handler = path_def.handler
        if not handler:
            logger.error(f"[ORCHESTRATOR] No handler for path: {mode}")
            return {
                'status': 'error',
                'message': f"No handler configured for path: {mode}"
            }

        # Execute handler
        try:
            result = handler.execute(context)
            logger.info(f"[ORCHESTRATOR] Successfully executed path: {mode}")

            return {
                'status': 'success',
                'mode': mode,
                'result': result
            }

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Execution failed for {mode}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Execution failed: {str(e)}"
            }
