from enum import Enum
from typing import Callable, List, Dict, Any


class PathType(Enum):
    """Types of execution paths."""
    TERMINAL = "terminal"  # Ends conversation flow (RESPOND, CLARIFY, etc.)
    TACTICAL = "tactical"  # Continues processing (ACT)


class PathDefinition:
    """Definition of an orchestrator path with validation and routing."""

    def __init__(
        self,
        name: str,
        path_type: PathType,
        required_fields: List[str],
        validator: Callable[[Dict[str, Any]], bool],
        handler: Any,  # Handler class (not instance)
        description: str
    ):
        """
        Initialize a path definition.

        Args:
            name: Path name (e.g., "RESPOND", "ACT")
            path_type: Terminal or Tactical
            required_fields: List of required context fields
            validator: Lambda to validate context beyond required fields
            handler: Handler class to execute this path
            description: Human-readable description
        """
        self.name = name
        self.path_type = path_type
        self.required_fields = required_fields
        self.validator = validator
        self.handler = handler
        self.description = description

    def validate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate context for this path.

        Args:
            context: Execution context dict

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check required fields
        missing_fields = [f for f in self.required_fields if f not in context]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        # Run custom validator
        try:
            if not self.validator(context):
                return False, "Custom validation failed"
        except Exception as e:
            return False, f"Validator error: {e}"

        return True, ""


# Validators for each path
def validate_respond(ctx: Dict[str, Any]) -> bool:
    """Validate RESPOND path context."""
    return bool(ctx.get('response')) and 0 <= ctx.get('confidence', 0) <= 1


def validate_act(ctx: Dict[str, Any]) -> bool:
    """Validate ACT path context."""
    return isinstance(ctx.get('actions'), list) and len(ctx.get('actions', [])) > 0


def validate_clarify(ctx: Dict[str, Any]) -> bool:
    """Validate CLARIFY path context."""
    return bool(ctx.get('clarification_question'))


def validate_acknowledge(ctx: Dict[str, Any]) -> bool:
    """Validate ACKNOWLEDGE path context."""
    return True  # Only requires topic and destination (checked in required_fields)


def validate_ignore(ctx: Dict[str, Any]) -> bool:
    """Validate IGNORE path context."""
    return True  # Only requires topic


# Orchestrator paths registry
# Handlers will be set during OrchestratorService initialization
ORCHESTRATOR_PATHS: Dict[str, PathDefinition] = {
    "RESPOND": PathDefinition(
        name="RESPOND",
        path_type=PathType.TERMINAL,
        required_fields=["response", "confidence", "topic", "destination"],
        validator=validate_respond,
        handler=None,  # Will be set to RespondHandler
        description="Terminal path: Queue response for delivery to user"
    ),

    "ACT": PathDefinition(
        name="ACT",
        path_type=PathType.TACTICAL,
        required_fields=["actions", "topic"],
        validator=validate_act,
        handler=None,  # Will be set to ActHandler
        description="Tactical path: Execute actions and continue processing"
    ),

    "CLARIFY": PathDefinition(
        name="CLARIFY",
        path_type=PathType.TERMINAL,
        required_fields=["clarification_question", "topic", "destination"],
        validator=validate_clarify,
        handler=None,  # Will be set to ClarifyHandler
        description="Terminal path: Request clarification from user"
    ),

    "ACKNOWLEDGE": PathDefinition(
        name="ACKNOWLEDGE",
        path_type=PathType.TERMINAL,
        required_fields=["topic", "destination"],
        validator=validate_acknowledge,
        handler=None,  # Will be set to AcknowledgeHandler
        description="Terminal path: Acknowledge user input without further action"
    ),

    "IGNORE": PathDefinition(
        name="IGNORE",
        path_type=PathType.TERMINAL,
        required_fields=["topic"],
        validator=validate_ignore,
        handler=None,  # Will be set to IgnoreHandler
        description="Terminal path: Log and ignore input"
    ),
}
