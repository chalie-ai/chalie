"""
ProviderClient — abstract base class for all thin provider clients.

Every platform (Anthropic, OpenAI, Gemini, Ollama) implements this ABC.
The contract is intentionally narrow: transform DTO → native, call, parse.
Telemetry, over-cap checking, and resolution live in Providers (the facade).

Consumed by: services.providers (resolution + orchestration).
Implementations: services.llm_clients.{anthropic,openai,gemini,ollama}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from services.provider_api import ProviderApiRequest, ProviderApiResponse


class ProviderClient(ABC):
    """Abstract thin client — one concrete subclass per provider platform.

    Responsibilities (SRP):
      - Transform a ProviderApiRequest into the platform's native wire format.
      - Execute the HTTP call.
      - Parse the native response into a ProviderApiResponse.
      - Map platform-specific errors to the typed exception hierarchy.

    Every public method that callers may call is declared here; helpers used
    only inside the concrete subclass are private (_prefixed).
    """

    # JSON path of the user-visible text field in this provider's response.
    # Read by configs/channels/_common.py:15 via Providers.selected_provider()
    # to substitute {{provider_content_field_name}} in system prompts.
    # Each concrete subclass MUST declare this as a ClassVar[str].
    CONTENT_FIELD_LABEL: ClassVar[str]

    @abstractmethod
    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform the DTO to native, call the provider, parse back.

        Raises:
            ResponseOverLimitError: provider rejected the payload for size.
            ProviderResponseError: other API or HTTP error.
            RateLimitError: HTTP 429; carries retry_after when available.
        """

    @abstractmethod
    def get_context_limit(self) -> int:
        """Return the model's input context window (tokens)."""

    @abstractmethod
    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Estimate the token cost of dto without sending.

        Folds in the same serialisation as send() so the estimate and the
        actual wire body use identical encoding. Used by Providers.send() for
        the pre-flight over-cap check and by Providers.measure() for the
        compactor's candidate-fit sizing.
        """
