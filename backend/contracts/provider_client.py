"""ProviderClient — the thin-client contract every provider platform satisfies.

Every platform (Anthropic, OpenAI, Gemini, Ollama) conforms to this Protocol.
The contract is intentionally narrow: transform DTO → native, call, parse.
Telemetry, over-cap checking, and resolution live in ProviderService.

Lives in ``contracts`` because it is a pure interface: it declares the shape a
provider client must satisfy, it stores no data and provides no implementation.
Concrete clients conform explicitly (``class AnthropicClient(ProviderClient)``),
mirroring PHP ``implements`` — the explicit link keeps the conformance
statically checked by mypy without coupling callers to any one platform.

Consumed by: services.provider_service (resolution + orchestration).
Implementations: services.llm_clients.{anthropic,openai,gemini,ollama}.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from services.provider_api import ProviderApiRequest, ProviderApiResponse


class ProviderClient(Protocol):
    """Thin client — one concrete implementation per provider platform.

    Responsibilities (SRP):
      - Transform a ProviderApiRequest into the platform's native wire format.
      - Execute the HTTP call.
      - Parse the native response into a ProviderApiResponse.
      - Map platform-specific errors to the typed exception hierarchy.

    Every public method that callers may call is declared here; helpers used
    only inside the concrete implementation are private (_prefixed).
    """

    # JSON path of the user-visible text field in this provider's response.
    # Read by configs/channels/_common.py:15 via ProviderService.selected_provider()
    # to substitute {{provider_content_field_name}} in system prompts.
    # Each concrete implementation MUST declare this as a ClassVar[str].
    CONTENT_FIELD_LABEL: ClassVar[str]

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform the DTO to native, call the provider, parse back.

        Raises:
            ResponseOverLimitError: provider rejected the payload for size.
            ProviderResponseError: other API or HTTP error.
            RateLimitError: HTTP 429.
            ProviderTimeoutError: the call exceeded PROVIDER_CALL_TIMEOUT_S.
        """
        ...

    def get_context_limit(self) -> int:
        """Return the model's input context window (tokens)."""
        ...

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Estimate the token cost of dto without sending.

        Folds in the same serialisation as send() so the estimate and the
        actual wire body use identical encoding. Used by ProviderService.send() for
        the pre-flight over-cap check and by ProviderService.measure() for the
        compactor's candidate-fit sizing.
        """
        ...
