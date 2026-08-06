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

    # ── Identity ────────────────────────────────────────────────────
    # What this platform is and what creating one requires. Declared on the
    # class rather than listed anywhere else so a new provider is a new module
    # and nothing more: the registry reads these to dispatch, to validate a
    # config, and to build the setup catalog, and none of those can drift from
    # the client they describe.

    #: Stable id stored in ``providers.platform``. Unique, and never reused —
    #: rows in existing databases are matched by this string.
    PLATFORM: ClassVar[str]

    #: Vendor name as shown during provider setup.
    LABEL: ClassVar[str]

    #: Base URL pre-filled when a user picks this platform. Empty when the SDK
    #: already knows the endpoint, or when only the user can know the host.
    DEFAULT_BASE_URL: ClassVar[str]

    #: Whether a config for this platform must carry an API key / a host.
    REQUIRES_KEY: ClassVar[bool]
    REQUIRES_HOST: ClassVar[bool]

    def __init__(self, config: dict[str, object]) -> None:
        """Build a client from a provider row (platform, model, host, api_key).

        Part of the contract because the factory constructs clients generically
        off the registry — it holds a class, not a name, so every platform must
        take the same one argument or the dispatch cannot be uniform.
        """
        ...

    @classmethod
    def fetch_models(
        cls, host: str, api_key: str,
    ) -> tuple[list[dict[str, str | None]] | None, str | None]:
        """List the models this platform offers, as ``(models, error)``.

        A classmethod because the setup wizard calls it before a provider row
        exists — there are credentials to try, but no configured provider yet.

        Returns an error string rather than raising: an unreachable host or a
        rejected key is an answer the wizard shows the user, not a fault.
        """
        ...

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform the DTO to native, call the provider, parse back.

        Raises:
            ContextLimit: provider rejected the payload for size.
            ProviderResponseError: other API or HTTP error.
            RateLimitError: HTTP 429.
            ProviderTimeoutError: the call exceeded PROVIDER_CALL_TIMEOUT_S.
        """
        ...

    def get_context_limit(self) -> int | None:
        """Return the model's real input context window (tokens), or None when
        the value cannot be determined.

        Implementations MUST cache the result (including None) so a second call
        does not re-probe.
        """
        ...
