"""
The provider registry — every platform Chalie speaks, in one list.

This module is the single place a provider is declared. A client class states
its own identity (``PLATFORM``, ``LABEL``, ``DEFAULT_BASE_URL``,
``REQUIRES_KEY``, ``REQUIRES_HOST``); this file states which classes exist and
in what order they are offered. Everything downstream is derived:

- ``services.llm_clients.factory`` dispatches a config to a class through it.
- ``services.provider_catalog_service`` builds the setup catalog from it.

Adding a provider is therefore one new module plus one line here — never an
edit to a factory branch, a catalog literal, and a validation tuple that each
have to be remembered separately.

Consumed by: services.llm_clients.factory, services.provider_catalog_service.
"""

from __future__ import annotations

from contracts.provider_client import ProviderClient
from services.llm_clients.alibaba import AlibabaClient
from services.llm_clients.anthropic import AnthropicClient
from services.llm_clients.baseten import BasetenClient
from services.llm_clients.codex_cli import CodexCliClient
from services.llm_clients.deepseek import DeepSeekClient
from services.llm_clients.fireworks import FireworksClient
from services.llm_clients.gemini import GeminiClient
from services.llm_clients.groq import GroqClient
from services.llm_clients.llama_cpp import LlamaCppClient
from services.llm_clients.minimax import MiniMaxClient
from services.llm_clients.mistral import MistralClient
from services.llm_clients.moonshot import MoonshotClient
from services.llm_clients.novita import NovitaClient
from services.llm_clients.nvidia import NvidiaClient
from services.llm_clients.ollama import OllamaClient
from services.llm_clients.openai import OpenAIClient
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.openrouter import OpenRouterClient
from services.llm_clients.vllm import VllmClient
from services.llm_clients.xai import XaiClient
from services.llm_clients.zhipu import ZhipuClient

#: Every platform, in the order they are offered during setup. This tuple IS
#: the order of the tiles a new user picks from, so it is ordered by the
#: quality of the first-day experience rather than by how little signup a
#: platform needs: the frontier vendors lead, then the OpenAI-protocol hosts
#: alphabetically, then the local servers, then the escape hatch last.
#:
#: Local servers come after the hosted vendors deliberately. A small local
#: model is the cheapest provider to start and the most likely to make Chalie
#: look incapable on day one, and a first impression is not recoverable — so
#: it stays available and stops being the default path.
PROVIDER_CLASSES: tuple[type[ProviderClient], ...] = (
    AnthropicClient,
    GeminiClient,
    OpenAIClient,
    CodexCliClient,
    AlibabaClient,
    BasetenClient,
    DeepSeekClient,
    FireworksClient,
    GroqClient,
    MiniMaxClient,
    MistralClient,
    MoonshotClient,
    NovitaClient,
    NvidiaClient,
    OpenRouterClient,
    XaiClient,
    ZhipuClient,
    OllamaClient,
    LlamaCppClient,
    VllmClient,
    OpenAICompatibleClient,
)


def _build_index() -> dict[str, type[ProviderClient]]:
    """Map every declared platform to its class, rejecting duplicates loudly.

    A repeated ``PLATFORM`` is a copy-paste in a new provider module, and it
    would otherwise resolve silently to whichever class happened to be listed
    last — sending a user's traffic to the wrong vendor's client. Cheap to
    catch here, at import, where the traceback names the offender.
    """
    index: dict[str, type[ProviderClient]] = {}
    for client in PROVIDER_CLASSES:
        platform = client.PLATFORM
        if platform in index:
            raise RuntimeError(
                f"Duplicate provider platform {platform!r}: "
                f"{index[platform].__name__} and {client.__name__} both claim it"
            )
        index[platform] = client
    return index


PROVIDERS_BY_PLATFORM: dict[str, type[ProviderClient]] = _build_index()


class Registry:
    """Lookup utilities for the provider registry."""

    @classmethod
    def client_class_for(cls, platform: str) -> type[ProviderClient] | None:
        """The client class serving *platform*, or None when nothing claims it."""
        return PROVIDERS_BY_PLATFORM.get(platform)

    @classmethod
    def platform_requires_key(cls, platform: str) -> bool:
        """Whether *platform* cannot be reached at all without a credential.

        Read off the client class rather than listed anywhere: a hand-kept tuple
        omits every provider added after it was written, and the omission is silent
        in both directions — a hosted vendor left out gets a guaranteed-to-fail call
        recorded as a real failure, and a self-hosted server wrongly included is
        refused before it is ever contacted.

        False for a platform nothing claims. An unknown platform is rejected at the
        factory with a message naming it, which is a better answer than a complaint
        about a missing key for a provider that does not exist.
        """
        client = cls.client_class_for(platform)
        return client is not None and client.REQUIRES_KEY
