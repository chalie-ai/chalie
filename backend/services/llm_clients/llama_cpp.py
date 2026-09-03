"""
llama.cpp (``llama-server``) — a self-hosted OpenAI-protocol server.

The only provider here that needs two endpoints, because which one answers
depends on the server's build:

- ``/v1/models`` → ``data[].meta.n_ctx`` is the **served** window (the runtime
  per-slot size). Always public: ``/models`` and ``/v1/models`` sit on the
  server's public-endpoint exemption list and answer even when ``--api-key`` is
  configured. Tried first.
- ``/props`` → ``default_generation_settings.n_ctx``, the same served figure.
  Reached only as a fallback: it lives at the **server root**, not under
  ``/v1``, and it is *not* on the exemption list, so a keyed server refuses it.

The fallback is not redundancy. ``meta.n_ctx`` is a relatively recent addition
to ``get_model_info`` — the published README example still shows an entry
carrying only ``n_ctx_train`` — so an older server answers ``/v1/models``
without ever naming its served window, and ``/props`` is the only place left
that does.

``meta.n_ctx_train`` is never read, here or anywhere. It is the checkpoint's
*training* length, and llama.cpp explicitly caps the served size down to it
when ``--ctx-size`` asks for more — so the two legitimately differ, and taking
the training figure over-promises a window the server will not honour.

https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

from __future__ import annotations

import logging
from typing import ClassVar, cast
from urllib.parse import urlsplit, urlunsplit

from services.llm_clients.openai_compatible import OpenAICompatibleClient, _positive_int

logger = logging.getLogger(__name__)


class LlamaCppClient(OpenAICompatibleClient):
    PLATFORM: ClassVar[str] = 'llama_cpp'
    LABEL: ClassVar[str] = 'llama.cpp (self-hosted)'
    DEFAULT_BASE_URL: ClassVar[str] = 'http://localhost:8080/v1'
    REQUIRES_KEY: ClassVar[bool] = False

    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ('meta.n_ctx',)

    def _probe_context_window(self) -> int | None:
        """The listing's served window, then ``/props`` for servers too old to publish it."""
        window = self._window_from_models_endpoint()
        if window is None:
            window = self._window_from_props()
        return window if window is not None else self._probe_via_ping()

    def _window_from_props(self) -> int | None:
        """The served window from ``/props``, or None when it cannot be read.

        Never raises: a server that is down, keyed, or too old to serve this
        route simply leaves the caller to fall through to a ping.
        """
        url = self._props_url()
        if url is None:
            return None

        import requests  # noqa: PLC0415
        from services.provider_api import PROVIDER_CALL_TIMEOUT_S  # noqa: PLC0415

        headers = {}
        api_key = self._config.get('api_key')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        try:
            resp = requests.get(url, headers=headers, timeout=PROVIDER_CALL_TIMEOUT_S)
            resp.raise_for_status()
            payload = cast(dict[str, object], resp.json())
        except Exception as exc:
            logger.warning(
                "[LlamaCppClient] /props unavailable for window detection on model=%s: %s",
                self.model, exc,
            )
            return None

        settings = payload.get('default_generation_settings')
        if not isinstance(settings, dict):
            return None
        return _positive_int(cast(dict[str, object], settings).get('n_ctx'))

    def _props_url(self) -> str | None:
        """``/props`` on the configured host — at the server root, below ``/v1``.

        The provider row stores the OpenAI base URL, which for llama.cpp is the
        server root plus ``/v1``; that segment is dropped here because ``/props``
        is registered without it.
        """
        host = cast(str, self._config.get('host') or '')
        if not host:
            return None
        parts = urlsplit(host)
        path = parts.path.rstrip('/')
        if path.endswith('/v1'):
            path = path[: -len('/v1')]
        return urlunsplit((parts.scheme, parts.netloc, f'{path}/props', '', ''))
