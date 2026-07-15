# Feature tests for ThinkingLevel.NONE across all thin clients.
# Real-stack — no mocks of production code.

import json
import stat
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, cast

import google.genai as _genai_mod
import pytest

from configs.channels.thread_gist import ThreadGistConfig
from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel

if TYPE_CHECKING:
    from services.llm_clients.gemini import GeminiClient, _GenaiClient, _GenCfg, _Genai
    from services.provider_api import ProviderApiResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ScriptedHandler(BaseHTTPRequestHandler):
    """HTTP handler that records requests and pops scripted responses."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default logging to keep test output clean."""
        pass

    def do_POST(self) -> None:
        server = cast("_ScriptedServer", self.server)

        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length) if content_length else b""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        server.recorded.append(parsed)

        status, resp_body = server.responses.popleft()
        body_bytes = json.dumps(resp_body).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)


class _ScriptedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the recorded-request list and scripted responses."""

    def __init__(self, recorded: "list[dict[str, object]]",
                 responses: "deque[tuple[int, dict[str, object]]]") -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)
        self.recorded = recorded
        self.responses = responses


def _openai_error(message: str) -> tuple[int, dict[str, object]]:
    """Return a scripted (status, body) 400 rejection in OpenAI's error shape."""
    return (400, {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        },
    })


_OPENAI_SUCCESS_BODY: dict[str, object] = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


# A stub that impersonates `codex`. Echoes argv as the agent message text,
# writes the same text to the -o outfile, and emits minimal JSONL events.
_STUB_CODEX_ECHO = """#!/usr/bin/env python3
import sys, os, json

args = sys.argv[1:]
if args and args[0] == "--version":
    print("codex-cli 0.143.0-stub")
    sys.exit(0)

outfile = None
for i, a in enumerate(args):
    if a == "-o":
        outfile = args[i + 1]

prompt = sys.stdin.read()
text = " ".join(sys.argv[1:])

if outfile:
    with open(outfile, "w") as f:
        f.write(text)

print(json.dumps({"type": "thread.started", "thread_id": "t1"}))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "item.completed",
                  "item": {"id": "item_0", "type": "agent_message", "text": text}}))
print(json.dumps({"type": "turn.completed",
                  "usage": {"input_tokens": 0, "cached_input_tokens": 0,
                            "output_tokens": 0, "reasoning_output_tokens": 0}}))
sys.exit(0)
"""


_MODELS_CACHE: dict[str, object] = {
    "fetched_at": "2026-07-09T00:00:00Z",
    "models": [
        {"slug": "gpt-5.5", "display_name": "GPT-5.5", "context_window": 272000},
    ],
}


def _install_stub_codex(tmp_path: Path) -> str:
    """Write the stub codex binary and return its path (executable)."""
    binary = tmp_path / "codex-stub"
    binary.write_text(_STUB_CODEX_ECHO)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(binary)


def _install_codex_home(tmp_path: Path) -> Path:
    """Create a fake ~/.codex with a models cache."""
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "models_cache.json").write_text(json.dumps(_MODELS_CACHE))
    return home


# Sentinel used by the Gemini fake-SDK tests.
_SENTINEL = object()


# ---------------------------------------------------------------------------
# 1. Anthropic native thinking flag mapping
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicThinkingNative:

    def test_all_five_levels_map_correctly(self) -> None:
        from services.llm_clients.anthropic import AnthropicClient

        client = AnthropicClient({"platform": "anthropic", "model": "claude-sonnet-4-20250514"})

        none_result = client._thinking_native(ThinkingLevel.NONE, max_tokens=9000)
        assert none_result == {"thinking": {"type": "disabled"}}

        low_result = client._thinking_native(ThinkingLevel.LOW, max_tokens=9000)
        assert low_result == {}

        med_result = client._thinking_native(ThinkingLevel.MEDIUM, max_tokens=9000)
        assert med_result == {"thinking": {"type": "enabled", "budget_tokens": 4096}}

        high_result = client._thinking_native(ThinkingLevel.HIGH, max_tokens=9000)
        assert high_result == {"thinking": {"type": "enabled", "budget_tokens": 16384}}

        max_result = client._thinking_native(ThinkingLevel.MAX, max_tokens=9000)
        assert max_result == {"thinking": {"type": "enabled", "budget_tokens": 9000}}


# ---------------------------------------------------------------------------
# 2. OpenAI native thinking flag mapping
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAIThinkingNative:

    def test_all_five_levels_map_correctly(self) -> None:
        from services.llm_clients.openai import OpenAIClient

        client = OpenAIClient({"platform": "openai", "model": "gpt-4o", "api_key": "k"})

        assert client._thinking_native(ThinkingLevel.NONE) == "none"
        assert client._thinking_native(ThinkingLevel.LOW) is None
        assert client._thinking_native(ThinkingLevel.MEDIUM) == "medium"
        assert client._thinking_native(ThinkingLevel.HIGH) == "high"
        assert client._thinking_native(ThinkingLevel.MAX) == "high"


# ---------------------------------------------------------------------------
# 3. OpenAI ladder retry over real HTTP
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAILadderRealHttp:
    """Drive OpenAIClient.send() against a scripted local HTTP server.

    The real openai SDK issues real HTTP requests; assertions run against
    the recorded wire bodies, proving extra_body actually merges into the
    request JSON. The SDK does not retry 400s, so scripted rejections
    drive the ladder deterministically.
    """

    def _send_scripted(
        self,
        platform: str,
        level: "ThinkingLevel",
        responses: "deque[tuple[int, dict[str, object]]]",
    ) -> "tuple[list[dict[str, object]], ProviderApiResponse]":
        from services.llm_clients.openai import OpenAIClient
        from services.provider_api import ProviderApiRequest

        recorded: list[dict[str, object]] = []
        server = _ScriptedServer(recorded, responses)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            client = OpenAIClient({
                "platform": platform,
                "model": "m",
                "api_key": "test-key",
                "host": f"http://127.0.0.1:{port}/v1",
            })
            dto = ProviderApiRequest(
                system="s",
                messages=[{"role": "user", "content": "hi"}],
                type=ProviderType.CHAT,
                thinking_mode=level,
                cache_prefix=False,
                max_tokens=32,
            )
            return recorded, client.send(dto)
        finally:
            server.shutdown()
            server.server_close()

    def test_none_sends_effort_and_extra_body(self) -> None:
        responses: "deque[tuple[int, dict[str, object]]]" = deque([(200, _OPENAI_SUCCESS_BODY)])
        recorded, resp = self._send_scripted("openai_compatible", ThinkingLevel.NONE, responses)

        assert len(recorded) == 1
        assert recorded[0].get("reasoning_effort") == "none"
        # extra_body merges into the top level of the wire JSON.
        assert recorded[0].get("thinking") == {"type": "disabled"}
        assert resp.text == "ok"

    def test_full_ladder_none_to_minimal_to_bare(self) -> None:
        responses: "deque[tuple[int, dict[str, object]]]" = deque([
            _openai_error("Extra inputs are not permitted: extra_forbidden thinking"),
            _openai_error("unsupported value for reasoning_effort"),
            (200, _OPENAI_SUCCESS_BODY),
        ])
        recorded, resp = self._send_scripted("openai_compatible", ThinkingLevel.NONE, responses)

        assert len(recorded) == 3
        assert recorded[0].get("reasoning_effort") == "none"
        assert recorded[0].get("thinking") == {"type": "disabled"}
        assert recorded[1].get("reasoning_effort") == "minimal"
        assert "thinking" not in recorded[1]
        assert "reasoning_effort" not in recorded[2]
        assert "thinking" not in recorded[2]
        assert resp.text == "ok"

    def test_platform_openai_never_sends_extra_body(self) -> None:
        responses: "deque[tuple[int, dict[str, object]]]" = deque([(200, _OPENAI_SUCCESS_BODY)])
        recorded, resp = self._send_scripted("openai", ThinkingLevel.NONE, responses)

        assert len(recorded) == 1
        assert recorded[0].get("reasoning_effort") == "none"
        assert "thinking" not in recorded[0]
        assert resp.text == "ok"

    def test_low_sends_no_flags(self) -> None:
        responses: "deque[tuple[int, dict[str, object]]]" = deque([(200, _OPENAI_SUCCESS_BODY)])
        recorded, resp = self._send_scripted("openai_compatible", ThinkingLevel.LOW, responses)

        assert len(recorded) == 1
        assert "reasoning_effort" not in recorded[0]
        assert "thinking" not in recorded[0]
        assert resp.text == "ok"


# ---------------------------------------------------------------------------
# 4. Gemini ladder retry (real genai types, fake SDK client)
# ---------------------------------------------------------------------------

class _FakeGenaiModels:
    """Records config kwargs per call and dispatches to a scripted sequence."""

    def __init__(self, parent: "_FakeGenaiClient") -> None:
        self._parent = parent

    def generate_content(self, **kwargs: object) -> object:
        self._parent.calls.append(kwargs)
        item = next(self._parent.script_iter)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeGenaiClient:
    """Boundary double for google.genai.Client — records calls, scripts responses."""

    def __init__(self, script: "list[object]") -> None:
        self.calls: list[dict[str, object]] = []
        self.script_iter = iter(script)
        self.models = _FakeGenaiModels(self)


@pytest.mark.unit
class TestGeminiLadder:

    def _build_client(self) -> "GeminiClient":
        from services.llm_clients.gemini import GeminiClient
        return GeminiClient({"platform": "gemini", "model": "gemini-2.5-flash", "api_key": "k"})

    def _run(self, client: "GeminiClient", fake: "_FakeGenaiClient", gen_cfg: "_GenCfg") -> object:
        contents: object = [{"role": "user", "parts": [{"text": "hi"}]}]
        return client._generate_with_fallback(
            cast("_GenaiClient", fake), cast("_Genai", _genai_mod), contents, gen_cfg,
        )

    def test_none_budget_zero_then_floor_128(self) -> None:
        client = self._build_client()

        gen_cfg = cast("_GenCfg", {
            "system_instruction": "s",
            "thinking_config": _genai_mod.types.ThinkingConfig(thinking_budget=0),
        })
        fake = _FakeGenaiClient([
            Exception("thinking is not supported"),
            _SENTINEL,
        ])

        result = self._run(client, fake, gen_cfg)

        assert result is _SENTINEL
        assert len(fake.calls) == 2
        retry_thinking = getattr(fake.calls[1]["config"], "thinking_config", None)
        assert retry_thinking is not None
        assert getattr(retry_thinking, "thinking_budget", None) == 128

    def test_none_floor_also_rejected_strips(self) -> None:
        client = self._build_client()

        gen_cfg = cast("_GenCfg", {
            "system_instruction": "s",
            "thinking_config": _genai_mod.types.ThinkingConfig(thinking_budget=0),
        })
        fake = _FakeGenaiClient([
            Exception("thinking not supported"),
            Exception("thinking not supported"),
            _SENTINEL,
        ])

        result = self._run(client, fake, gen_cfg)

        assert result is _SENTINEL
        assert len(fake.calls) == 3
        # After stripping thinking_config, the attribute should be absent/None.
        assert getattr(fake.calls[2]["config"], "thinking_config", None) is None

    def test_medium_rejection_strips_directly(self) -> None:
        client = self._build_client()

        gen_cfg = cast("_GenCfg", {
            "system_instruction": "s",
            "thinking_config": _genai_mod.types.ThinkingConfig(thinking_budget=4096),
        })
        fake = _FakeGenaiClient([
            Exception("unsupported"),
            _SENTINEL,
        ])

        result = self._run(client, fake, gen_cfg)

        assert result is _SENTINEL
        assert len(fake.calls) == 2
        assert getattr(fake.calls[1]["config"], "thinking_config", None) is None

    def test_non_thinking_error_propagates(self) -> None:
        client = self._build_client()

        # gen_cfg WITHOUT thinking_config
        gen_cfg = cast("_GenCfg", {"system_instruction": "s"})
        fake = _FakeGenaiClient([
            Exception("boom"),
        ])

        with pytest.raises(Exception, match="boom"):
            self._run(client, fake, gen_cfg)

        assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# 5. Ollama think payload
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaThinkPayload:

    def test_think_flag_per_level_when_supported(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        client = OllamaClient({"host": "http://127.0.0.1:1", "model": "m"})
        client._thinking_supported = True

        none_payload = client._build_payload("s", [], None, ThinkingLevel.NONE)
        assert none_payload.get("think") is False

        low_payload = client._build_payload("s", [], None, ThinkingLevel.LOW)
        assert "think" not in low_payload

        for level in (ThinkingLevel.MEDIUM, ThinkingLevel.HIGH, ThinkingLevel.MAX):
            payload = client._build_payload("s", [], None, level)
            assert payload.get("think") is True

    def test_think_absent_when_not_supported(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        client = OllamaClient({"host": "http://127.0.0.1:1", "model": "m"})
        client._thinking_supported = False

        for level in (ThinkingLevel.NONE, ThinkingLevel.LOW,
                      ThinkingLevel.MEDIUM, ThinkingLevel.HIGH, ThinkingLevel.MAX):
            payload = client._build_payload("s", [], None, level)
            assert "think" not in payload, f"Unexpected 'think' for level={level}"


# ---------------------------------------------------------------------------
# 6. Codex CLI effort flag via stub binary
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCodexCliEffortFlag:

    def _send(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              level: "ThinkingLevel") -> "ProviderApiResponse":
        from services.llm_clients.codex_cli import CodexCliClient
        from services.provider_api import ProviderApiRequest

        monkeypatch.setenv("CODEX_BIN", _install_stub_codex(tmp_path))
        monkeypatch.setenv("CODEX_HOME", str(_install_codex_home(tmp_path)))

        client = CodexCliClient({"platform": "codex_cli", "model": "gpt-5.5"})
        dto = ProviderApiRequest(
            system="You are helpful.",
            messages=[{"role": "user", "content": "hello"}],
            type=ProviderType.CHAT,
            thinking_mode=level,
            cache_prefix=False,
            max_tokens=64,
        )
        return client.send(dto)

    def test_none_includes_minimal_effort_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, ThinkingLevel.NONE)
        assert "-c model_reasoning_effort=minimal" in resp.text

    def test_low_omits_effort_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, ThinkingLevel.LOW)
        assert "model_reasoning_effort" not in resp.text

    def test_medium_includes_medium_effort_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, ThinkingLevel.MEDIUM)
        assert "=medium" in resp.text

    def test_high_includes_high_effort_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, ThinkingLevel.HIGH)
        assert "=high" in resp.text

    def test_max_includes_high_effort_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, ThinkingLevel.MAX)
        assert "=high" in resp.text


# ---------------------------------------------------------------------------
# 7. ThreadGist pins thinking_mode to 'none'
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestThreadGistPinsNone:

    def test_thread_gist_thinking_mode_is_none(self) -> None:
        assert ThreadGistConfig.thinking_mode == "none"
        assert ThinkingLevel("none") is ThinkingLevel.NONE


# ---------------------------------------------------------------------------
# 8. _is_thinking_rejection recognition
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsThinkingRejection:

    def test_reasoning_effort_rejection_with_kwarg(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("unsupported value for reasoning_effort")
        assert _is_thinking_rejection(exc, {"reasoning_effort": "none"}) is True

    def test_reasoning_effort_error_without_kwarg(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("unsupported value for reasoning_effort")
        assert _is_thinking_rejection(exc, {}) is False

    def test_extra_body_thinking_rejection(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("Extra inputs are not permitted extra_forbidden")
        kwargs: dict[str, object] = {"extra_body": {"thinking": {"type": "disabled"}}}
        assert _is_thinking_rejection(exc, kwargs) is True

    def test_unrelated_error_with_both_kwargs(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("boom")
        kwargs: dict[str, object] = {
            "reasoning_effort": "none",
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        assert _is_thinking_rejection(exc, kwargs) is False

    def test_extra_body_without_thinking_key(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("Extra inputs are not permitted extra_forbidden")
        kwargs: dict[str, object] = {"extra_body": {"something_else": "value"}}
        assert _is_thinking_rejection(exc, kwargs) is False


# ---------------------------------------------------------------------------
# 9. Map integrity — LOW absent, NONE present where expected
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMapIntegrity:

    def test_low_absent_from_all_maps(self) -> None:
        from services.llm_clients.thinking_map import (
            ANTHROPIC_THINKING_BUDGETS,
            CODEX_REASONING_EFFORTS,
            GEMINI_THINKING_BUDGETS,
            OLLAMA_THINK,
            OPENAI_REASONING_EFFORTS,
        )

        assert ThinkingLevel.LOW not in OPENAI_REASONING_EFFORTS
        assert ThinkingLevel.LOW not in GEMINI_THINKING_BUDGETS
        assert ThinkingLevel.LOW not in OLLAMA_THINK
        assert ThinkingLevel.LOW not in CODEX_REASONING_EFFORTS
        assert ThinkingLevel.LOW not in ANTHROPIC_THINKING_BUDGETS

    def test_none_present_in_every_map_except_anthropic_budgets(self) -> None:
        from services.llm_clients.thinking_map import (
            ANTHROPIC_NONE_THINKING,
            ANTHROPIC_THINKING_BUDGETS,
            CODEX_REASONING_EFFORTS,
            GEMINI_THINKING_BUDGETS,
            OLLAMA_THINK,
            OPENAI_REASONING_EFFORTS,
        )

        assert ThinkingLevel.NONE in OPENAI_REASONING_EFFORTS
        assert ThinkingLevel.NONE in GEMINI_THINKING_BUDGETS
        assert ThinkingLevel.NONE in OLLAMA_THINK
        assert ThinkingLevel.NONE in CODEX_REASONING_EFFORTS
        # Anthropic NONE lives in ANTHROPIC_NONE_THINKING, not in the budgets map.
        assert ThinkingLevel.NONE not in ANTHROPIC_THINKING_BUDGETS
        assert ANTHROPIC_NONE_THINKING == {"type": "disabled"}
