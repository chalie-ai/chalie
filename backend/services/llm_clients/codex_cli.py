"""
Codex CLI thin client — shells out to the locally-installed `codex` CLI.

Subprocess-based, chat-only. One turn = one ``codex exec`` invocation.

Auth model: subscription / OAuth via ``~/.codex/auth.json``. No ``api_key``
and no ``host`` required — codex_cli is the ONLY platform with zero
credentials. Removing ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` from the
child env forces codex onto its local auth flow.

Native size-rejection signal: non-zero exit code from the subprocess.
Confirmed against codex-cli 0.143.0 (JSONL stdout, ``-o`` output file).

Depends on: services.provider_api (contract), services.llm_service (estimate_tokens).
Consumed by: services.llm_clients.factory (platform dispatch).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import ClassVar, Optional, cast

from contracts.provider_client import ProviderClient
from services.llm_clients.thinking_map import CODEX_REASONING_EFFORTS
from exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
)
from services.provider_api import (
    PROVIDER_CALL_TIMEOUT_S,
    ProviderApiRequest,
    ProviderApiResponse,
)

logger = logging.getLogger(__name__)


# ── Module-level helpers (single source of truth) ────────────────────────────


def _codex_home() -> Path:
    """Return the ``CODEX_HOME`` directory (env override or ``~/.codex``)."""
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def list_codex_models() -> list[dict[str, str | None]]:
    """Return the list of models known to codex from its local cache.

    Safe: returns ``[]`` on any error (missing file, malformed JSON, etc.).
    """
    try:
        cache_path = _codex_home() / "models_cache.json"
        if not cache_path.is_file():
            return []
        data = json.loads(cache_path.read_text())
        if not isinstance(data, dict):
            return []
        models = data.get("models", [])
        if not isinstance(models, list):
            return []
        return [
            {"id": m["slug"], "display_name": m.get("display_name")}
            for m in models
            if isinstance(m, dict) and m.get("slug")
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return []


def codex_model_context_window(slug: str | None) -> int:
    """Return the context window (tokens) for a codex model slug.

    Falls back to 272_000 (GPT-5 class default) on any lookup failure.
    """
    if not slug:
        return 272_000
    try:
        cache_path = _codex_home() / "models_cache.json"
        if not cache_path.is_file():
            return 272_000
        data = json.loads(cache_path.read_text())
        if not isinstance(data, dict):
            return 272_000
        for m in data.get("models", []):
            if isinstance(m, dict) and m.get("slug") == slug:
                return int(m.get("context_window", 272_000))
        return 272_000
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return 272_000


# ── Prompt flattening ────────────────────────────────────────────────────────


def _extract_text(content: object) -> str:
    """Extract a plain-text string from a message's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")
        )
    return ""


def _flatten_prompt(system: str, messages: list[dict[str, object]]) -> str:
    """Flatten system + messages into a single prompt string for codex.

    Images and tool_calls are dropped (chat-only) with a logged warning.
    """
    parts: list[str] = []
    if system:
        parts.append(f"System instructions:\n{system}\n")

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        text = _extract_text(content)

        # Drop images — codex exec is chat-only.
        if msg.get("image"):
            logger.warning(
                "[CodexCliClient] Dropping image from message (role=%s) — codex_cli is chat-only",
                role,
            )

        # Drop tool_calls — codex exec does not return tool calls we can execute.
        if msg.get("tool_calls"):
            logger.warning(
                "[CodexCliClient] Dropping tool_calls from message (role=%s) — codex_cli is chat-only",
                role,
            )

        prefix = {"assistant": "Assistant: ", "user": "User: ", "tool": "Tool: "}.get(role, "")
        if text:
            parts.append(f"{prefix}{text}")

    return "\n\n".join(parts)


# ── JSONL parser ─────────────────────────────────────────────────────────────


def _parse_jsonl(stdout: str) -> tuple[str, dict[str, Optional[int]], list[str]]:
    """Parse codex JSONL stdout.

    Returns (agent_text, usage, errors) where:
      - agent_text: concatenated agent_message text (or empty string)
      - usage: dict with keys tokens_input, tokens_output, tokens_thinking,
        tokens_cache_read (values may be None)
      - errors: list of error messages found (type==error or turn.failed)
    """
    agent_text = ""
    usage: dict[str, Optional[int]] = {
        "tokens_input": None,
        "tokens_output": None,
        "tokens_thinking": None,
        "tokens_cache_read": None,
    }
    errors: list[str] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        obj_type = obj.get("type")

        # Agent message text
        if obj_type == "item.completed":
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    agent_text += text

        # Turn completion → token usage (usage sits at the top level of the event)
        elif obj_type == "turn.completed":
            usage_data = obj.get("usage", {})
            if isinstance(usage_data, dict):
                usage["tokens_input"] = usage_data.get("input_tokens")
                usage["tokens_output"] = usage_data.get("output_tokens")
                usage["tokens_thinking"] = usage_data.get("reasoning_output_tokens")
                usage["tokens_cache_read"] = usage_data.get("cached_input_tokens")

        # Upstream errors
        elif obj_type == "error":
            msg = obj.get("message", "")
            if isinstance(msg, str) and msg:
                errors.append(msg)
        elif obj_type == "turn.failed":
            err = obj.get("error", {})
            if isinstance(err, dict):
                msg = err.get("message", "")
                if isinstance(msg, str) and msg:
                    errors.append(msg)

    return agent_text, usage, errors


# ── Client ───────────────────────────────────────────────────────────────────


class CodexCliClient(ProviderClient):
    CONTENT_FIELD_LABEL: ClassVar[str] = "item.completed.text"

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config
        self.model: Optional[str] = cast(Optional[str], config.get('model'))

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → ``codex exec`` subprocess → ProviderApiResponse."""
        start = time.time()

        # (a) Flatten system + messages into a single prompt.
        prompt = _flatten_prompt(dto.system, dto.messages)

        # (b) Create an isolated working directory and output file.
        tmpdir = tempfile.mkdtemp(prefix="chalie-codex-")
        outfile = Path(tmpdir) / "out.txt"

        # (c) Build argv — NO shell=True.
        codex_bin = os.environ.get("CODEX_BIN", "codex")
        argv: list[str] = [
            codex_bin, "exec", "--json", "--skip-git-repo-check",
            "--sandbox", "read-only", "--ephemeral",
            "-C", tmpdir, "-o", str(outfile),
        ]
        if self.model:
            argv += ["-m", self.model]
        # LOW is absent from the map — no override, codex uses its own default.
        effort = CODEX_REASONING_EFFORTS.get(dto.thinking_mode)
        if effort:
            argv += ["-c", f"model_reasoning_effort={effort}"]
        argv.append("-")

        # (d) Spawn subprocess with sanitized env.
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)

        proc: Optional[subprocess.Popen[str]] = None
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            stdout, stderr = proc.communicate(input=prompt, timeout=PROVIDER_CALL_TIMEOUT_S)
        except FileNotFoundError as exc:
            raise ProviderResponseError(
                "Codex CLI not found on PATH — install the codex CLI and run `codex login`",
                response_code=0,
                provider='codex_cli',
            ) from exc
        except subprocess.TimeoutExpired as exc:
            if proc is not None:
                proc.kill()
                proc.communicate()
            raise ProviderTimeoutError(
                f"codex exec exceeded {PROVIDER_CALL_TIMEOUT_S}s timeout",
                provider='codex_cli',
            ) from exc

        latency_ms = int((time.time() - start) * 1000)

        # (g) Non-zero return code.
        if proc is not None and proc.returncode != 0:
            err_text = (stderr or "").strip() or (stdout or "").strip() or f"codex exited {proc.returncode}"
            raise ProviderResponseError(
                err_text,
                response_code=0,
                provider='codex_cli',
            )

        # Always parse the JSONL stream for token usage (j) and upstream errors (i).
        jsonl_text, usage, errors = _parse_jsonl(stdout or "")

        # (h) Prefer the -o output file for the agent text; fall back to JSONL.
        agent_text = ""
        try:
            if outfile.is_file() and outfile.stat().st_size > 0:
                agent_text = outfile.read_text().strip()
        except OSError:
            pass
        if not agent_text:
            agent_text = jsonl_text

        # (i) Surface an upstream error only when no assistant text was produced.
        if errors and not agent_text:
            raise ProviderResponseError(
                errors[0],
                response_code=0,
                provider='codex_cli',
            )

        # (j) Token usage from the turn.completed event.
        tokens_input = usage.get("tokens_input")
        tokens_output = usage.get("tokens_output")
        tokens_thinking = usage.get("tokens_thinking")
        tokens_cache_read = usage.get("tokens_cache_read")

        # (k) Build response.
        log_fn = logger.info if (agent_text and agent_text.strip()) else logger.warning
        log_fn(
            "[CodexCliClient] model=%s tokens=%s+%s latency=%dms",
            self.model or "codex",
            tokens_input,
            tokens_output,
            latency_ms,
        )

        return ProviderApiResponse(
            text=agent_text,
            model=self.model or "codex",
            provider='codex_cli',
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_thinking=tokens_thinking,
            tokens_cache_read=tokens_cache_read,
            tool_calls=None,
            stop_reason='stop',
            latency_ms=latency_ms,
            response_code=200,
        )

    def get_context_limit(self) -> int:
        """Return the model's context window from codex's local cache."""
        return codex_model_context_window(self.model)

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Estimate token cost using the same heuristic as OpenAIClient.

        Adds +8000 to account for codex's large injected base-instruction
        overhead (measured ~8.7k for a trivial prompt).
        """
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        parts: list[str] = []
        if dto.system:
            parts.append(dto.system)
        for msg in dto.messages:
            parts.append(_extract_text(msg.get('content', '') or ''))
        text = ' '.join(parts)
        return estimate_tokens(text) + 8000
