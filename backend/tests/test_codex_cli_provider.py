# Feature tests for the codex_cli LLM platform.
#
# Real-stack, no mocks: the codex CLI is exercised through a *stub binary* (a
# real executable script wired in via the CODEX_BIN env var) so the whole
# subprocess path — argv, stdin prompt, -o output file, JSONL usage parsing,
# env sanitisation — runs end to end without spending any real subscription
# tokens. codex_cli is the one platform that needs neither api_key nor host;
# auth is the local ~/.codex/auth.json, faked here via CODEX_HOME.

import json
import sqlite3
import stat
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from flask.testing import FlaskClient

if TYPE_CHECKING:
    from services.provider_api import ProviderApiResponse

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel
import services.vault_service as _vault_mod
from services.vault_service import _vault_state, get_vault_service


def _unwrap_listing(body: "dict[str, object]") -> "list[dict[str, object]]":
    """Assert the success listing envelope shape and return the result array."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("list[dict[str, object]]", body["result"])


def _unwrap_single(body: "dict[str, object]") -> "dict[str, object]":
    """Assert the success single-resource envelope shape and return the result dict."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("dict[str, object]", body["result"])

# A stub that impersonates `codex`. `--version` prints a banner; `exec` reads the
# prompt from stdin, writes the agent message to the -o file, and emits the 0.143.0
# JSONL event stream (including real-shaped token usage). It reports whether
# OPENAI_API_KEY survived into its environment so the env-strip contract is testable.
_STUB_CODEX = """#!/usr/bin/env python3
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
key_state = "present" if os.environ.get("OPENAI_API_KEY") else "absent"
text = "STUB key=%s promptlen=%d" % (key_state, len(prompt))

if outfile:
    with open(outfile, "w") as f:
        f.write(text)

print(json.dumps({"type": "thread.started", "thread_id": "t1"}))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "item.completed",
                  "item": {"id": "item_0", "type": "agent_message", "text": text}}))
print(json.dumps({"type": "turn.completed",
                  "usage": {"input_tokens": 8762, "cached_input_tokens": 7552,
                            "output_tokens": 31, "reasoning_output_tokens": 23}}))
sys.exit(0)
"""

_MODELS_CACHE = {
    "fetched_at": "2026-07-09T00:00:00Z",
    "models": [
        {"slug": "gpt-5.5", "display_name": "GPT-5.5", "context_window": 272000},
        {"slug": "gpt-5.4-mini", "display_name": "GPT-5.4-Mini", "context_window": 272000},
        {"slug": "codex-auto-review", "display_name": "Codex Auto Review", "context_window": 272000},
    ],
}


def _unlock_vault(password: str = "test-password") -> None:
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


def _install_stub_codex(tmp_path: Path) -> str:
    """Write the stub codex binary and return its path (executable)."""
    binary = tmp_path / "codex-stub"
    binary.write_text(_STUB_CODEX)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(binary)


def _install_codex_home(tmp_path: Path, *, logged_in: bool = True) -> Path:
    """Create a fake ~/.codex with a models cache and (optionally) auth.json."""
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "models_cache.json").write_text(json.dumps(_MODELS_CACHE))
    if logged_in:
        (home / "auth.json").write_text(json.dumps({"tokens": {"access_token": "x"}}))
    return home


@pytest.mark.unit
class TestCodexCliProvider:

    @pytest.fixture(autouse=True)
    def _reset_vault_state(self) -> object:
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None
        yield
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

    # ------------------------------------------------------------------
    # Catalog — codex_cli is a credential-less, host-less preset.
    # ------------------------------------------------------------------

    def test_codex_cli_is_in_the_curated_catalog(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client
        body = cast("dict[str, object]", client.get('/api/providers/catalog').get_json())
        catalog = _unwrap_listing(body)
        by_id = {cast(str, p['id']): p for p in catalog}
        assert 'codex_cli' in by_id, "codex_cli preset missing from catalog"
        preset = by_id['codex_cli']
        assert preset['platform'] == 'codex_cli'
        assert preset['host'] == ''
        assert preset['needs_key'] is False

    # ------------------------------------------------------------------
    # Factory — dispatch works with neither api_key nor host.
    # ------------------------------------------------------------------

    def test_build_client_returns_codex_client_without_credentials(self) -> None:
        from services.llm_clients.codex_cli import CodexCliClient
        from services.llm_clients.factory import build_client

        client = build_client({'platform': 'codex_cli', 'model': 'gpt-5.5'})
        assert isinstance(client, CodexCliClient)
        assert client.CONTENT_FIELD_LABEL == 'item.completed.text'

    # ------------------------------------------------------------------
    # Stored + read back with no key and no host; vision never probed.
    # ------------------------------------------------------------------

    def test_provider_stored_without_key_or_host_and_vision_off(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client
        _unlock_vault()

        resp = client.post('/api/providers/-1', json={
            'name': 'my-codex',
            'platform': 'codex_cli',
            'model': 'gpt-5.5',
        })
        assert resp.status_code == 201, resp.get_data(as_text=True)
        created = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        assert created['platform'] == 'codex_cli'
        assert created['model'] == 'gpt-5.5'
        # chat-only: the vision probe must be skipped (never spawns codex).
        assert created['supports_vision'] in (0, False)

    # ------------------------------------------------------------------
    # Model list is sourced from codex's own local cache.
    # ------------------------------------------------------------------

    def test_list_codex_models_reads_local_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from services.llm_clients.codex_cli import list_codex_models

        home = _install_codex_home(tmp_path)
        monkeypatch.setenv('CODEX_HOME', str(home))

        models = list_codex_models()
        ids = {m['id'] for m in models}
        assert ids == {'gpt-5.5', 'gpt-5.4-mini', 'codex-auto-review'}
        assert any(m['display_name'] == 'GPT-5.5' for m in models)

    def test_context_limit_from_cache_with_default_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from services.llm_clients.codex_cli import CodexCliClient

        home = _install_codex_home(tmp_path)
        monkeypatch.setenv('CODEX_HOME', str(home))

        assert CodexCliClient({'platform': 'codex_cli', 'model': 'gpt-5.5'}).get_context_limit() == 272000
        # Unknown slug → documented default.
        assert CodexCliClient({'platform': 'codex_cli', 'model': 'nope'}).get_context_limit() == 272000

    # ------------------------------------------------------------------
    # send() — the full subprocess path via a real stub binary.
    # ------------------------------------------------------------------

    def _send(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, object]]) -> "ProviderApiResponse":
        from services.llm_clients.codex_cli import CodexCliClient
        from services.provider_api import ProviderApiRequest

        monkeypatch.setenv('CODEX_BIN', _install_stub_codex(tmp_path))
        monkeypatch.setenv('CODEX_HOME', str(_install_codex_home(tmp_path)))

        client = CodexCliClient({'platform': 'codex_cli', 'model': 'gpt-5.5'})
        dto = ProviderApiRequest(
            system="You are helpful.",
            messages=messages,
            type=ProviderType.CHAT,
            thinking_mode=ThinkingLevel.LOW,
            cache_prefix=False,
            max_tokens=64,
        )
        return client.send(dto)

    def test_send_returns_text_and_parses_usage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, [{"role": "user", "content": "hello"}])

        assert resp.text.startswith("STUB")
        assert resp.provider == 'codex_cli'
        assert resp.model == 'gpt-5.5'
        assert resp.stop_reason == 'stop'
        assert resp.tool_calls is None          # chat-only, always
        # usage from turn.completed
        assert resp.tokens_input == 8762
        assert resp.tokens_output == 31
        assert resp.tokens_thinking == 23
        assert resp.tokens_cache_read == 7552

    def test_send_strips_openai_api_key_from_child_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A stray key in the parent env must NOT leak to codex (keeps it on OAuth).
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-should-be-stripped')
        resp = self._send(tmp_path, monkeypatch, [{"role": "user", "content": "hi"}])
        assert "key=absent" in resp.text

    def test_send_drops_images_and_tool_calls_without_failing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = self._send(tmp_path, monkeypatch, [
            {"role": "user", "content": "look", "image": "data:image/png;base64,AAAA"},
            {"role": "assistant", "content": "prior", "tool_calls": [{"id": "1", "name": "x"}]},
            {"role": "user", "content": "final question"},
        ])
        assert resp.text.startswith("STUB")   # survived; images/tool_calls dropped
        assert resp.tool_calls is None

    def test_estimate_request_tokens_includes_overhead(self) -> None:
        from services.llm_clients.codex_cli import CodexCliClient
        from services.provider_api import ProviderApiRequest

        client = CodexCliClient({'platform': 'codex_cli', 'model': 'gpt-5.5'})
        dto = ProviderApiRequest(
            system="", messages=[{"role": "user", "content": "hi"}],
            type=ProviderType.CHAT, thinking_mode=ThinkingLevel.LOW,
            cache_prefix=False, max_tokens=1,
        )
        # codex injects ~8.7k base-instruction tokens — the estimate must account for it.
        assert client.estimate_request_tokens(dto) >= 8000

    # ------------------------------------------------------------------
    # Connectivity test — presence check only, ZERO inference tokens.
    # ------------------------------------------------------------------

    def test_test_endpoint_reports_binary_missing(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _db, _store = authed_client
        monkeypatch.setenv('CODEX_BIN', str(tmp_path / 'does-not-exist'))

        resp = client.post('/api/providers/test', json={
            'platform': 'codex_cli', 'model': 'gpt-5.5',
        })
        assert resp.status_code == 200
        result = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        assert result['success'] is False
        assert 'not found' in cast(str, result['error']).lower()

    def test_test_endpoint_reports_not_logged_in(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _db, _store = authed_client
        monkeypatch.setenv('CODEX_BIN', _install_stub_codex(tmp_path))
        monkeypatch.setenv('CODEX_HOME', str(_install_codex_home(tmp_path, logged_in=False)))

        resp = client.post('/api/providers/test', json={
            'platform': 'codex_cli', 'model': 'gpt-5.5',
        })
        assert resp.status_code == 200
        result = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        assert result['success'] is False
        assert 'logged in' in cast(str, result['error']).lower()

    def test_test_endpoint_ready_when_binary_and_auth_present(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _db, _store = authed_client
        monkeypatch.setenv('CODEX_BIN', _install_stub_codex(tmp_path))
        monkeypatch.setenv('CODEX_HOME', str(_install_codex_home(tmp_path, logged_in=True)))

        resp = client.post('/api/providers/test', json={
            'platform': 'codex_cli', 'model': 'gpt-5.5',
        })
        assert resp.status_code == 200
        result = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        assert result['success'] is True
        assert 'ready' in cast(str, result['message']).lower()

    # ------------------------------------------------------------------
    # list-models endpoint routes codex_cli to the local cache.
    # ------------------------------------------------------------------

    def test_list_models_endpoint_returns_codex_models(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _db, _store = authed_client
        monkeypatch.setenv('CODEX_HOME', str(_install_codex_home(tmp_path)))

        resp = client.post('/api/providers/list-models', json={'platform': 'codex_cli'})
        assert resp.status_code == 200
        result = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        models = cast("list[dict[str, object]]", result['models'])
        ids = {m['id'] for m in models}
        assert 'gpt-5.5' in ids
