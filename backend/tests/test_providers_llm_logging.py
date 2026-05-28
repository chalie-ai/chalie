# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration tests for the LLM request logging hook inside Providers.send_messages().

Stubs the provider layer via Providers._resolve so no real HTTP/LLM calls fire,
and redirects _LOGS_DIR to tmp_path for full filesystem isolation.
"""

import pytest

pytestmark = pytest.mark.unit


def _make_response(text='ok', provider='anthropic', model='claude-sonnet-4-6'):
    from services.llm_service import LLMResponse
    return LLMResponse(text=text, model=model, provider=provider, tool_calls=[])


def _stub_providers(monkeypatch, response):
    """Patch Providers._resolve to return a fake provider, and _get_tools to skip config."""
    from services import providers as prov_mod

    class _FakeProvider:
        def send_messages(self, system_prompt, messages, cache_prefix=True, tools=None, thinking_mode=None):
            return response

    monkeypatch.setattr(prov_mod.Providers, '_resolve', lambda self, job: _FakeProvider())
    monkeypatch.setattr(prov_mod.Providers, '_get_tools', lambda self, job: [])


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    from services import llm_request_logger
    d = tmp_path / 'logs'
    monkeypatch.setattr(llm_request_logger, '_LOGS_DIR', d)
    return d


def _send(**overrides):
    from services.providers import Providers
    kwargs = {
        'system_prompt': 'sys',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'job': 'test',
        'tools': [],
    }
    kwargs.update(overrides)
    return Providers.instance().send_messages(**kwargs)


def test_send_messages_produces_log_with_provider_and_model(logs_dir, monkeypatch):
    """send_messages() writes one log file containing the provider/model from
    the LLMResponse — proves the hook runs AFTER the LLM call returns."""
    _stub_providers(monkeypatch, _make_response(provider='anthropic', model='claude-sonnet-4-6'))

    _send()

    files = list(logs_dir.glob('*.log'))
    assert len(files) == 1
    content = files[0].read_text(encoding='utf-8')
    assert '# provider: anthropic' in content
    assert '# model: claude-sonnet-4-6' in content


def test_log_caller_from_bound_processor(logs_dir, monkeypatch):
    """Filename starts with the bound processor's class name."""
    from services.message_processor import bind_current_processor
    _stub_providers(monkeypatch, _make_response())

    class FakeUserProc:
        pass

    with bind_current_processor(FakeUserProc()):
        _send()

    files = list(logs_dir.glob('*.log'))
    assert files[0].name.startswith('FakeUserProc-')


def test_log_caller_unknown_when_unbound(logs_dir, monkeypatch):
    """With no processor bound the filename falls back to 'unknown-'."""
    _stub_providers(monkeypatch, _make_response())

    from services import message_processor as mp_mod
    token = mp_mod._CURRENT_PROCESSOR.set(None)
    try:
        _send()
    finally:
        mp_mod._CURRENT_PROCESSOR.reset(token)

    assert list(logs_dir.glob('*.log'))[0].name.startswith('unknown-')
