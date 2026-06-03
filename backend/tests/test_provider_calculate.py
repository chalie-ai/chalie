# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""§9a E. Provider.calculate() — measurement.

Four tests covering E1–E4.  All tests stub _resolve and never make real HTTP
or LLM calls.  The test DB fixture (from conftest.py) is NOT needed here —
calculate() is a pure computation on the provider object.

E1. Returns a fraction of the context window (float 0.0–1.0), never an
    absolute count.
E2. Measures the real request body (same build_request_body() path as send).
E3. Returns 0.0 on error (loop skips compaction, sends).
E4. Returns 0.0 when max_tokens unknown (no division).
"""

import json
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit


def _make_fake_provider(*, body: str, context_limit: int):
    """Return a fake LLM service whose build_request_body returns *body*
    and whose get_context_limit returns *context_limit*."""
    fake = MagicMock()
    fake.build_request_body.return_value = body
    fake.get_context_limit.return_value = context_limit
    return fake


class TestProviderCalculateE1ReturnsFraction:
    """E1: calculate() returns a float 0.0–1.0, never a raw token count."""

    def test_result_is_float_between_zero_and_one(self, monkeypatch):
        from services.providers import Providers

        # A body of ~10 words → ~13 tokens; context limit 10 000 → fraction ≈ 0.0013
        fake_provider = _make_fake_provider(
            body=json.dumps({"messages": [{"role": "user", "content": "hello world test"}]}),
            context_limit=10_000,
        )
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        pct = Providers.instance().calculate('sys', 'hello world test', [], job='unified')

        assert isinstance(pct, float), "calculate() must return a float"
        assert 0.0 <= pct <= 1.0, f"calculate() must return a value in [0,1], got {pct}"

    def test_result_is_fraction_not_absolute_count(self, monkeypatch):
        from services.providers import Providers
        from services.llm_service import estimate_tokens

        body_str = json.dumps({"messages": [{"role": "user", "content": "a b c d e"}]})
        context_limit = 100_000
        fake_provider = _make_fake_provider(body=body_str, context_limit=context_limit)
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        pct = Providers.instance().calculate('sys', 'a b c d e', [], job='unified')

        # The result must be < 1 (it is a fraction of 100k), not a raw token count
        assert pct < 1.0, (
            f"calculate() should return a fraction (got {pct}), not a raw token count"
        )
        # Raw token count for our body is much larger than 1
        raw_tokens = estimate_tokens(body_str)
        assert raw_tokens > 1, "test setup: body should have multiple tokens"
        assert pct != raw_tokens, "calculate() must return a fraction, not a token count"


class TestProviderCalculateE2MeasuresRealBody:
    """E2: calculate() builds the real request body via build_request_body(),
    the identical serialisation path used by send_messages()."""

    def test_build_request_body_is_called_with_correct_args(self, monkeypatch):
        from services.providers import Providers

        fake_provider = _make_fake_provider(body='{}', context_limit=50_000)
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        system = 'You are a helpful assistant.'
        user_body = 'What is 2+2?'
        tools = [{'name': 'calculator', 'description': 'does math'}]

        Providers.instance().calculate(system, user_body, tools, job='unified')

        # build_request_body must have been called exactly once with system,
        # a single-element user-role message list, and tools — the same
        # arguments send_messages() would pass.
        fake_provider.build_request_body.assert_called_once()
        call_args = fake_provider.build_request_body.call_args
        positional = call_args[0]
        assert positional[0] == system, "first arg must be the system prompt"
        assert positional[1] == [{'role': 'user', 'content': user_body}], (
            "second arg must be a single-element user-role message list"
        )
        assert positional[2] == tools, "third arg must be the tools list"

    def test_token_count_uses_body_not_raw_input(self, monkeypatch):
        """The token count must come from the serialised body, not from
        user_body directly.  If provider serialisation adds overhead (headers,
        extra JSON keys) that overhead must be reflected in the fraction."""
        from services.providers import Providers
        from services.llm_service import estimate_tokens

        # A 'fat' body that has far more tokens than the raw user_body alone
        fat_body = json.dumps({'messages': ['word ' * 1000]})
        fake_provider = _make_fake_provider(body=fat_body, context_limit=200_000)
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        pct = Providers.instance().calculate('sys', 'hello', [], job='unified')

        # The fraction must reflect the fat body, not the tiny user input
        raw_tokens_of_user_body = estimate_tokens('hello')
        fraction_from_user_body_only = raw_tokens_of_user_body / 200_000
        assert pct > fraction_from_user_body_only, (
            "calculate() must measure the full serialised body, not just user_body"
        )


class TestProviderCalculateE3ReturnsZeroOnError:
    """E3: calculate() returns 0.0 on any error so the loop can skip
    compaction and proceed to send_messages()."""

    def test_returns_zero_when_resolve_raises(self, monkeypatch):
        from services.providers import Providers

        def _raise(job):
            raise RuntimeError("no provider configured")

        monkeypatch.setattr(Providers, '_resolve', lambda self, job: _raise(job))

        pct = Providers.instance().calculate('sys', 'hello', [], job='unified')

        assert pct == pytest.approx(0.0), (
            f"calculate() must return 0.0 on _resolve failure, got {pct}"
        )

    def test_returns_zero_when_build_request_body_raises(self, monkeypatch):
        from services.providers import Providers

        fake_provider = MagicMock()
        fake_provider.build_request_body.side_effect = RuntimeError("serialisation failed")
        fake_provider.get_context_limit.return_value = 100_000
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        pct = Providers.instance().calculate('sys', 'hello', [], job='unified')

        assert pct == pytest.approx(0.0), (
            f"calculate() must return 0.0 when build_request_body raises, got {pct}"
        )


class TestProviderCalculateE4ReturnsZeroWhenMaxTokensUnknown:
    """E4: calculate() returns 0.0 when max_tokens is unknown (zero / None)
    so there is no zero-division error."""

    def test_returns_zero_when_context_limit_is_zero(self, monkeypatch):
        from services.providers import Providers

        fake_provider = _make_fake_provider(body='{"messages":[]}', context_limit=0)
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        pct = Providers.instance().calculate('sys', 'hello', [], job='unified')

        assert pct == pytest.approx(0.0), (
            f"calculate() must return 0.0 when context_limit is 0, got {pct}"
        )

    def test_returns_zero_when_context_limit_is_none(self, monkeypatch):
        from services.providers import Providers

        fake_provider = _make_fake_provider(body='{"messages":[]}', context_limit=None)
        monkeypatch.setattr(Providers, '_resolve', lambda self, job: fake_provider)

        pct = Providers.instance().calculate('sys', 'hello', [], job='unified')

        assert pct == pytest.approx(0.0), (
            f"calculate() must return 0.0 when context_limit is None, got {pct}"
        )
