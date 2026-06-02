# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""§9a area H — Metrics at the send chokepoint (spec §4e).

The ACT-loop refactor moves ALL token/request/turn counting onto the one
gateway every LLM send funnels through — ``Providers._log_after_call`` — and
collapses the ``user_messages_total`` stored counter into an on-read COUNT over
the transcript table.

These tests assert the runtime behaviour at the gateway, not in the loop:

  * H1 token totals fold into the bound processor's accumulator per send (all 5
        token fields) and accumulate across sends.
  * H2 one ``requests_total`` increment per send.
  * H3 a ``delegate:*`` channel attributes its turn to ``subagent_turns_total``.
  * H5 a ``dmn`` channel attributes its turn to ``dmn_turns_total``; a ``user``
        channel records ``requests_total`` only (no turn bucket).
  * H6 latency still recorded at the gateway (``llm_calls`` increments).
  * H7 the dashboard reads ``user_messages_total`` from the transcript COUNT, so
        a stale stored counter is ignored.
  * H8 the COUNT is exact and scoped to user/user rows (other rows don't
        inflate it).

The provider layer is stubbed via ``Providers._resolve`` so no real HTTP/LLM
call fires; the LLM-request log writer is redirected to a tmp dir.
"""

import pytest

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_response(**tokens):
    """Build an LLMResponse carrying the requested token fields."""
    from services.llm_service import LLMResponse

    defaults = dict(
        text='ok',
        model='claude-sonnet-4-6',
        provider='anthropic',
        tool_calls=[],
    )
    defaults.update(tokens)
    return LLMResponse(**defaults)


def _stub_providers(monkeypatch, response):
    """Patch Providers._resolve to a fake provider returning ``response``."""
    from services import providers as prov_mod

    class _FakeProvider:
        def send_messages(self, system_prompt, messages, cache_prefix=True,
                           tools=None, thinking_mode=None):
            return response

    monkeypatch.setattr(prov_mod.Providers, '_resolve',
                        lambda self, job: _FakeProvider())
    monkeypatch.setattr(prov_mod.Providers, '_get_tools', lambda self, job: [])


class _FakeConfig:
    """Minimal stand-in for ProcessorConfig carrying only ``channel``."""

    def __init__(self, channel):
        self.channel = channel


class _FakeProc:
    """Lightweight bound processor: real MetricsAccumulator + a config channel."""

    def __init__(self, channel, usage_class='chat'):
        from services.metrics_accumulator import MetricsAccumulator
        self._metrics = MetricsAccumulator()
        self.config = _FakeConfig(channel)
        self.USAGE_CLASS = usage_class


def _send(monkeypatch, logs_dir):
    """Fire one Providers.send_messages() through the gateway."""
    from services import llm_request_logger
    monkeypatch.setattr(llm_request_logger, '_LOGS_DIR', logs_dir)
    from services.providers import Providers
    return Providers.instance().send_messages(
        system_prompt='sys',
        messages=[{'role': 'user', 'content': 'hello'}],
        job='test',
        tools=[],
    )


@pytest.fixture
def logs_dir(tmp_path):
    return tmp_path / 'logs'


@pytest.fixture
def counter_calls(monkeypatch):
    """Capture every MetricsService.record_counter(name, value) call."""
    calls = []
    from services import metrics_service as ms_mod

    def _capture(self, metric_name, value=1):
        calls.append((metric_name, value))

    monkeypatch.setattr(ms_mod.MetricsService, 'record_counter', _capture)
    return calls


# ── H1 — token totals recorded per send at the gateway ─────────────────────────


def test_h1_token_totals_recorded_at_gateway(monkeypatch, logs_dir):
    """All five token fields on the LLMResponse fold into the bound
    processor's accumulator at the gateway — not in the loop."""
    from services.message_processor import bind_current_processor

    resp = _make_response(
        tokens_input=11, tokens_output=22, tokens_thinking=33,
        tokens_cache_read=44, tokens_cache_create=55,
    )
    _stub_providers(monkeypatch, resp)

    proc = _FakeProc('user')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)

    m = proc._metrics
    assert m.tokens_input == 11
    assert m.tokens_output == 22
    assert m.tokens_thinking == 33
    assert m.tokens_cache_read == 44
    assert m.tokens_cache_create == 55


def test_h1_two_sends_accumulate(monkeypatch, logs_dir):
    """Two sends in one turn accumulate token totals (not last-wins)."""
    from services.message_processor import bind_current_processor

    _stub_providers(monkeypatch, _make_response(tokens_input=10, tokens_output=5))

    proc = _FakeProc('user')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)
        _send(monkeypatch, logs_dir)

    assert proc._metrics.tokens_input == 20
    assert proc._metrics.tokens_output == 10


# ── H2 — one request counter per send ──────────────────────────────────────────


def test_h2_one_request_counter_per_send(monkeypatch, logs_dir, counter_calls):
    """Exactly one requests_total increment is recorded per send."""
    from services.message_processor import bind_current_processor

    _stub_providers(monkeypatch, _make_response(tokens_input=1, tokens_output=1))

    proc = _FakeProc('user')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)

    requests = [c for c in counter_calls if c[0] == 'requests_total']
    assert len(requests) == 1, counter_calls
    assert requests[0][1] == 1


# ── H3 — bucketed by the bound processor's channel ─────────────────────────────


def test_h3_delegate_channel_attribution(monkeypatch, logs_dir, counter_calls):
    """A send bound to a delegate processor attributes its turn counter to the
    subagent bucket — not to the user/dmn buckets."""
    from services.message_processor import bind_current_processor

    _stub_providers(monkeypatch, _make_response(tokens_input=2, tokens_output=2))

    proc = _FakeProc('delegate:web_search', usage_class='subconscious')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)

    names = [c[0] for c in counter_calls]
    assert 'subagent_turns_total' in names
    assert 'dmn_turns_total' not in names


# ── H5 — turn counters recorded at the gateway per channel (not post_turn) ──────


def test_h5_dmn_channel_records_dmn_turns(monkeypatch, logs_dir, counter_calls):
    """A send on the dmn channel records requests_total + dmn_turns_total."""
    from services.message_processor import bind_current_processor

    _stub_providers(monkeypatch, _make_response(tokens_input=1, tokens_output=1))

    proc = _FakeProc('dmn', usage_class='subconscious')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)

    names = [c[0] for c in counter_calls]
    assert 'requests_total' in names
    assert 'dmn_turns_total' in names
    assert 'subagent_turns_total' not in names


def test_h5_user_channel_no_turn_counter(monkeypatch, logs_dir, counter_calls):
    """A user-channel send records requests_total but neither dmn nor subagent
    turn counters (those buckets belong to other channels)."""
    from services.message_processor import bind_current_processor

    _stub_providers(monkeypatch, _make_response(tokens_input=1, tokens_output=1))

    proc = _FakeProc('user')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)

    names = [c[0] for c in counter_calls]
    assert 'requests_total' in names
    assert 'dmn_turns_total' not in names
    assert 'subagent_turns_total' not in names


# ── H6 — latency still recorded at the gateway ─────────────────────────────────


def test_h6_latency_recorded_at_gateway(monkeypatch, logs_dir):
    """record_llm_call still fires at the gateway (llm_calls increments)."""
    from services.message_processor import bind_current_processor

    _stub_providers(monkeypatch, _make_response(tokens_input=1, tokens_output=1))

    proc = _FakeProc('user')
    with bind_current_processor(proc):
        _send(monkeypatch, logs_dir)

    assert proc._metrics.llm_calls == 1
    assert proc._metrics.llm_ms >= 0


# ── H7 — user_messages_total deleted; dashboard uses read-time COUNT ────────────


def test_h7_stored_counter_does_not_affect_user_messages(db, store):
    """Recording a stale user_messages_total counter does NOT change the
    dashboard value — the dashboard reads the transcript COUNT only."""
    from services.metrics_service import MetricsService

    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES "
        "('user', 'user', 'one')"
    )
    db.commit()

    m = MetricsService()
    m.record_counter('user_messages_total', 999)  # stale / ignored
    dash = m.get_dashboard_data()
    assert dash['counters']['user_messages_total'] == 1


# ── H8 — one user turn = exactly one user/user transcript row ───────────────────


def test_h8_count_exactness_n_turns(db, store):
    """N user turns → COUNT N over (role='user' AND channel='user')."""
    from services.metrics_service import MetricsService

    for i in range(5):
        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES "
            "('user', 'user', ?)",
            (f'turn-{i}',),
        )
    # Rows on other channels/roles must not inflate the count.
    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES "
        "('user', 'assistant', 'a'), ('dmn', 'user', 'b'), "
        "('external-agent:bot', 'external_agent', 'c')"
    )
    db.commit()

    dash = MetricsService().get_dashboard_data()
    assert dash['counters']['user_messages_total'] == 5
