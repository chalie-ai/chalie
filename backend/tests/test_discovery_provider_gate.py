# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the discovery job's LLM-provider precondition gate.

The discovery cron fires every minute as an ``IdleGatedJob``. On a fresh
install, before the user has configured any provider, every tick would drive
a CHAT turn through ``MessageProcessor.process`` — which calls
``ProviderService._select`` → ``ProviderCacheService.get_selected_provider``
(``None``) → ``ProviderCacheService.get_providers`` (``{}``) →
``build_client({})`` → ``RuntimeError("LLM config missing 'platform'")`` —
three attempts per tick, logged as "[MessageProcessor] turn 1 crashed".

The fix: ``DiscoveryJob.should_run`` calls
``cron.base.llm_provider_configured`` before touching the durable
6-hour clock. When it returns False, the tick is skipped and ``last_fired``
is never stamped — so the next tick can still fire once the user finishes
provider setup.

These tests exercise that gate in isolation: the base ``IdleGatedJob`` gates
(cron match, idle window, min-interval) and the durable 6-hour clock are all
patched to pass, so the only thing the test controls is the provider state.
"""

from unittest.mock import patch

import pytest

from cron.base import CronBase
from cron.jobs.discovery import DiscoveryJob

pytestmark = pytest.mark.unit

_PROVIDER_CACHE = "cron.jobs.discovery.CronBase.llm_provider_configured"
_BASE_SHOULD_RUN = "cron.base.IdleGatedJob.should_run"


def test_llm_provider_configured_returns_false_when_no_provider() -> None:
    """Fresh install: no selected provider and no providers at all."""
    with patch(
        "services.provider_cache_service.ProviderCacheService.get_selected_provider",
        return_value=None,
    ), patch(
        "services.provider_cache_service.ProviderCacheService.get_providers",
        return_value={},
    ):
        assert CronBase.llm_provider_configured() is False


def test_llm_provider_configured_returns_true_when_selected_provider_exists() -> None:
    """User has picked a provider — even an empty-looking dict is truthy."""
    fake_selected = {"platform": "openai", "model": "gpt-4"}
    with patch(
        "services.provider_cache_service.ProviderCacheService.get_selected_provider",
        return_value=fake_selected,
    ), patch(
        "services.provider_cache_service.ProviderCacheService.get_providers",
        return_value={},
    ):
        assert CronBase.llm_provider_configured() is True


def test_llm_provider_configured_returns_true_when_any_provider_exists() -> None:
    """No selected provider, but at least one provider in the list counts."""
    fake_providers = {"main": {"platform": "openai", "model": "gpt-4"}}
    with patch(
        "services.provider_cache_service.ProviderCacheService.get_selected_provider",
        return_value=None,
    ), patch(
        "services.provider_cache_service.ProviderCacheService.get_providers",
        return_value=fake_providers,
    ):
        assert CronBase.llm_provider_configured() is True


def test_should_run_returns_false_when_no_provider_even_when_other_gates_pass() -> None:
    """The provider gate blocks discovery even when cron/idle/interval/6h all pass."""
    job = DiscoveryJob()

    with patch(_PROVIDER_CACHE, return_value=False), patch(
        "cron.jobs.discovery.logger"
    ) as mock_logger, patch(_BASE_SHOULD_RUN, return_value=True), patch(
        "cron.jobs.discovery._DISCOVERY_TIMESTAMP"
    ) as mock_ts:
        mock_ts.load.return_value = None
        assert job.should_run() is False
        mock_logger.info.assert_called_once_with(
            "[CRON] Skipping discovery — no LLM provider configured"
        )


def test_should_run_passes_provider_gate_when_selected_provider_exists() -> None:
    """With a selected provider, the gate lets the 6-hour clock decide."""
    job = DiscoveryJob()

    with patch(_PROVIDER_CACHE, return_value=True), patch(
        _BASE_SHOULD_RUN, return_value=True
    ), patch("cron.jobs.discovery._DISCOVERY_TIMESTAMP") as mock_ts:
        mock_ts.load.return_value = None
        assert job.should_run() is True
