"""Discovery cron job — idle-gated autonomous research, at most once per 6h.

Ported verbatim from ``services.subconscious_worker._step_discovery`` so the
step remains self-contained when the source worker is deleted. The 6-hour
durable clock gate is preserved inline inside ``_run``; the base
``IdleGatedJob`` still applies its own 30-min idle window and 5-min
min-interval on top of that.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from cron.base import CronBase, IdleGatedJob
from configs.channels import DiscoveryConfig
from configs.channels.discovery import DISCOVERY_PROMPT
from services.durable_timestamp import DurableTimestamp
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# ── Durable clock gate (ported from subconscious_worker) ──────────────────────
# Self-throttling on a dedicated dual-write timestamp so the discovery loop
# fires on its own multi-hour cadence, independent of the per-tick clock.
_DISCOVERY_INTERVAL = timedelta(hours=6)
_DISCOVERY_TIMESTAMP = DurableTimestamp(
    memory_key="subconscious:discovery_last_fired_at",
    data_graph_key="discovery_last_fired_at",
    source="discovery_worker",
)


class DiscoveryJob(IdleGatedJob):
    """Idle-gated cron job for proactive autonomous research.

    The 6-hour interval is enforced inside ``_run`` by the durable
    ``_DISCOVERY_TIMESTAMP`` gate copied from the source step; the base
    class still applies its 30-min idle window and 5-min min-interval on top.
    A fresh install has no CHAT provider yet — ``llm_provider_configured``
    gates the whole job in ``should_run`` so the discovery cron never drives
    an LLM turn before any provider is set up.
    """

    name = "discovery"

    def should_run(self) -> bool:
        """Base cron/idle/interval gates, plus the dedicated 6-hour clock.

        The discovery interval is a scheduling decision, so it gates here —
        not inside ``_run``. On top of the base ``IdleGatedJob`` gates (cron
        match, 30-min idle window, 5-min min-interval) this only lets the job
        fire once its own durable 6-hour clock has elapsed.

        Additionally, the job requires a CHAT LLM provider to be configured
        (see ``cron.base.llm_provider_configured``). On a fresh install the
        discovery cron can fire ~1s before the provider-setup flow completes;
        in that window every tick would burn three provider attempts on
        "LLM config missing 'platform'" and log "[MessageProcessor] turn 1
        crashed" as the install's first act. The provider gate lives in
        ``should_run`` (never ``_run``) so a skipped tick does not stamp
        ``last_fired`` and block the next legitimate discovery after setup.
        """
        if not super().should_run():
            return False
        if not CronBase.llm_provider_configured():
            logger.info("[CRON] Skipping discovery — no LLM provider configured")
            return False
        last = _DISCOVERY_TIMESTAMP.load()
        return last is None or utc_now() - last >= _DISCOVERY_INTERVAL

    def _run(self) -> str:
        """Fire the proactive research pass and stamp the discovery clock.

        Reached only when ``should_run`` has already cleared the 6-hour gate,
        so this fires unconditionally. DiscoveryConfig reads the ``user``
        channel's history directly (its grounding is the live user spine), so
        no metadata is threaded in; the loop records anything worth keeping as
        a discovery memory on its own.
        """
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        MessageProcessor.process(DiscoveryConfig(), DISCOVERY_PROMPT)
        _DISCOVERY_TIMESTAMP.persist(utc_now())
        return "fired"
