"""Consolidate apex episodes into super-episodes per channel.

Ported from ``services.subconscious_worker.SubconsciousWorker._step_consolidate``
and its four private helpers. Idle-gated: runs only while the user is idle and
at least ``min_interval`` has passed since the last tick.

This job is fully self-contained — every constant, helper, and import the
five methods depend on is carried verbatim, so the original
``subconscious_worker`` service could be removed once all nine steps landed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from cron.base import IdleGatedJob
from models.episode import Episode

if TYPE_CHECKING:
    from services.embedding_service import EmbeddingService
    from services.episodic_service import EpisodicService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"

# Per-tick consolidation summarization cap: at most this many cluster→parent LLM
# summarization calls run per tick across all channels and both roll-up rounds,
# so a large backlog drains over several ticks instead of stalling one tick.
_SUMMARIZATION_CLUSTER_BUDGET = 5


class ConsolidateJob(IdleGatedJob):
    """Idle-gated cron job that consolidates apex episodes into super-episodes."""

    name = "consolidate"

    def _run(self) -> str:
        """Consolidate apex episodes into super-episodes, per channel."""
        from services.embedding_service import get_embedding_service

        emb_svc = get_embedding_service()

        # Per-tick summarization budget shared across every channel and round.
        # Reset each tick so a backlog drains over successive ticks, never
        # blowing one tick's LLM budget.
        self._summarization_budget_remaining = _SUMMARIZATION_CLUSTER_BUDGET

        channels = self._consolidating_channels()
        total_clusters = 0
        supers_written = 0
        for channel in channels:
            found, written = self._consolidate_channel(channel, emb_svc)
            total_clusters += found
            supers_written += written

        if total_clusters == 0:
            return f"checked channels={channels}, no clusters formed"
        if supers_written == 0:
            return (
                f"checked channels={channels}, {total_clusters} cluster(s) found, "
                "0 written"
            )
        return (
            f"consolidated {total_clusters} cluster(s) across {len(channels)} "
            f"channel(s), {supers_written} super-ep(s) written"
        )

    @staticmethod
    def _consolidating_channels() -> list[str]:
        """Return the channels to consolidate: the HEAVY exact channels (user,
        external agents)."""
        from services.source_profiles import (
            LIKE_EXTERNAL_AGENT,
            consolidating_exact_channels,
        )

        channels = list(consolidating_exact_channels())
        try:
            channels.extend(Episode.apex_channels(LIKE_EXTERNAL_AGENT))
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} external-agent channel discovery failed: {exc}"
            )
        return channels

    def _consolidate_channel(
        self, channel: str, emb_svc: "EmbeddingService"
    ) -> tuple[int, int]:
        """Consolidate one channel across both hierarchy rounds. Returns
        (clusters_found, supers_written)."""
        from services.episodic_constants import ERA_DIGEST_TRIGGER
        from services.episodic_service import (
            EpisodicService,
            cluster_apex_embeddings,
            find_super_candidates,
        )

        episodic_svc = EpisodicService()

        # Leaf round — count trigger lives inside find_super_candidates.
        try:
            leaf_clusters = find_super_candidates(channel)
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} find_super_candidates failed (channel={channel}): {exc}"
            )
            leaf_clusters = []

        # Era round — cluster stored level-1 summary embeddings when enough
        # level-1 apexes have accumulated. The count gate is explicit here
        # because the era round reads its own (level=1) apex set.
        # Intentional one-tick lag: this reads level-1 apexes BEFORE the leaf
        # round below writes this tick's new level-1 parents, so a super is never
        # rolled into an era the same tick it is born — it waits one tick. Do not
        # "fix" this into a same-tick re-read.
        era_clusters: list[list[str]] = []
        try:
            l1_ids, l1_embs = Episode.apex_embeddings(channel, level=1)
            if len(l1_ids) >= ERA_DIGEST_TRIGGER:
                era_clusters = cluster_apex_embeddings(l1_ids, l1_embs)
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} era clustering failed (channel={channel}): {exc}"
            )

        # Leaf clusters → level-1 parents; era clusters → level-2 parents.
        rounds = ((leaf_clusters, 1), (era_clusters, 2))
        found = sum(len(clusters) for clusters, _ in rounds)
        if found == 0:
            return 0, 0

        written = 0
        for clusters, level in rounds:
            written += self._write_round(
                channel, clusters, level, emb_svc, episodic_svc
            )
        return found, written

    def _write_round(
        self,
        channel: str,
        clusters: list[list[str]],
        level: int,
        emb_svc: "EmbeddingService",
        episodic_svc: "EpisodicService",
    ) -> int:
        """Write one roll-up round's clusters at ``level``. Returns supers written."""
        if not clusters or self._summarization_budget_remaining <= 0:
            return 0

        try:
            prior_embeddings = Episode.novelty_comparison_blobs(channel)
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} novelty comparison-set fetch failed "
                f"(channel={channel}): {exc}"
            )
            prior_embeddings = []

        written = 0
        for cluster_ids in clusters:
            if self._summarization_budget_remaining <= 0:
                break
            if self._write_super_episode(
                channel, cluster_ids, level, emb_svc, episodic_svc, prior_embeddings
            ):
                written += 1
                self._summarization_budget_remaining -= 1
        return written

    @staticmethod
    def _write_super_episode(
        channel: str,
        cluster_ids: list[str],
        level: int,
        emb_svc: "EmbeddingService",
        episodic_svc: "EpisodicService",
        prior_embeddings: list[bytes],
    ) -> bool:
        """Encode + store one parent episode for a cluster. Returns True on write."""
        from configs.channels import (
            SuperEpisodeConfig,
            _collect_transcript_ids,
            _safe_json_load_object,
            _spans_for_level,
        )
        from services.episodic_constants import HDBSCAN_MIN_CLUSTER_SIZE
        from services.episodic_service import compute_novelty, compute_salience
        from controllers.message_processor import MessageProcessor

        try:
            sources = [
                ep.to_dict() for ep in (
                    Episode.by_id(eid) for eid in cluster_ids
                )
                if ep
            ]
            if len(sources) < HDBSCAN_MIN_CLUSTER_SIZE:
                return False

            # all_t_ids is collected at every level for lineage/provenance
            # (transcript_id_start/end stamped below). Raw spans are fetched only
            # at the leaf level; higher levels distil the child gists alone. See
            # _spans_for_level for the rationale.
            all_t_ids = _collect_transcript_ids(cast(list[object], sources))
            transcript_spans = _spans_for_level(all_t_ids, level)

            config = SuperEpisodeConfig(
                channel, cast(list[object], sources), transcript_spans
            )
            response = MessageProcessor.process(config).result()

            if not response:
                logger.warning(
                    f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response "
                    f"for cluster {cluster_ids}"
                )
                return False

            super_ep = _safe_json_load_object(response)
            if not super_ep or not super_ep.get("gist"):
                logger.warning(
                    f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty "
                    f"gist for cluster {cluster_ids}"
                )
                return False

            super_ep["channel"] = channel
            super_ep["level"] = level
            unique_t_ids = sorted(all_t_ids)
            super_ep["transcript_ids"] = unique_t_ids
            super_ep["transcript_id_start"] = (
                min(unique_t_ids) if unique_t_ids else None
            )
            super_ep["transcript_id_end"] = (
                max(unique_t_ids) if unique_t_ids else None
            )
            super_ep["consolidated_from"] = [ep["id"] for ep in sources]

            gist = cast(str, super_ep["gist"])
            embedding = emb_svc.generate_embedding(gist)
            novelty = (
                compute_novelty(embedding, prior_embeddings) if embedding else 1.0
            )
            super_ep["salience"] = compute_salience(
                has_open_loop=bool(super_ep.get("has_open_loop", False)),
                novelty=novelty,
            )
            super_ep.pop("has_open_loop", None)

            new_id = episodic_svc.store_episode(super_ep, embedding=embedding)
            for src_id in cluster_ids:
                Episode.set_consolidated_into(src_id, new_id)

            logger.info(
                f"{LOG_PREFIX} level-{level} episode {new_id} created from cluster "
                f"{cluster_ids} (channel={channel})"
            )
            return True

        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} level-{level} consolidation failed for cluster "
                f"{cluster_ids} (channel={channel}): {exc}"
            )
            return False
