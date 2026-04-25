"""
SuperEpisodeEncoderProcessor — self-gating processor for super-episode synthesis.

Usage::

    summary = SuperEpisodeEncoderProcessor(channel='user-123').send()
    # summary → "consolidated 2 clusters, 3 super-eps written" (or "")

The processor is self-gating:
  1. Calls find_super_candidates(channel) to discover semantic clusters.
  2. If no clusters, returns '' immediately (zero LLM call, zero side effects).
  3. For each cluster, runs one ACT-loop call to synthesise a consolidated gist,
     computes embedding + novelty + salience, writes the super-episode, and sets
     consolidated_into back-pointers on every leaf.
  4. Per-cluster failures are caught and logged — one bad cluster does not block
     the rest.

North star: /Volumes/llm/chalie-plans/message-processing.md
"""

import json
import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import SuperEpisodeEncoderSystemPrompt

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SUPER_EP]"


def _strip_code_fence(text: str) -> str:
    """Remove a single markdown code fence wrapper (```[lang]...```) from text."""
    open_end = text.find("```")
    if open_end == -1:
        return text
    newline = text.find("\n", open_end)
    if newline == -1:
        return text
    close_start = text.rfind("```", newline)
    if close_start <= newline:
        return text
    return text[newline + 1 : close_start].strip()


def _safe_json_load_object(text: str) -> dict:
    """Parse a JSON object from LLM response text. Returns {} on any failure."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = _strip_code_fence(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned non-dict JSON")
        return {}
    except ValueError:
        logger.warning(f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable JSON")
        return {}


def _parse_transcript_ids_field(raw) -> list:
    """Normalize an episode.transcript_ids field (JSON string or list) to a list."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    return []


def _collect_transcript_ids(episodes: list[dict]) -> set[int]:
    """Return the union of all transcript IDs referenced by *episodes*.

    Handles both JSON-string and list forms of the ``transcript_ids`` field,
    and also expands the ``transcript_id_start``–``transcript_id_end`` range so
    overlap rows are always included.
    """
    ids: set[int] = set()
    for ep in episodes:
        for tid in _parse_transcript_ids_field(ep.get('transcript_ids')):
            if tid is None:
                continue
            try:
                ids.add(int(tid))
            except (TypeError, ValueError):
                pass

    starts = [ep['transcript_id_start'] for ep in episodes if ep.get('transcript_id_start') is not None]
    ends = [ep['transcript_id_end'] for ep in episodes if ep.get('transcript_id_end') is not None]
    if starts and ends:
        ids.update(range(min(starts), max(ends) + 1))

    return ids


def _fetch_transcript_spans(sources: list[dict], db) -> str:
    """Fetch and format the raw transcript rows spanning all source episodes.

    Collects the union of transcript IDs from the source episodes, then
    fetches those rows and formats them as ``[id] (ts) role: content`` lines.

    Returns formatted string oldest-first, or '' if no transcript IDs found.
    """
    all_t_ids = _collect_transcript_ids(sources)
    if not all_t_ids:
        return ''

    try:
        placeholders = ','.join('?' * len(all_t_ids))
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, role, content, tool_name, created_at
                FROM transcript
                WHERE id IN ({placeholders})
                ORDER BY id ASC
                """,
                list(all_t_ids),
            )
            rows = cursor.fetchall()
            cursor.close()

        lines = []
        for r in rows:
            entry_id, role, content, tool_name, created_at = r
            if tool_name:
                lines.append(f"[{entry_id}] ({created_at}) {role} [{tool_name}]: {content}")
            else:
                lines.append(f"[{entry_id}] ({created_at}) {role}: {content}")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} _fetch_transcript_spans failed: {exc}")
        return ''


class SuperEpisodeEncoderProcessor(MessageProcessor):
    """Self-gating processor that synthesises semantic clusters into super-episodes.

    Pass ``channel`` to the constructor. Call ``send()`` — the processor does its
    own gating (find_super_candidates) and either returns '' immediately when there
    is nothing to do, or runs one ACT-loop call per cluster and returns a structured
    summary string.

    Usage::

        summary = SuperEpisodeEncoderProcessor(channel='user-abc').send()
        # "consolidated 2 clusters, 3 super-eps written" or ""

    One instance per call.  Per-cluster state (sources, spans) is set just before
    each ACT loop and torn down after.
    """

    CHANNEL = 'super_episode_encoder'
    ROLE = 'super_episode_encoder'
    JOB = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = SuperEpisodeEncoderSystemPrompt
    NATIVE_TOOLS: list[str] = []
    MAX_ITERATIONS = 1
    MAX_TIMEOUT = 120  # seconds
    SKIP_TRANSCRIPT_WRITE = True

    def __init__(
        self,
        channel: str,
        metadata: dict | None = None,
    ):
        super().__init__(raw_input='', metadata=metadata)
        self._target_channel = channel
        # Per-cluster state — set by send() before each ACT loop
        self._sources: list[dict] = []
        self._spans: str = ''

    def getUserDefinition(self) -> str:
        return (
            "The user is 'super_episode_encoder' — a background process that "
            "consolidates clusters of related episodes into a single super-episode."
        )

    def getUserPrompt(self) -> str:
        src = "\n\n".join(f"[{e['id']}] {e['gist']}" for e in self._sources)
        return (
            f"Source episodes:\n\n{src}\n\n"
            f"Raw transcript spans covering these episodes:\n\n{self._spans}"
        )

    def postTurn(self) -> None:
        """No post-turn fan-out — orchestration lives in send()."""
        pass

    # ── Public entry point ────────────────────────────────────────────────────

    def send(self, request_id: str | None = None) -> str:  # type: ignore[override]
        """Gate, cluster, and synthesise.  Returns a summary string or ''.

        Overrides MessageProcessor.send() — the gating loop wraps the parent
        send() call once per cluster rather than exposing the raw ACT loop.

        Steps:
          1. find_super_candidates(channel) → clusters.  Return '' if empty.
          2. Hoist novelty comparison set once (channel-scoped).
          3. For each cluster: build sources → fetch spans → run parent send()
             once → parse response → compute salience → store super-ep →
             set consolidated_into back-pointers.
          4. Return summary string.

        Any per-cluster failure is caught and logged; remaining clusters
        continue unaffected.
        """
        from services.database_service import get_shared_db_service
        from services.episodic_service import (
            EpisodicService,
            find_super_candidates,
            _fetch_novelty_comparison_set,
            compute_novelty,
        )
        from services.episodic_constants import SUPER_EPISODE_MIN_CLUSTER
        from services.salience_service import compute_salience
        from services.config_service import ConfigService
        from services.embedding_service import get_embedding_service

        try:
            clusters = find_super_candidates(self._target_channel)
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} find_super_candidates failed: {exc}")
            return ''

        if not clusters:
            return ''

        db = get_shared_db_service()

        try:
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        except Exception:
            episodic_config = {}

        episodic_svc = EpisodicService(db, episodic_config)
        emb_svc = get_embedding_service()

        # Hoisted once — minor staleness between clusters is acceptable.
        # novelty is a soft bias on salience, not a correctness gate.
        try:
            prior_embeddings = _fetch_novelty_comparison_set(self._target_channel)
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} _fetch_novelty_comparison_set failed: {exc}")
            prior_embeddings = []

        supers_written = 0

        for cluster_ids in clusters:
            try:
                wrote = self._process_cluster(
                    cluster_ids=cluster_ids,
                    db=db,
                    episodic_svc=episodic_svc,
                    emb_svc=emb_svc,
                    prior_embeddings=prior_embeddings,
                    compute_novelty=compute_novelty,
                    compute_salience=compute_salience,
                    min_cluster=SUPER_EPISODE_MIN_CLUSTER,
                )
                if wrote:
                    supers_written += 1
            except Exception as exc:
                logger.warning(
                    f"{LOG_PREFIX} Super-episode creation failed for cluster {cluster_ids}: {exc}"
                )

        if supers_written == 0:
            return ''

        return f"consolidated {len(clusters)} cluster(s), {supers_written} super-ep(s) written"

    # ── Private per-cluster orchestration ─────────────────────────────────────

    def _process_cluster(
        self,
        cluster_ids: list[str],
        db,
        episodic_svc,
        emb_svc,
        prior_embeddings: list[bytes],
        compute_novelty,
        compute_salience,
        min_cluster: int,
    ) -> bool:
        """Assemble sources, invoke the LLM, compute salience, and persist one super-episode.

        Returns True if a super-episode was successfully written.
        """
        sources = [ep for ep in (episodic_svc.get_episode_by_id(eid) for eid in cluster_ids) if ep]
        if len(sources) < min_cluster:
            return False

        transcript_spans = _fetch_transcript_spans(sources, db)

        # Wire per-cluster state so getUserPrompt() picks it up
        self._sources = sources
        self._spans = transcript_spans

        response = super().send()

        # Tear down per-cluster state immediately
        self._sources = []
        self._spans = ''

        if not response:
            logger.warning(
                f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response for cluster {cluster_ids}"
            )
            return False

        super_ep = _safe_json_load_object(response)
        if not super_ep or not super_ep.get('gist'):
            logger.warning(
                f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty gist "
                f"for cluster {cluster_ids}"
            )
            return False

        super_ep['channel'] = self._target_channel
        unique_t_ids = sorted(_collect_transcript_ids(sources))
        super_ep['transcript_ids'] = unique_t_ids
        super_ep['transcript_id_start'] = min(unique_t_ids) if unique_t_ids else None
        super_ep['transcript_id_end'] = max(unique_t_ids) if unique_t_ids else None
        super_ep['consolidated_from'] = [ep['id'] for ep in sources]

        gist = super_ep.get('gist', '')
        embedding = emb_svc.generate_embedding(gist) if gist else None
        novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
        super_ep['salience'] = compute_salience(
            valence=float(super_ep.get('emotional_valence') or 0.0),
            arousal=float(super_ep.get('emotional_arousal') or 0.0),
            has_open_loop=bool(super_ep.get('has_open_loop', False)),
            novelty=novelty,
        )
        super_ep.pop('has_open_loop', None)

        new_id = episodic_svc.store_episode(super_ep, embedding=embedding)
        for src_id in cluster_ids:
            episodic_svc.set_consolidated_into(src_id, new_id)

        logger.info(f"{LOG_PREFIX} Super-episode {new_id} created from cluster {cluster_ids}")
        return True
