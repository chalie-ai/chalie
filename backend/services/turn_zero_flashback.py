# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Turn-0 flashback — the hot-path memory seed (spec §4.5).

Owns the single declarative behaviour the orchestrator fires once before
iteration 0 when ``config.memory_seed`` is set: ground the model's first
request in a small, curated memory bundle.

Three responsibilities, in order:

  1. **Continuation gate** — pure-math, zero-LLM. The flashback fires only on
     session start or a topic shift; it is skipped when the new message's
     embedding sits close to the running conversation centroid (a continuation
     such as "yes, do that"). This replaces the old unconditional every-message
     recall.
  2. **Topic inference** — embed the raw message; when the message is terse, the
     embed text is composed with the compaction living-doc "Now" section so
     topic continuity survives "yes"/"lol" turns. An OPTIONAL single HyDE-style
     rewrite (arXiv 2212.10496) sharpens the query when a strong provider is
     active and degrades silently to the plain embedding otherwise — the hot
     path carries zero MANDATORY LLM calls.
  3. **Curated render** — instead of injecting raw recall JSON, the seed injects
     a small bundle: up to five live facts as bullets plus up to three episodes
     as ``On <date>: <one-liner>`` flashbacks (super-episodes preferred). The
     explicit, model-invoked ``memory.recall`` keeps its JSON contract
     unchanged — this render is the seed path only.

The recall it runs uses ``caller='seed'`` so the ``memory_recall_log`` seed row
and its telemetry stay exactly as the retrieval rework left them, and it records
its own ``memory(action='recall', _auto=True)`` act-trail row so the rest of the
trail machinery (``_is_auto_memory_recall``) still recognises it as the
framework seed. Moments never appear here (§4.7).
"""

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from services.time_formatter_service import TimeFormatterService

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

#: The tool name + params the framework records for the turn-0 seed. The name and
#: the ``_auto`` flag are load-bearing: ``_is_auto_memory_recall`` keys the
#: trail-compaction exclusion on exactly this shape, and the persisted
#: ``caller='seed'`` recall_log row is frozen by the scenario lock.
_SEED_TOOL_NAME = "memory"
_SEED_ACTION = "recall"

#: Cosine similarity at/above which the new message is treated as a continuation
#: of the running conversation and the flashback is SKIPPED. Embeddings are
#: L2-normalised, so dot product == cosine. Calibrated to the active embedding
#: model (``gte-modernbert-base`` — ``embedding_service._MODEL_ID``): the median
#: pairwise cosine on this corpus is
#: ~0.55, so "closer than a typical unrelated pair" is the natural continuation
#: boundary. (The 0.80-0.85 figure cited in early design notes was the p99 of an
#: unrelated-pair band, not a per-turn continuation level — measured on-topic
#: follow-ups sit well below it: a terse turn composed with the living-doc "Now"
#: lands ~0.89, an on-topic full turn ~0.74, while genuine topic shifts fall to
#: ~0.34-0.38. The corpus median cleanly separates the two populations and is not
#: tuned to any single example.) A terse message with no resolvable topic is
#: handled separately by the embed-text builder and never reaches this threshold.
_CONTINUATION_SIMILARITY_THRESHOLD = 0.55

#: Transcript roles that make up the running conversation centroid. The
#: compaction living-doc row is deliberately excluded: it is a derived summary,
#: and its "Now" section already feeds the terse-message embed on the message
#: side — folding it into the centroid too would let it cancel itself out and
#: blind the gate to a shifted "Now".
_CENTROID_ROLES = frozenset({"user", "assistant"})

#: How many of the most recent transcript messages form the conversation
#: centroid. A short window tracks the *current* thread rather than the whole
#: session, and bounds the per-turn embedding work to the <100ms-class budget
#: (embeddings are cached by text, so warm turns add no inference).
_CENTROID_WINDOW = 6

#: A message with fewer than this many whitespace tokens is "terse" — too thin to
#: carry its own topic ("yes", "lol", "do that"). Terse messages are composed
#: with the living-doc "Now" section before embedding so topic continuity
#: survives them. ~8 tokens per spec §4.5.
_TERSE_TOKEN_CEILING = 8

#: Declared context window (tokens) at/above which the active provider is treated
#: as "strong" enough to spend the OPTIONAL HyDE rewrite on. Context window is
#: the only structural capability signal the provider config exposes
#: (``providers.max_tokens`` / ``get_context_limit``); frontier models advertise
#: large windows, small local models advertise small ones. This is a heuristic,
#: never a hard contract — a weak provider simply skips HyDE and degrades to the
#: plain embedding, so the hot path never depends on it.
_STRONG_PROVIDER_CONTEXT_FLOOR = 200_000

#: Curated-bundle caps (spec §4.5: ≤5 facts + ≤3 dated one-liners).
_MAX_FACTS = 5
_MAX_EPISODES = 3

#: Episodes at this hierarchy level or above are super-episodes / era digests and
#: are preferred over raw leaves when choosing the ≤3 dated one-liners.
_SUPER_LEVEL_FLOOR = 1

#: Over-fetch multiplier for the episode recall: pull this many times
#: ``_MAX_EPISODES`` candidates so the super-episode-preferred re-sort has leaves
#: AND supers to choose from before the final ≤3 clip, rather than clipping at
#: the retrieval layer and starving the preference.
_EPISODE_OVERFETCH_FACTOR = 3

#: Episode-gist one-liners are clipped to keep the flashback dense.
_ONELINER_CHARS = 160

#: HyDE rewrite request shaping — a single constrained generation that drafts a
#: hypothetical answer paragraph to embed instead of the bare query (HyDE,
#: 2212.10496). Kept short and tool-free; failure degrades silently.
_HYDE_SYSTEM = (
    "Write a single short paragraph that a perfect memory of this user would "
    "contain about the topic of their message. State plausible specifics as if "
    "recalling them. Output only the paragraph, no preamble."
)
_HYDE_MAX_TOKENS = 200
_HYDE_JOB_NAME = "turn_zero_hyde"


class TurnZeroFlashback:

    def __init__(self, mp: "MessageProcessor") -> None:
        self._mp = mp

    # ── Public entry ──────────────────────────────────────────────────────────

    def seed(self) -> None:
        """Run the continuation gate and, if it passes, inject the flashback.

        Never raises: a failure in any sub-step logs and leaves the turn
        un-seeded rather than aborting the user's message. The seed is a
        best-effort grounding aid, not a correctness dependency.
        """
        try:
            if self._is_unresolvable_terse():
                logger.debug(
                    "[FLASHBACK] terse turn with no living-doc 'Now' — "
                    "no new topic, treated as continuation, flashback skipped"
                )
                return
            embed_text = self._build_embed_text()
            if self._is_continuation(embed_text):
                logger.debug("[FLASHBACK] continuation — flashback skipped")
                return
            query = self._resolve_query(embed_text)
            block = self._build_flashback_block(query)
            self._record_seed(query, block)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FLASHBACK] seed failed (non-fatal): %s", exc)

    # ── Topic inference ───────────────────────────────────────────────────────

    def _is_unresolvable_terse(self) -> bool:
        """True when the turn is terse AND carries no resolvable topic.

        A terse message ("yes", "ok", "do that") introduces no topic of its own;
        the terse rule (spec §4.5) resolves it through the living-doc "Now"
        section. When no "Now" checkpoint exists yet, the message cannot name a
        new topic, so by definition it is not a topic shift — it continues the
        running thread and the flashback must not re-fire. Composing it with the
        centroid is meaningless here (it would compare a contentless token against
        the conversation), so this is decided before the embedding gate runs.

        Only applies once a conversation is under way: on session start there is
        no running thread to continue, so even a terse first message fires the
        flashback (the gate's session-start rule).
        """
        raw = (self._mp.raw_input or "").strip()
        if not self._is_terse(raw):
            return False
        if not self._prior_messages():
            return False  # session start — fire the flashback
        return not self._living_doc_now()

    def _build_embed_text(self) -> str:
        """The text the gate embeds: the raw message, composed with the living-doc
        "Now" section when the message is terse so a continuation embeds in the
        conversation's current context rather than in isolation.
        """
        raw = (self._mp.raw_input or "").strip()
        if not self._is_terse(raw):
            return raw
        now_section = self._living_doc_now()
        if not now_section:
            return raw
        return f"{now_section}\n{raw}" if raw else now_section

    @staticmethod
    def _is_terse(text: str) -> bool:
        """A message with fewer than the ceiling of whitespace tokens carries too
        little topic on its own (spec §4.5)."""
        return len(text.split()) < _TERSE_TOKEN_CEILING

    def _living_doc_now(self) -> str:
        """Extract the "Now" section from the channel's compaction living document.

        The chat-history compactor writes one living document per channel whose
        sections are markdown bullets (``- Person —``, ``- Now —``, ``- Holding —``
        …). The "Now" section runs from the ``- Now`` bullet up to the next
        top-level bullet. Returns the section body (heading stripped) or "" when
        there is no checkpoint yet.
        """
        from services import compaction_persistence  # noqa: PLC0415

        row = compaction_persistence.get_compaction(self._mp.config.channel)
        if not row:
            return ""
        doc = cast(str, row.get("compacted_text") or "").strip()
        if not doc:
            return ""
        return self._extract_now_section(doc)

    @staticmethod
    def _extract_now_section(doc: str) -> str:
        """Pull the body of the ``- Now —`` living-doc section out of *doc*.

        Section boundaries are top-level ``- `` bullets at column 0; the "Now"
        section ends at the next such bullet (or end of document). The leading
        ``Now`` label and its em-dash separator are stripped so only the content
        is embedded.
        """
        lines = doc.splitlines()
        collected: list[str] = []
        in_now = False
        for line in lines:
            is_top_bullet = line.startswith("- ")
            if is_top_bullet:
                label = line[2:].lstrip()
                starts_now = label[:3].lower() == "now"
                if in_now and not starts_now:
                    break
                if starts_now:
                    in_now = True
                    body = label[3:].lstrip(" —-:").strip()
                    if body:
                        collected.append(body)
                    continue
            if in_now:
                collected.append(line.strip())
        return "\n".join(c for c in collected if c).strip()

    def _resolve_query(self, embed_text: str) -> str:
        """The retrieval query for the flashback recall.

        Defaults to the gate's embed text. When a strong provider is active, an
        OPTIONAL single HyDE rewrite (2212.10496) replaces it with a hypothetical
        answer paragraph that embeds closer to stored memories; any failure or a
        weak provider degrades silently back to the embed text.
        """
        if not self._is_strong_provider():
            return embed_text
        rewrite = self._hyde_rewrite(embed_text)
        return rewrite or embed_text

    def _is_strong_provider(self) -> bool:
        """Heuristic capability gate for the OPTIONAL HyDE rewrite.

        Uses the declared context window — the only structural capability signal
        the provider config exposes — as a proxy for model strength. Never raises:
        an unresolved provider is treated as weak so the hot path stays LLM-free.
        """
        try:
            return self._mp.providers.get_context_limit() >= _STRONG_PROVIDER_CONTEXT_FLOOR
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FLASHBACK] provider-strength probe failed: %s", exc)
            return False

    def _hyde_rewrite(self, embed_text: str) -> str:
        """One constrained generation drafting a hypothetical answer to embed.

        Returns the drafted paragraph, or "" on any failure (the caller then
        degrades to the plain embed text). Tool-free, low-thinking, bounded
        output — the only LLM call this path ever makes, and only when a strong
        provider is active.
        """
        from services.provider_api import (  # noqa: PLC0415
            ProviderApiRequest,
            ThinkingLevel,
        )

        try:
            dto = ProviderApiRequest(
                system=_HYDE_SYSTEM,
                messages=[{"role": "user", "content": embed_text}],
                tools=None,
                thinking_mode=ThinkingLevel.LOW,
                cache_prefix=False,
                max_tokens=_HYDE_MAX_TOKENS,
                _job_name=_HYDE_JOB_NAME,
            )
            response = self._mp.providers.send(dto)
            return (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FLASHBACK] HyDE rewrite failed, degrading: %s", exc)
            return ""

    # ── Continuation gate ─────────────────────────────────────────────────────

    def _is_continuation(self, embed_text: str) -> bool:
        """True when *embed_text* is close to the running conversation centroid.

        Session start (no centroid) is never a continuation — the flashback
        always fires. A failure in the embedding/centroid math is treated as
        "not a continuation" so the gate fails open to a flashback rather than
        silently starving the model of memory.
        """
        if not embed_text:
            return False
        try:
            from services.embedding_service import get_embedding_service  # noqa: PLC0415

            centroid = self._conversation_centroid()
            if centroid is None:
                return False  # session start — fire the flashback
            emb_svc = get_embedding_service()
            msg_vec = emb_svc.generate_embedding_np(embed_text, mp=self._mp)
            similarity = self._cosine(msg_vec, centroid)
            logger.debug("[FLASHBACK] centroid similarity=%.4f", similarity)
            return similarity >= _CONTINUATION_SIMILARITY_THRESHOLD
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FLASHBACK] continuation gate failed open: %s", exc)
            return False

    def _prior_messages(self) -> list[str]:
        """The recent PRIOR user/assistant message texts on this channel.

        The running conversation as the gate sees it: the last
        ``_CENTROID_WINDOW`` non-empty user/assistant messages that precede the
        current turn, with the current message's own input row (already written
        before the seed fires) excluded. Both the session-start check and the
        centroid embedding read from this single definition so they never drift.
        An empty list means there is no prior conversation (session start).
        """
        from services import transcript_service  # noqa: PLC0415

        # Fetch one extra row so dropping the current turn still leaves a full
        # window of prior context.
        rows = transcript_service.get_recent(
            self._mp.config.channel, limit=_CENTROID_WINDOW + 1
        )
        current_uid = self._mp.uid
        return [
            cast(str, r.get("content") or "").strip()
            for r in rows
            if r.get("role") in _CENTROID_ROLES
            and cast(str, r.get("content") or "").strip()
            and (current_uid is None or r.get("id") != current_uid)
        ][:_CENTROID_WINDOW]

    def _conversation_centroid(self) -> "npt.NDArray[np.float64] | None":
        """Mean (re-normalised) embedding of the recent PRIOR channel messages.

        The centroid is the running mean of the last ``_CENTROID_WINDOW``
        non-empty messages on the channel that precede the current turn — the
        current message's own input row (already written by ``_setup`` before the
        seed fires) is excluded so the gate compares the new message against the
        conversation, not against itself. Returns None when there is no prior
        conversation (session start) so the gate fires the flashback. Embeddings
        are L2-normalised and cached by text; the mean is re-normalised to a unit
        vector for cosine.
        """
        import numpy as np  # noqa: PLC0415
        from services.embedding_service import get_embedding_service  # noqa: PLC0415

        texts = self._prior_messages()
        if not texts:
            return None

        emb_svc = get_embedding_service()
        vectors = [emb_svc.generate_embedding_np(t, mp=self._mp) for t in texts]
        mean = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            return None
        return cast("npt.NDArray[np.float64]", mean / norm)

    @staticmethod
    def _cosine(a: "npt.NDArray[np.float64]", b: "npt.NDArray[np.float64]") -> float:
        """Cosine similarity of two vectors. Inputs from the embedding service are
        L2-normalised, but the centroid mean is re-normalised by the caller, so a
        plain dot product is the cosine."""
        import numpy as np  # noqa: PLC0415

        return float(np.dot(a, b))

    # ── Retrieval + render ────────────────────────────────────────────────────

    def _build_flashback_block(self, query: str) -> str:
        """Assemble the curated flashback: ≤5 fact bullets + ≤3 dated one-liners.

        Facts come from the live data-graph lane (moments excluded by the lane
        itself); episodes come from the seed recall (``caller='seed'`` — frozen
        contract) with super-episodes preferred. Returns the rendered block, or a
        short "no relevant memory" line when both lanes are empty so the recorded
        seed row is still honest.
        """
        facts = self._recall_facts(query)
        episodes = self._recall_episodes(query)
        return self._render_block(facts, episodes)

    def _recall_facts(self, query: str) -> list[str]:
        """Up to ``_MAX_FACTS`` live data-graph fact strings for *query*.

        Reuses ``memory_retrieval._search_data_graph`` — the same live-rows-only,
        moment-excluding lane the explicit recall uses — so the seed never invents
        a second retrieval path.
        """
        from services.memory_retrieval import _search_data_graph  # noqa: PLC0415

        hits, _status = _search_data_graph(query, _MAX_FACTS)
        return [
            text
            for hit in hits
            if (text := cast(str, hit.get("text") or "").strip())
        ][:_MAX_FACTS]

    def _recall_episodes(self, query: str) -> list[dict[str, object]]:
        """Up to ``_MAX_EPISODES`` raw episodes for *query*, super-episodes first.

        Runs the shared episode recall with ``caller='seed'`` so the frozen
        ``memory_recall_log`` seed row + telemetry are written exactly as the
        retrieval rework left them. Recall is cross-channel (TKT-926): a user
        turn's flashback surfaces episodes from every episode-producing channel
        (user, dmn, external-agent:*), not just the caller's own. Returns raw
        episode dicts (gist + created_at) ordered so super-episodes / era digests
        precede leaves, then by the retrieval composite score.
        """
        from services.memory_retrieval import recall_episodes  # noqa: PLC0415

        episodes, _status = recall_episodes(
            self._mp,
            query=query,
            caller="seed",
            limit=_MAX_EPISODES * _EPISODE_OVERFETCH_FACTOR,
            return_raw=True,
        )
        ordered = sorted(
            episodes,
            key=lambda ep: (
                0 if int(cast(int, ep.get("level") or 0)) >= _SUPER_LEVEL_FLOOR else 1,
                -float(cast(float, ep.get("composite_score") or 0.0)),
            ),
        )
        return ordered[:_MAX_EPISODES]

    def _render_block(self, facts: list[str], episodes: list[dict[str, object]]) -> str:
        """Render the curated bundle the model reads in place of recall JSON."""
        sections: list[str] = []
        if facts:
            sections.append("\n".join(f"- {f}" for f in facts))
        for ep in episodes:
            line = self._render_episode(ep)
            if line:
                sections.append(line)
        if not sections:
            return "No relevant memory surfaced for this turn."
        return "\n".join(sections)

    def _render_episode(self, ep: dict[str, object]) -> str:
        """One episode as ``On <date>: <one-liner>`` (spec §4.5).

        The date is the episode's local-time creation day; an unparseable /
        missing timestamp drops the date prefix rather than emitting a bogus one.
        """
        gist = cast(str, ep.get("gist") or "").strip().replace("\n", " ")
        if not gist:
            return ""
        if len(gist) > _ONELINER_CHARS:
            gist = gist[: _ONELINER_CHARS - 1].rstrip() + "…"
        date = self._episode_date(ep.get("created_at"))
        return f"On {date}: {gist}" if date else gist

    @staticmethod
    def _episode_date(raw: object) -> str:
        """Local calendar date (YYYY-MM-DD) for a stored UTC timestamp, or "".

        Storage is timezone-aware UTC; the model sees the user's local day.
        ``parse_utc`` never raises — a sentinel/empty value yields "" so the
        caller drops the date prefix instead of rendering a year-0001 date.
        """
        if not raw:
            return ""
        return TimeFormatterService.local(cast("str | None", raw), fmt="%Y-%m-%d") or ""

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record_seed(self, query: str, block: str) -> None:
        """Record the framework seed row so the model sees the curated block.

        Writes a ``memory(action='recall', _auto=True)`` ``tool_calls`` row whose
        ``result`` is the curated block. The ``_auto`` flag keeps
        ``_is_auto_memory_recall`` recognising it as the framework seed (excluded
        from trail compaction); the result is the curated bundle, NOT recall JSON
        — that JSON contract belongs to the explicit, model-invoked recall only.
        """
        from services.act_trail import ActTrail  # noqa: PLC0415

        params = {"action": _SEED_ACTION, "query": query, "_auto": True}
        ActTrail().record(
            tool_name=_SEED_TOOL_NAME,
            params=params,
            result=block,
            transcript_id=self._mp.uid,
        )
