# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Turn-0 flashback — the hot-path memory seed.

Owns the single declarative behaviour the orchestrator fires once before
iteration 0 when ``config.memory_seed`` is set: ground the model's first
request in a small, curated memory bundle.

Three responsibilities, in order:

  1. **Terse gate** — pure-string, zero-LLM. A terse message ("yes", "lol",
     "do that") is the signal that the user is mid-conversation: the model
     already holds the full thread in context and needs no recall, and the
     message carries no topic of its own to recall against. Terse turns skip
     the flashback outright.
  2. **Continuation gate** — pure-math, zero-LLM. For substantive messages the
     flashback fires only on session start or a topic shift; it is skipped when
     the message's embedding sits close to the running conversation centroid
     (a continuation such as "yes, go ahead and book that ferry").
  3. **Curated render** — the seed runs ``memory.recall`` with the raw user
     message as the query (no rewriting, no steering — purely mechanical) and
     injects a small bundle: up to five live facts as bullets plus up to three
     episodes as ``On <date>: <gist>`` flashbacks, each gist in full
     (super-episodes preferred). The explicit, model-invoked ``memory.recall``
     keeps its JSON contract unchanged — this render is the seed path only.

The recall it runs uses ``caller='seed'`` so the ``memory_recall_log`` seed row
and its telemetry stay exactly as the retrieval rework left them, and it records
its own ``memory(action='recall', _auto=True)`` act-trail row so the seed recall
and its curated result are visible in the turn's act-trail, marked ``_auto`` as
the framework seed (distinct from a model-invoked recall). Moments never appear
here in the seed.
"""

import logging
from typing import TYPE_CHECKING, cast

from services.time_formatter_service import TimeFormatterService

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

#: The tool name + params the framework records for the turn-0 seed. The
#: ``_auto`` flag marks the recorded act-trail row as the framework seed (vs a
#: model-invoked recall); the persisted ``caller='seed'`` recall_log row is
#: frozen by the scenario lock.
_SEED_TOOL_NAME = "memory"
_SEED_ACTION = "recall"

#: Cosine similarity at/above which the new message is treated as a continuation
#: of the running conversation and the flashback is SKIPPED. Embeddings are
#: L2-normalised, so dot product == cosine. Calibrated to the active embedding
#: model (``gte-modernbert-base`` — ``embedding_service._MODEL_ID``): the median
#: pairwise cosine on this corpus is ~0.55, so "closer than a typical unrelated
#: pair" is the natural continuation boundary. Measured on-topic full turns sit
#: well above it (~0.74) while genuine topic shifts fall to ~0.34-0.38, so the
#: corpus median cleanly separates the two populations and is not tuned to any
#: single example. Terse messages never reach this gate — they are skipped first.
_CONTINUATION_SIMILARITY_THRESHOLD = 0.55

#: Transcript roles that make up the running conversation centroid. The
#: compaction living-doc row is deliberately excluded: it is a derived summary,
#: not a turn the user or model actually sent.
_CENTROID_ROLES = frozenset({"user", "assistant"})

#: How many of the most recent transcript messages form the conversation
#: centroid. A short window tracks the *current* thread rather than the whole
#: session, and bounds the per-turn embedding work to the <100ms-class budget
#: (embeddings are cached by text, so warm turns add no inference).
_CENTROID_WINDOW = 6

#: A message with fewer than this many whitespace tokens is "terse" — too thin to
#: carry its own topic ("yes", "lol", "do that"). A terse message means the user
#: is actively in conversation (the model already holds the full thread in
#: context) and there is no topic to recall against, so the flashback is skipped
#: outright before any embedding work. ~8 tokens.
_TERSE_TOKEN_CEILING = 8

#: Curated-bundle caps (≤5 facts + ≤3 dated episode gists). These are
#: selection counts (how many memories surface), not text limits — each surfaced
#: gist is rendered in full.
_MAX_FACTS = 5
_MAX_EPISODES = 3

#: Episodes at this hierarchy level or above are super-episodes / era digests and
#: are preferred over raw leaves when choosing the ≤3 dated episodes.
_SUPER_LEVEL_FLOOR = 1

#: Over-fetch multiplier for the episode recall: pull this many times
#: ``_MAX_EPISODES`` candidates so the super-episode-preferred re-sort has leaves
#: AND supers to choose from before the final ≤3 clip, rather than clipping at
#: the retrieval layer and starving the preference.
_EPISODE_OVERFETCH_FACTOR = 3


class TurnZeroFlashback:

    def __init__(self, mp: "MessageProcessor") -> None:
        self._mp = mp

    # ── Public entry ──────────────────────────────────────────────────────────

    def seed(self) -> None:
        """Run the gates and, if they pass, inject the flashback.

        Never raises: a failure in any sub-step logs and leaves the turn
        un-seeded rather than aborting the user's message. The seed is a
        best-effort grounding aid, not a correctness dependency.
        """
        try:
            message = (self._mp.raw_input or "").strip()
            if self._is_terse(message):
                logger.debug(
                    "[FLASHBACK] terse message — user is mid-conversation, "
                    "recall skipped"
                )
                return
            if self._is_continuation(message):
                logger.debug("[FLASHBACK] continuation — flashback skipped")
                return
            block = self._build_flashback_block(message)
            self._record_seed(message, block)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FLASHBACK] seed failed (non-fatal): %s", exc)

    # ── Gates ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_terse(text: str) -> bool:
        """A message with fewer than the ceiling of whitespace tokens carries too
        little topic on its own and signals an active conversation — the
        flashback is skipped for it."""
        return len(text.split()) < _TERSE_TOKEN_CEILING

    def _is_continuation(self, message: str) -> bool:
        """True when *message* is close to the running conversation centroid.

        Session start (no centroid) is never a continuation — the flashback
        always fires. A failure in the embedding/centroid math is treated as
        "not a continuation" so the gate fails open to a flashback rather than
        silently starving the model of memory.
        """
        if not message:
            return False
        try:
            from services.embedding_service import get_embedding_service  # noqa: PLC0415

            centroid = self._conversation_centroid()
            if centroid is None:
                return False  # session start — fire the flashback
            emb_svc = get_embedding_service()
            msg_vec = emb_svc.generate_embedding_np(message, mp=self._mp)
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
        before the seed fires) excluded. An empty list means there is no prior
        conversation (session start).
        """
        from services.transcript_service import Transcript  # noqa: PLC0415

        # Fetch one extra row so dropping the current turn still leaves a full
        # window of prior context.
        rows = Transcript.get_recent(
            self._mp.config.channel, limit=_CENTROID_WINDOW + 1
        )
        current_uid = self._mp.uid
        return [
            cast("str", r.get("content") or "").strip()
            for r in rows
            if r.get("role") in _CENTROID_ROLES
            and cast("str", r.get("content") or "").strip()
            and (current_uid is None or r.get("id") != current_uid)
        ][:_CENTROID_WINDOW]

    def _conversation_centroid(self) -> object | None:
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
        return cast("object", mean / norm)

    @staticmethod
    def _cosine(a: object, b: object) -> float:
        """Cosine similarity of two vectors. Inputs from the embedding service are
        L2-normalised, but the centroid mean is re-normalised by the caller, so a
        plain dot product is the cosine."""
        import numpy as np  # noqa: PLC0415

        return float(np.dot(cast("list[float]", a), cast("list[float]", b)))

    # ── Retrieval + render ────────────────────────────────────────────────────

    def _build_flashback_block(self, query: str) -> str:
        """Assemble the curated flashback: ≤5 fact bullets + ≤3 dated episodes.

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
            if (text := cast("str", hit.get("text") or "").strip())
        ][:_MAX_FACTS]

    def _recall_episodes(self, query: str) -> list[dict[str, object]]:
        """Up to ``_MAX_EPISODES`` raw episodes for *query*, super-episodes first.

        Runs the shared episode recall with ``caller='seed'`` so the frozen
        ``memory_recall_log`` seed row + telemetry are written exactly as the
        retrieval rework left them. Recall is cross-channel (): a user
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
                0 if int(cast("int", ep.get("level") or 0)) >= _SUPER_LEVEL_FLOOR else 1,
                -float(cast("float", ep.get("composite_score") or 0.0)),
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
        """One episode as ``On <date>: <gist>``.

        The gist is rendered in full — NO truncation (); newlines are
        collapsed to keep the bundle one line per episode, but no characters are
        dropped. The date is the episode's local-time creation day; an
        unparseable / missing timestamp drops the date prefix rather than
        emitting a bogus one.
        """
        gist = (cast("str", ep.get("gist")) or "").strip().replace("\n", " ")
        if not gist:
            return ""
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
        return TimeFormatterService.local(cast("str", raw), fmt="%Y-%m-%d") or ""

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record_seed(self, query: str, block: str) -> None:
        """Record the framework seed row so the model sees the curated block.

        Writes a ``memory(action='recall', _auto=True)`` ``tool_calls`` row whose
        ``result`` is the curated block, so the seed recall is visible in the
        turn's act-trail. The ``_auto`` flag marks it as the framework seed; the
        result is the curated bundle, NOT recall JSON — that JSON contract belongs
        to the explicit, model-invoked recall only.
        """
        from services.act_trail import ActTrail  # noqa: PLC0415

        params = {"action": _SEED_ACTION, "query": query, "_auto": True}
        # Anchors to this turn's input row: _render_act_trail's
        # turn-keyed fetch and the cancel cleanup both reach it by joining
        # transcript on (channel, turn_id), so no turn column is stamped here.
        ActTrail().record(
            tool_name=_SEED_TOOL_NAME,
            params=params,
            result=block,
            transcript_id=self._mp.uid,
        )
