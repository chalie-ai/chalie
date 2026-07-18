"""PromptService — assembles the system/user/definition prompts for one turn
(§4.2 D8). System-prompt bodies live on each ProcessorConfig's ``system_prompt``
property; this service wraps them with runtime data on the two dynamic channels.

Channel dispatch (public methods take zero params — §2.4) is keyed off
:meth:`_channel` — ``self.mp.config.prompt_channel or self.mp.config.channel`` —
so a config whose ``channel`` is dynamic (EAMP's ``external-agent:{name}``) or
that reuses another channel's assembly (Discovery ⇒ ``user``) routes on its
declarative ``prompt_channel`` override. Every per-channel prompt body that used
to live on a frozen config side-car (its ``get_system_prompt`` /
``get_user_prompt`` / ``get_user_definition`` / ``get_image``) is relocated here
in Phase E — the configs are now pure declarative data. Each builder's docstring
names the config it was ported from.

Memory boundary (§3.11): every structured user-context read goes through a
sibling service — the FACTS vertical (``FactRow.traits``) for the traits lane and
``self.mp.behavioral_pattern_service`` for the pattern lane — never a raw
``data_graph`` read here. The behavioural-pattern confidence ranking + cap lives
on ``BehavioralPatternService.top_patterns``; this file only formats the rows the
service hands back.

Cross-turn / cross-channel reads (skill-building's trigger-turn tool trail,
thread-gist's opener rows) are the trigger turn's identity — a DIFFERENT
(channel, turn_id) than the one in flight — carried on ``self.mp._trigger_channel``
/ ``self.mp._trigger_turn_id`` (§4.3, added by F1). A service reads its own
models, so those go straight through ``ToolCall`` / ``Transcript`` classmethods.

``self.mp.post_compaction_continuation`` is read verbatim (same name as today's
``MessageProcessor``) to gate the post-compaction continuity banner; the banner's
user-query text comes straight off ``self.mp.raw_input`` — the new processor holds
the turn's input for its whole recursion, so the old ``continuation_user_query``
DB-refetch (a redundant read of the same value) is gone.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, cast

from configs.channels.dmn import DmnConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from models.behavioral_pattern import BehavioralPattern
from models.fact import FactRow
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.personality.personality_service import personality_service
from services.time_formatter_service import TimeFormatterService
from services.user_synthesis import UserSynthesis
from services.world_state import world_state

if TYPE_CHECKING:
    from typing import Protocol

    from controllers.message_processor import MessageProcessor

    class _FactExtractionConfig(Protocol):
        _gist: str
        _neighbours: list[object]

    class _ExternalAgentConfig(Protocol):
        _agent_name: str
        _project: str
        system_prompt: str

    class _SuperEpisodeConfig(Protocol):
        _sources: list[object]

logger = logging.getLogger(__name__)

_CHANNEL_USER = Channel.USER
_CHANNEL_USER_SUMMARY = Channel.USER_SUMMARY
_CHANNEL_PATTERN_MATCH = Channel.PATTERN_MATCH
_CHANNEL_SCHEDULE = Channel.SCHEDULE
_CHANNEL_FACT_EXTRACTION = Channel.FACT_EXTRACTION
_CHANNEL_SKILL_ASSOCIATION = Channel.SKILL_ASSOCIATION
_CHANNEL_SKILLS_BUILDING = Channel.SKILLS_BUILDING
_CHANNEL_THREAD_GIST = Channel.DELEGATE_THREAD_GIST
_CHANNEL_VISION = Channel.DELEGATE_VISION
_CHANNEL_WEB_BROWSE = Channel.DELEGATE_WEB_BROWSE
_CHANNEL_WEB_SEARCH = Channel.DELEGATE_WEB_SEARCH
_CHANNEL_PIM = Channel.DELEGATE_PIM
_CHANNEL_EXTERNAL_AGENT = "external_agent"
_CHANNEL_COMPACTION = Channel.COMPACTION
_CHANNEL_SUPER_EPISODE = Channel.SUPER_EPISODE_ENCODER
_CHANNEL_GEO_PATTERN = Channel.GEO_PATTERN
_CHANNEL_DMN = Channel.DMN
_CHANNEL_EPISODE_ENCODER = Channel.EPISODE_ENCODER

_USER_DEFINITION_FALLBACK = (
    "The user is a real human. Treat this conversation as peer-to-peer dialogue."
)
_CONTENT_FIELD_PLACEHOLDER = "{{provider_content_field_name}}"
_MISSING_TS_PLACEHOLDER = "????-??-?? ??:??"

#: Appended to the system prompt on any channel whose config sets
#: ``SUPPORTS_ASYNC`` — the SAME gate that exposes the ``async`` tool parameter —
#: so enabling async on a new ProcessorConfig surfaces this guidance with zero
#: extra wiring. Appended trailing (constant text) so the cached system prefix
#: stays byte-stable across turns.
_ASYNC_GUIDANCE = """

## Background tasks

Some tools accept an `async` flag. Set `async: true` to run a tool in the background: you get an immediate acknowledgement, the current turn ends, and the moment the tool finishes you are automatically invoked again with its result as a new turn — so you can keep talking to the user while the work runs.

Choose `async: true` when the user asks for something to happen "in the background" or "while" they do something else, or when a call is likely to be slow (web research, browsing, lengthy shell or file work) and the user should not have to wait. Call tools normally (synchronously) for quick results the user is actively waiting on."""

_MAX_TRAIT_ROWS = 200
_MAX_PATTERN_ROWS = 25


class PromptService:
    """Builds the three prompt pieces (system / user / definition) plus the
    vision image payload for the turn's channel, and the shared history/trail
    fragments they compose from."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    # ── public dispatch (channel-keyed, zero-param) ─────────────────────────

    def system_prompt(self) -> str:
        """The turn's system instruction block, with the background-tasks
        guidance appended on any channel that exposes the ``async`` tool flag
        (``SUPPORTS_ASYNC``) — the one place all system-prompt assembly lands."""
        base = self._system_prompt_body()
        if self.mp.config.SUPPORTS_ASYNC:
            return base + _ASYNC_GUIDANCE
        return base

    def _system_prompt_body(self) -> str:
        """The per-channel system block.

        Static channels carry their prompt as the ``system_prompt`` property on
        their ProcessorConfig (pure declarative data). The two dynamic channels
        — ``UserConfig`` (and its ``DiscoveryConfig`` subclass) and
        ``EAMPConfig`` — wrap that base literal with runtime data (voice line,
        provider content-field, resolved names) here. Branching is on the config
        class, never the channel string, so subclass routing is automatic."""
        config = self.mp.config
        if isinstance(config, UserConfig):
            return self._user_system_prompt()
        if isinstance(config, EAMPConfig):
            return self._external_agent_system_prompt()
        return config.system_prompt

    def user_definition(self) -> str:
        """``UserConfig.get_user_definition``: the short user synthesis via
        :class:`UserSynthesis`, or the peer-to-peer fallback."""
        return UserSynthesis.get(shorthand=True) or _USER_DEFINITION_FALLBACK

    def user_prompt(self) -> str:
        """The turn's user-message body, ported from each config's
        ``get_user_prompt``."""
        channel = self._channel()
        if channel == _CHANNEL_USER:
            return self._user_prompt()
        if channel == _CHANNEL_USER_SUMMARY:
            return self._user_summary_prompt()
        if channel == _CHANNEL_PATTERN_MATCH:
            return self._pattern_prompt()
        if channel == _CHANNEL_SCHEDULE:
            return self._schedule_prompt()
        if channel == _CHANNEL_FACT_EXTRACTION:
            return self._fact_extraction_prompt()
        if channel == _CHANNEL_SKILLS_BUILDING:
            return self._skill_suggestion_prompt()
        if channel == _CHANNEL_THREAD_GIST:
            return self._thread_gist_prompt()
        if channel == _CHANNEL_WEB_BROWSE:
            return self._web_browse_prompt()
        if channel == _CHANNEL_WEB_SEARCH:
            return self._web_search_prompt()
        if channel == _CHANNEL_PIM:
            return self._pim_prompt()
        if channel == _CHANNEL_EXTERNAL_AGENT:
            return self._external_agent_prompt()
        if channel == _CHANNEL_SUPER_EPISODE:
            return self._super_episode_prompt()
        if channel == _CHANNEL_GEO_PATTERN:
            return self._geo_pattern_prompt()
        if channel == _CHANNEL_DMN:
            return self._dmn_prompt()
        if channel == _CHANNEL_EPISODE_ENCODER:
            return self._episode_encoder_prompt()
        # skill_association, vision and compaction pass the raw input straight
        # through.
        if channel in (_CHANNEL_SKILL_ASSOCIATION, _CHANNEL_VISION, _CHANNEL_COMPACTION):
            return self.mp.raw_input
        logger.warning("[PromptService] user_prompt: unhandled channel=%s", channel)
        return ""

    def image(self) -> dict[str, str] | None:
        """``VisionConfig.get_image``: the vision turn's attached image as a
        base64 payload, read from ``self.mp.metadata`` (keys ``image_path`` /
        ``mime_type``). ``None`` when no image is attached — every non-vision
        channel and a vision turn without a path."""
        meta = self.mp.metadata or {}
        path = meta.get("image_path")
        if not path:
            return None
        with open(cast("str", path), "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        return {"data": data, "mime_type": cast("str", meta.get("mime_type")) or "image/png"}

    # ── shared assembly fragments ────────────────────────────────────────────

    def previous_messages(self, drop_oldest: int = 0) -> str:
        """The ``## Previous Messages`` block body (no header): this turn's
        history view (``self.mp.transcript_service.read()``), each row formatted
        ``[local-ts] Role: content``. ``drop_oldest`` lets a caller (e.g. the
        history compactor, shrinking its own input) drop the N oldest rows — a
        genuine external count, not mp-reachable."""
        rows = self.mp.transcript_service.read()[drop_oldest:]
        if not rows:
            return ""
        lines: list[str] = []
        for row in rows:
            fields = row.to_dict()
            ts = self._format_ts(cast("str | None", fields.get("created_at")))
            raw_role = cast("str", fields.get("role") or "unknown")
            role_label = "Assistant" if raw_role == "assistant" else raw_role
            content = cast("str", fields.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {role_label}: {content}")
        return "\n".join(lines)

    def act_trail(self) -> str:
        """The current exchange's tool-call trail, rendered ``[tool] params →
        result`` one call per line. Scoped to this MP recursion instance
        (``by_exchange``), so a reply or async re-entry sharing the turn sees
        only its own calls — a prior exchange's raw trail lives on in the DB but
        leaves context once that exchange synthesises (its synthesis carries
        forward via Previous Messages). A mid-exchange compaction resets the
        visible trail: only calls after the most recent ``chat_history_compactor``
        marker render, mirroring the pre-rewrite ``_render_act_trail``."""
        calls = self.mp.tool_call_service.by_exchange()
        last_compaction = max(
            (cast("int", call.id) for call in calls if call.tool_name == "chat_history_compactor"),
            default=0,
        )
        lines = [
            f"[{call.tool_name}] {call.params} → {call.result}"
            for call in calls
            if call.tool_name != "chat_history_compactor" and cast("int", call.id) > last_compaction
        ]
        return "\n".join(lines)

    # ── dispatch + safe-fragment helpers ─────────────────────────────────────

    def _channel(self) -> str:
        """The dispatch key: the config's ``prompt_channel`` override, or its
        ``channel`` when unset (§2.5 — the one declarative routing field)."""
        return self.mp.config.prompt_channel or self.mp.config.channel

    def _prev(self) -> str:
        """:meth:`previous_messages`, guarded — a history-render hiccup must
        never crash the turn (the pre-rewrite builders each wrapped this)."""
        try:
            return self.previous_messages()
        except Exception as exc:  # noqa: BLE001 — a history-render hiccup must not crash the turn
            logger.debug("[PromptService] previous_messages failed: %s", exc)
            return ""

    def _trail(self) -> str:
        """:meth:`act_trail`, guarded — the exception-safety the delegate
        channels' ``render_trail`` wrapper used to provide."""
        try:
            return self.act_trail()
        except Exception as exc:  # noqa: BLE001 — a trail-render hiccup must not crash the turn
            logger.warning("[PromptService] act_trail failed: %s", exc, exc_info=True)
            return ""

    # ── UserConfig (channel="user") ──────────────────────────────────────────

    def _user_system_prompt(self) -> str:
        """``UserConfig``: the voice line (cache-warm prefix) over the config's
        ``system_prompt`` base literal, with the provider content-field
        placeholder substituted. Runs for ``UserConfig`` and its
        ``DiscoveryConfig`` subclass."""
        try:
            voice_line = f"When responding; {personality_service.get_voice()}"
            prompt = f"{voice_line}\n\n{self.mp.config.system_prompt}"
            return self._substitute_content_field(prompt)
        except Exception as exc:  # noqa: BLE001 — a prompt build hiccup must not crash the turn
            logger.warning("[PromptService] user system prompt build failed: %s", exc)
            return ""

    def _user_prompt(self) -> str:
        """``UserConfig.get_user_prompt``: user_def, World State, Previous
        Messages, blank, (post-compaction banner), input line, act trail —
        same section order as the pre-rewrite assembly."""
        parts: list[str] = []

        user_def = self.user_definition()
        if user_def:
            parts.append(user_def)

        try:
            rendered_ws = world_state.render()
        except Exception as exc:  # noqa: BLE001 — off-spine telemetry render must not crash the turn
            logger.debug("[PromptService] world_state.render failed: %s", exc)
            rendered_ws = ""
        if rendered_ws:
            parts.append(rendered_ws)

        prev = self._prev()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")

        parts.append("")

        if self.mp.post_compaction_continuation:
            query = self.mp.raw_input
            parts.append(
                "You are continuing after a mid-turn compaction. "
                f"The user query was: {query}. "
                "Read the Checkpoint section above to recover what you were "
                "working on, and use the review_transcript tool to read the "
                "previous turns of this conversation."
            )

        parts.append(f"user: {self.mp.raw_input}")

        trail = self._trail()
        if trail:
            parts.append(trail)

        return "\n".join(parts)

    def _substitute_content_field(self, body: str) -> str:
        """``substitute_provider_content_field``: replace the
        ``{{provider_content_field_name}}`` placeholder with the selected
        provider's content-field label. Best-effort — placeholder absent or
        resolution failure leaves ``body`` unchanged."""
        if _CONTENT_FIELD_PLACEHOLDER not in body:
            return body
        try:
            label = self.mp.provider_service.selected_provider().CONTENT_FIELD_LABEL
        except Exception:  # noqa: BLE001 — a missing/misconfigured provider must not crash the turn
            label = None
        return body.replace(_CONTENT_FIELD_PLACEHOLDER, label) if label else body

    # ── UserSummaryConfig (channel="user_summary") ───────────────────────────

    def _user_summary_prompt(self) -> str:
        """``UserSummaryConfig.get_user_prompt``: traits facts section, plus the
        active-patterns section when any exist."""
        facts_section = self._traits_block()
        patterns_section = self._user_summary_patterns_block()
        if not patterns_section:
            return facts_section
        return f"{facts_section}\n\n{patterns_section}"

    def _traits_block(self) -> str:
        """Section 1 of ``UserSummaryConfig.get_user_prompt``: up to
        ``_MAX_TRAIT_ROWS`` live ``user_specific`` facts, most-reinforced first,
        via ``FactRow.traits()``."""
        rows = FactRow.traits().get()[:_MAX_TRAIT_ROWS]
        lines = [f"{row.key}: {row.value}" for row in rows if row.key and row.value]
        if not lines:
            return "Facts:\n(no facts available)"
        return "Facts:\n" + "\n".join(lines)

    def _user_summary_patterns_block(self) -> str:
        """Section 2 of ``UserSummaryConfig.get_user_prompt``: up to
        ``_MAX_PATTERN_ROWS`` active behavioural patterns, most-recently-confirmed
        first, via ``self.mp.behavioral_pattern_service.patterns()``."""
        lines: list[str] = []
        for row in self.mp.behavioral_pattern_service.patterns()[:_MAX_PATTERN_ROWS]:
            content = BehavioralPattern.parse(row.value)
            if content is not None:
                lines.append(BehavioralPattern.render_line(content, include_last_seen=True))
        if not lines:
            return ""
        return "## Behavioural patterns (frequency, last seen)\n" + "\n".join(
            f"- {line}" for line in lines
        )

    # ── PatternConfig (channel="pattern_match") ──────────────────────────────

    def _pattern_prompt(self) -> str:
        """``PatternConfig.get_user_prompt``: the pattern window's transcripts
        (``self.mp.transcript_service.window()`` — the id-bounded ``user`` read)
        + the existing-patterns block + this turn's act trail."""
        rows = self.mp.transcript_service.window()
        transcript_block = (
            "\n".join(
                f"[id={row.id} | {cast('str', row.to_dict()['role'])} | "
                f"{cast('str', row.to_dict()['created_at'])}] "
                f"{cast('str', row.to_dict()['content'])}"
                for row in rows
            )
            if rows
            else "(no transcripts in window)"
        )
        parts = [f"Existing patterns:\n{self._pattern_existing_patterns_block()}", transcript_block]
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def _pattern_existing_patterns_block(self) -> str:
        """The top behavioural patterns by confidence as a ``name -> summary``
        JSON object — the pattern/geo channels' existing-patterns block. The
        confidence ranking + cap live on ``BehavioralPatternService.top_patterns``;
        this builder only renders the result as prompt text."""
        patterns = self.mp.behavioral_pattern_service.top_patterns()
        return json.dumps(patterns, indent=2) if patterns else "(none yet)"

    # ── ScheduledConfig (channel="schedule") ─────────────────────────────────

    def _schedule_prompt(self) -> str:
        """``ScheduledConfig.get_user_prompt``: previous messages, the scheduled
        task, then this turn's act trail — joined by blank lines."""
        parts: list[str] = []
        prev = self._prev()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")
        parts.append(f"Scheduled task:\n{self.mp.raw_input}")
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    # ── FactExtractionConfig (channel="fact_extraction") ─────────────────────

    def _fact_extraction_prompt(self) -> str:
        """``FactExtractionConfig.get_user_prompt``: the episode gist and the
        pre-fetched neighbour facts captured on the config (``_gist`` /
        ``_neighbours`` — per-episode frozen data, not mp-reachable)."""
        config = cast("_FactExtractionConfig", self.mp.config)
        if config._neighbours:
            known = "\n".join(
                f"- key={cast('dict[str, object]', n).get('key')!r} "
                f"value={cast('dict[str, object]', n).get('value')!r}"
                for n in config._neighbours
            )
        else:
            known = "(no similar facts on record)"
        return f"Episode:\n{config._gist}\n\nMost similar known facts:\n{known}"

    # ── SkillSuggestionConfig (channel="skills_building") ────────────────────

    def _skill_suggestion_prompt(self) -> str:
        """``SkillSuggestionConfig.get_user_prompt``: the trigger turn's original
        request and its FULL tool-call trail — every ``ToolCall.by_turn`` row
        rendered ``[tool] params → result`` with NO compaction filtering (this
        channel judges the raw workflow, unlike :meth:`act_trail`)."""
        trigger_channel = self.mp._trigger_channel
        trigger_turn_id = self.mp._trigger_turn_id
        rows: list[ToolCall] = []
        if trigger_channel is not None and trigger_turn_id is not None:
            rows = ToolCall.by_turn(trigger_channel, trigger_turn_id)
        parts = [
            f"## Original User Request\n{self.mp.raw_input or ''}",
            f"\n## Completed ACT Loop Trail ({len(rows)} iterations)",
        ]
        parts.extend(
            f"[{row.tool_name}] {row.params or '{}'} → {row.result or ''}" for row in rows
        )
        return "\n".join(parts)

    # ── ThreadGistConfig (channel="delegate:thread_gist") ────────────────────

    def _thread_gist_prompt(self) -> str:
        """``ThreadGistConfig.get_user_prompt``: the text to label. A caller that
        already holds it (a schedule's prompt — a schedule has no transcript to
        read) passes it as ``raw_input`` and it IS the prompt; otherwise it is
        read from the trigger thread — its opener and the first
        row beyond its settle0, non-assistant only, each rendered
        ``[local-ts] content``. Empty when no trigger turn or no rows."""
        if self.mp.raw_input:
            return f"# User Message Prompt\n## User Messages\n{self.mp.raw_input}"
        trigger_channel = self.mp._trigger_channel
        trigger_turn_id = self.mp._trigger_turn_id
        if trigger_channel is None or trigger_turn_id is None:
            return ""
        rows = [
            r for r in Transcript.by_turn(trigger_channel, trigger_turn_id)
            if r.get("role") != "assistant"
        ]
        if not rows:
            return ""
        settle = Transcript.settle0(trigger_channel, trigger_turn_id)
        beyond = next(
            (r for r in rows if settle is not None and cast("int", r.get("id")) > settle),
            None,
        )
        picked = [rows[0]] + ([beyond] if beyond is not None and beyond is not rows[0] else [])
        lines = ["# User Message Prompt", "## User Messages"]
        for r in picked:
            ts = format_date(cast("str | None", r.get("created_at")), CHAT_TIMESTAMP_FMT, for_ui=True) or ""
            content = cast("str", r.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {content}")
        return "\n".join(lines)

    # ── WebBrowseConfig (channel="delegate:web_browse") ──────────────────────

    def _web_browse_prompt(self) -> str:
        """``WebBrowseConfig.get_user_prompt``: the browsing goal, the
        screenshots captured this run (the browser session's ledger, keyed on
        the turn uid), then this turn's act trail."""
        from tools.browser.session import screenshot_ledger  # noqa: PLC0415

        parts = [f"Browsing goal:\n{self.mp.raw_input}"]
        shots = screenshot_ledger(self.mp.uid or 0)
        if shots:
            lines = "\n".join(f"- doc_id={doc_id} ({url})" for doc_id, url in shots)
            parts.append(f"Screenshots captured this run:\n{lines}")
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    # ── WebSearchConfig (channel="delegate:web_search") ──────────────────────

    def _web_search_prompt(self) -> str:
        """``WebSearchConfig.get_user_prompt``: the research query then this
        turn's act trail."""
        parts = [f"Research query:\n{self.mp.raw_input}"]
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    # ── PimConfig (channel="delegate:pim") ───────────────────────────────────

    def _pim_prompt(self) -> str:
        """``PimConfig.get_user_prompt``: the personal-information instruction
        then this turn's act trail."""
        parts = [f"Instruction:\n{self.mp.raw_input}"]
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    # ── EAMPConfig (channel="external-agent:{name}", prompt_channel="external_agent") ──

    def _external_agent_definition(self) -> str:
        """``EAMPConfig.get_user_definition``: the static agent identity string
        from the config's captured ``_agent_name`` / ``_project``."""
        config = cast("_ExternalAgentConfig", self.mp.config)
        return (
            f"The user is {config._agent_name}, an external agent. "
            f"This conversation is about: {config._project}."
        )

    def _external_agent_system_prompt(self) -> str:
        """``EAMPConfig.get_system_prompt``: the external-agent producer body
        with the provider content-field placeholder substituted, the user's
        first name resolved from data_graph, and the agent/project placeholders
        filled — prefixed with the agent identity definition. Any failure yields
        ``""`` (the pre-rewrite try/except contract)."""
        config = cast("_ExternalAgentConfig", self.mp.config)
        try:
            body = self._substitute_content_field(config.system_prompt)
            summary = UserSynthesis.get(shorthand=True)
            user_name = summary.split()[0] if summary and summary.split() else "the user"
            body = (
                body
                .replace("{user_name}", user_name)
                .replace("{agent_name}", config._agent_name)
                .replace("{project_or_task_name}", config._project)
            )
            return f"{self._external_agent_definition()}\n\n{body}"
        except Exception as exc:  # noqa: BLE001 — a prompt build hiccup must not crash the turn
            logger.warning("[PromptService] external_agent system prompt build failed: %s", exc)
            return ""

    def _external_agent_prompt(self) -> str:
        """``EAMPConfig.get_user_prompt``: previous messages, a blank line, the
        input line, then this turn's act trail — joined by single newlines (the
        input line comes BEFORE the trail, the pre-rewrite ordering)."""
        parts: list[str] = []
        prev = self._prev()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")
        parts.append("")
        parts.append(f"user: {self.mp.raw_input}")
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n".join(parts)

    # ── SuperEpisodeConfig (channel="super_episode_encoder") ─────────────────

    def _super_episode_prompt(self) -> str:
        """``SuperEpisodeConfig.get_user_prompt``: the cluster's source-episode
        gists alone (``_sources``, per-cluster frozen data captured on the config
        at construction, not mp-reachable — same pattern as fact-extraction's
        payload). Every level distils its child gists; raw transcript turns are
        never re-hydrated into the prompt, so a super-episode always contracts
        the level beneath it."""
        config = cast("_SuperEpisodeConfig", self.mp.config)
        src = "\n\n".join(
            f"[{cast('dict[str, object]', e)['id']}] {cast('dict[str, object]', e)['gist']}"
            for e in config._sources
        )
        return f"Source episodes:\n\n{src}"

    # ── EpisodeEncoderConfig (channel="episode_encoder") ─────────────────────

    def _episode_encoder_prompt(self) -> str:
        """``EpisodeEncoderConfig.get_user_prompt``: the formatted transcript
        window, then (when present) the episodes referenced during those turns —
        both computed by ``EpisodicService`` and carried on ``self.mp.metadata``
        (keys ``window`` / ``referenced``), the sanctioned per-turn payload channel
        (§2.4), replacing the pre-rewrite ``mp._window`` / ``mp._referenced``
        instance-attribute injection."""
        meta = self.mp.metadata or {}
        window = cast("str", meta.get("window") or "")
        referenced = cast("str", meta.get("referenced") or "")
        parts = [
            "Transcript window — each line is `[id] (timestamp) role: content`:",
            "",
            window,
        ]
        if referenced:
            parts.extend([
                "",
                "Episodes referenced during these turns (candidates for update / delete):",
                "",
                referenced,
            ])
        return "\n".join(parts)

    # ── GeoConfig (channel="geo_pattern") ────────────────────────────────────

    def _geo_pattern_prompt(self) -> str:
        """``GeoConfig.get_user_prompt``: the existing-patterns block, the
        location-tagged transcript window, then this turn's act trail."""
        parts = [
            f"Existing patterns:\n{self._pattern_existing_patterns_block()}",
            self._geo_transcript_block(),
        ]
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def _geo_transcript_block(self) -> str:
        """The geo window's location-tagged ``user`` rows
        (``self.mp.transcript_service.location_window()``) rendered
        ``[id | role | ts | lat,lon place] content``. Guarded — a window-read
        hiccup must not crash the turn (the pre-rewrite builder wrapped this)."""
        try:
            rows = self.mp.transcript_service.location_window()
        except Exception as exc:  # noqa: BLE001 — a window-read hiccup must not crash the turn
            logger.warning("[PromptService] geo transcript block failed: %s", exc)
            return "(transcript fetch failed)"
        if not rows:
            return "(no location-tagged transcripts in window)"
        lines: list[str] = []
        for row in rows:
            fields = row.to_dict()
            lat, lon = fields.get("location_lat"), fields.get("location_lon")
            place = cast("str", fields.get("location_name") or "")
            location = f"{lat},{lon}" if not place else f"{lat},{lon} {place}"
            lines.append(
                f"[id={fields.get('id')} | {cast('str', fields.get('role'))} | "
                f"{cast('str', fields.get('created_at'))} | {location}] "
                f"{cast('str', fields.get('content'))}"
            )
        return "\n".join(lines)

    # ── DmnConfig (channel="dmn") ────────────────────────────────────────────

    def _dmn_prompt(self) -> str:
        """``DmnConfig.get_user_prompt``: the user synthesis (long summary,
        falling back to the short one), the recent-salient-episodes reflection
        context, then this turn's act trail."""
        parts: list[str] = []
        synthesis = UserSynthesis.get()
        if synthesis:
            parts.append(f"## About the User\n{synthesis}")
        episodes = self._dmn_episodes_block()
        if episodes:
            parts.append(f"## Episodes\n{episodes}")
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def _dmn_episodes_block(self) -> str:
        """DMN's recent-salient-episode reflection context — the episodic read
        at prompt-assembly time. Sourced from ``DmnConfig``'s own DB-reaching
        static (marked ``@todo: Refactor`` there): one home for the DMN reads
        until episodic prompt-context is folded onto the spine as a service."""
        try:
            return DmnConfig.recent_salient_user_episodes()
        except Exception as exc:  # noqa: BLE001 — a reflection-context read must not crash the turn
            logger.debug("[PromptService] dmn episodes read failed: %s", exc)
            return ""

    # ── formatting helpers ───────────────────────────────────────────────────

    def _format_ts(self, raw: str | None) -> str:
        """``_format_ts``: one transcript row's ``created_at`` in the user's local
        timezone, or the placeholder on a missing/unparseable value."""
        if raw is None or not raw.strip():
            logger.warning("[PromptService] missing created_at on transcript row")
            return _MISSING_TS_PLACEHOLDER
        formatted = TimeFormatterService.local(raw)
        if formatted is None:
            logger.warning("[PromptService] unparseable created_at=%r", raw)
            return _MISSING_TS_PLACEHOLDER
        return formatted
