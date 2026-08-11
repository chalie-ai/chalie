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
sibling service — ``self.mp.behavioral_pattern_service`` for the pattern lane —
never a raw ``data_graph`` read here. The behavioural-pattern confidence ranking
+ cap lives on ``BehavioralPatternService.top_patterns``; this file only formats
the rows the service hands back.

Cross-turn / cross-channel reads (skill-building's trigger-turn tool trail,
thread-gist's opener rows) are the trigger turn's identity — a DIFFERENT
(channel, turn_id) than the one in flight — carried on ``self.mp._trigger_channel``
/ ``self.mp._trigger_turn_id`` (§4.3, added by F1). A service reads its own
models, so those go straight through ``ToolCall`` / ``Transcript`` classmethods.

``self.mp.turn_handover`` is read by :meth:`_handover` to gate the post-compaction
continuity banner; the banner carries the first-person hand-over summary produced
by :class:`~services.turn_handover_service.TurnHandoverService` immediately before
the compactor erases the act trail.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, cast

from abilities.recall import Recall
from configs.channels.external_agent import EAMPConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from exceptions import UnroutedPromptChannel
from models.behavioral_pattern import BehavioralPattern
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.transcript_thinking import TranscriptThinking
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.markup import PROMPT_TAGS
from services.personality.personality_service import personality_service
from services.time_formatter_service import TimeFormatterService
from services.user_synthesis import UserSynthesis
from services.world_state import world_state

if TYPE_CHECKING:
    from typing import Protocol

    from controllers.message_processor import MessageProcessor

    class _ExternalAgentConfig(Protocol):
        _agent_name: str
        _project: str
        system_prompt: str

logger = logging.getLogger(__name__)

_CHANNEL_USER = Channel.USER
_CHANNEL_USER_SUMMARY = Channel.USER_SUMMARY
_CHANNEL_PATTERN_MATCH = Channel.PATTERN_MATCH
_CHANNEL_SCHEDULE = Channel.SCHEDULE
_CHANNEL_SKILL_ASSOCIATION = Channel.SKILL_ASSOCIATION
_CHANNEL_SKILLS_BUILDING = Channel.SKILLS_BUILDING
_CHANNEL_THREAD_GIST = Channel.DELEGATE_THREAD_GIST
_CHANNEL_VISION = Channel.DELEGATE_VISION
_CHANNEL_WEB_BROWSE = Channel.DELEGATE_WEB_BROWSE
_CHANNEL_WEB_SEARCH = Channel.DELEGATE_WEB_SEARCH
_CHANNEL_PIM = Channel.DELEGATE_PIM
_CHANNEL_CODE_AGENT = Channel.DELEGATE_CODE_AGENT
_CHANNEL_EXTERNAL_AGENT = "external_agent"
_CHANNEL_COMPACTION = Channel.COMPACTION
_CHANNEL_GEO_PATTERN = Channel.GEO_PATTERN
_CHANNEL_DMN = Channel.DMN

_USER_DEFINITION_FALLBACK = (
    "The user is a real human. Treat this conversation as peer-to-peer dialogue."
)
_CONTENT_FIELD_PLACEHOLDER = "{{provider_content_field_name}}"
_MISSING_TS_PLACEHOLDER = "????-??-?? ??:??"

_HANDOVER_FRAME = (
    "You hit your context limit mid-task and are continuing the same task — "
    "this is not a new conversation. Below is the hand-over you produced. "
    "Trust it: do not re-verify or repeat completed work; continue from the "
    "pending list.\n\n{handover}"
)

#: The tag list the model is given, rendered from the sanitiser's own authority
#: (``markup.PROMPT_TAGS``) so widening or narrowing the contract rewrites this
#: instruction in the same edit. Built separately rather than inlined into an
#: f-string: the literal below carries ``{{provider_content_field_name}}``, which
#: an f-string would collapse to single braces before ``_substitute_content_field``
#: ever sees it.
_TAG_LIST = ", ".join(f"<{tag}>" for tag in PROMPT_TAGS)

#: The single source of the HTML output contract, shared by every channel whose
#: output is rendered to a human, and gated by ``ProcessorConfig.RENDERS_HTML``.
_RESPONSE_FORMAT = (
    """

────────────────────────────────

## Response format

In the {{provider_content_field_name}} field (what the user sees) format your response as HTML.
Specifically only use the following tags: """
    + _TAG_LIST
    + """
NEVER use markdown syntax. Use <b> not **, use <i> not _, use <h1> not #, use <ul><li> not - or *. No backtick fences. HTML tags only.
Avoid using table structures to represent data. If you do need to use tables, output in html only NEVER as markdown and keep column count under 4.

────────────────────────────────"""
)

#: Appended to the system prompt on any channel whose config sets
#: ``SUPPORTS_ASYNC`` — the SAME gate that exposes the ``async`` tool parameter —
#: so enabling async on a new ProcessorConfig surfaces this guidance with zero
#: extra wiring. Appended trailing (constant text) so the cached system prefix
#: stays byte-stable across turns.
_ASYNC_GUIDANCE = """

## Background tasks

Some tools accept an `async` flag. Set `async: true` to run a tool in the background: you get an immediate acknowledgement, the current turn ends, and the moment the tool finishes you are automatically invoked again with its result as a new turn — so you can keep talking to the user while the work runs.

Choose `async: true` when the user asks for something to happen "in the background" or "while" they do something else, or when a call is likely to be slow (web research, browsing, lengthy shell or file work) and the user should not have to wait. Call tools normally (synchronously) for quick results the user is actively waiting on."""

_MAX_PATTERN_ROWS = 25


class PromptService:
    """Builds the three prompt pieces (system / user / definition) plus the
    vision image payload for the turn's channel, and the shared history/trail
    fragments they compose from."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    # ── public dispatch (channel-keyed, zero-param) ─────────────────────────

    def system_prompt(self) -> str:
        """The turn's system instruction block: the per-channel body, plus the
        shared response-format contract on any channel that renders to a human
        (``RENDERS_HTML``) and the background-tasks guidance on any channel that
        exposes the ``async`` tool flag (``SUPPORTS_ASYNC``) — the one place all
        system-prompt assembly and placeholder substitution lands."""
        base = self._system_prompt_body()
        if self.mp.config.RENDERS_HTML:
            base += _RESPONSE_FORMAT
        if self.mp.config.SUPPORTS_ASYNC:
            base += _ASYNC_GUIDANCE
        return self._substitute_content_field(base)

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
        ``get_user_prompt``.

        A channel with no arm below raises :class:`UnroutedPromptChannel` rather
        than returning an empty body. The sole caller is ``_build_messages`` on
        the drive thread, so the raise stamps the turn CRASHED naming the
        unrouted channel — the wiring error — instead of sending a contentless
        message and letting the provider decide whether to answer nonsense or
        reject it (Case V)."""
        channel = self._channel()
        if channel == _CHANNEL_USER:
            return self._user_prompt()
        if channel == _CHANNEL_USER_SUMMARY:
            return self._user_summary_prompt()
        if channel == _CHANNEL_PATTERN_MATCH:
            return self._pattern_prompt()
        if channel == _CHANNEL_SCHEDULE:
            return self._schedule_prompt()
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
        if channel == _CHANNEL_CODE_AGENT:
            return self._code_agent_prompt()
        if channel == _CHANNEL_EXTERNAL_AGENT:
            return self._external_agent_prompt()
        if channel == _CHANNEL_GEO_PATTERN:
            return self._geo_pattern_prompt()
        if channel == _CHANNEL_DMN:
            return self._dmn_prompt()
        # skill_association, vision and compaction pass the raw input straight
        # through.
        if channel in (_CHANNEL_SKILL_ASSOCIATION, _CHANNEL_VISION, _CHANNEL_COMPACTION):
            return self.mp.raw_input
        raise UnroutedPromptChannel(channel)

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
        marker render, mirroring the pre-rewrite ``_render_act_trail``.

        The turn-0 auto-seed recall (``"_auto": true`` in its params) renders as
        a ``[background_memory]`` context block instead of a tool-call row — a
        seed presented as a call teaches the model that every turn opens with a
        memory invocation. Same envelope body, relabeled wrapper; the
        ``tool_calls`` row is untouched (episodic dedup and telemetry read it
        there).

        Interim-prose interleave: before the first surviving call anchored to
        each assistant row of the current exchange, ``act_trail`` emits a line
        ``[interim_response] <row content with newlines flattened to spaces>``.
        Calls anchored directly to the input row (``transcript_id == mp.uid``)
        render without an interim prefix; an assistant row anchoring no calls
        (the final synthesis) never triggers an interim line; assistant rows
        from a prior exchange never surface — ``exchange_assistant_rows``
        floors at ``mp.uid`` so only this exchange's steps are in the lookup.
        A chat-history compaction anchors to one assistant row and becomes the
        interim cutoff: that row's interim prose AND its ordinary calls are
        dropped (``call.id <= last_compaction``), while rows written after the
        compactor marker still render with their interim prose.

        Thinking-trace interleave: before the first surviving call (or interim
        prose) anchored to each transcript id of the current exchange,
        ``act_trail`` emits that anchor's stored thinking traces (in row order),
        each wrapped ``[thinking]{trace}[end_thinking]``. Generation order was
        think → prose → tools, so the trail reads that way. Traces anchored to
        a row with no surviving call (an empty-completion steer at the exchange
        input row, or a trace-only exchange) are emitted before the call trail —
        chronologically they precede every call-bearing anchor — so no step's
        trace is dropped and the prompt-level guard still renders a trace-only
        trail."""
        calls = self.mp.tool_call_service.by_exchange()
        last_compaction = max(
            (cast("int", call.id) for call in calls if call.tool_name == "chat_history_compactor"),
            default=0,
        )
        cutoff = max(
            (call.transcript_id for call in calls if call.tool_name == "chat_history_compactor"),
            default=0,
        )
        interim_rows = self.mp.transcript_service.exchange_assistant_rows()
        interim: dict[int, str] = {}
        for row in interim_rows:
            content = cast("str", row.to_dict().get("content") or "")
            if not content or not content.strip():
                continue
            interim[cast("int", row.id)] = content.replace("\n", " ").strip()
        thinking_rows = TranscriptThinking.by_exchange(self.mp.channel, self.mp.turn_id, self.mp.uid)
        thinking: dict[int, list[str]] = {}
        for think_row in thinking_rows:
            tid = think_row.transcript_id
            if tid <= cutoff:
                continue
            trace = think_row.thinking_trace
            if not trace or not trace.strip():
                continue
            thinking.setdefault(tid, []).append(trace)
        emitted: set[int] = set()
        parts: list[str] = []
        # Traces anchored to rows that never got a call (an empty-completion
        # steer at the exchange input row) precede every call-bearing anchor
        # chronologically — emit them first so no step's trace is dropped.
        called_tids = {call.transcript_id for call in calls}
        for tid in sorted(t for t in thinking if t not in called_tids):
            for trace in thinking[tid]:
                parts.append(f"[thinking]{trace}[end_thinking]")
            emitted.add(tid)
        for call in calls:
            if call.tool_name == "chat_history_compactor" or cast("int", call.id) <= last_compaction:
                continue
            result = call.result
            tid = call.transcript_id
            if tid not in emitted and tid > cutoff:
                if tid in thinking:
                    for trace in thinking[tid]:
                        parts.append(f"[thinking]{trace}[end_thinking]")
                if tid in interim:
                    parts.append(f"[interim_response] {interim[tid]}")
                emitted.add(tid)
            if call.tool_name == Recall.NAME and '"_auto": true' in call.params:
                body = result.split("\n", 1)[1] if "\n" in result else result
                parts.append(
                    f"[background_memory]\n{body}".replace("[end:recall]", "[end:background_memory]")
                )
                continue
            parts.append(f"[{call.tool_name}] {call.params} → {result}")
        return "\n".join(parts)

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

    def _handover(self) -> str:
        """The post-compaction continuity banner, or ``""`` when there is no
        hand-over stored on this turn."""
        if not self.mp.turn_handover:
            return ""
        return _HANDOVER_FRAME.format(handover=self.mp.turn_handover)

    def _world(self) -> str:
        """``world_state.render()``, guarded — the turn's telemetry block, whose
        ``local_time`` line is the ONLY date anchor any channel receives. Off-spine
        telemetry, so a render hiccup must never crash the turn."""
        try:
            return world_state.render()
        except Exception as exc:  # noqa: BLE001 — off-spine telemetry render must not crash the turn
            logger.debug("[PromptService] world_state.render failed: %s", exc)
            return ""

    # ── UserConfig (channel="user") ──────────────────────────────────────────

    def _user_system_prompt(self) -> str:
        """``UserConfig``: the voice line (cache-warm prefix) over the config's
        ``system_prompt`` base literal. Runs for ``UserConfig`` and its
        ``DiscoveryConfig`` subclass."""
        try:
            voice_line = f"When responding; {personality_service.get_voice()}"
            prompt = f"{voice_line}\n\n{self.mp.config.system_prompt}"
            return prompt
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

        rendered_ws = self._world()
        if rendered_ws:
            parts.append(rendered_ws)

        prev = self._prev()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")

        parts.append("")

        handover = self._handover()
        if handover:
            parts.append(handover)

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
        """``UserSummaryConfig.get_user_prompt``: the active-patterns section
        when any exist."""
        return self._user_summary_patterns_block()

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
        handover = self._handover()
        if handover:
            parts.append(handover)
        parts.append(f"Scheduled task:\n{self.mp.raw_input}")
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

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
        """``ThreadGistConfig.get_user_prompt``: the trigger thread's opener and
        the first row beyond its settle0, non-assistant only, each rendered
        ``[local-ts] content``. Empty when no trigger turn or no rows."""
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
        """``WebBrowseConfig.get_user_prompt``: the browsing goal then this
        turn's act trail."""
        parts = [f"Browsing goal:\n{self.mp.raw_input}"]
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

    # ── CodeAgentConfig (channel="delegate:code_agent") ──────────────────────

    def _code_agent_prompt(self) -> str:
        """``CodeAgentConfig.get_user_prompt``: World State, the coding task,
        then this turn's act trail.

        ``suppress_history=True`` makes this string the delegate's ENTIRE input,
        so World State rides along deliberately: it carries the only date anchor
        any channel receives, and a coding agent that cannot date its own work
        writes wrong dates into the files it creates. Kept as its own builder
        rather than remapped onto the user channel through ``prompt_channel`` —
        a task is a hand-off, not a user utterance, and that assembly would also
        inject the user-identity synthesis and a post-compaction banner naming a
        tool this channel does not pin."""
        parts: list[str] = []
        rendered_ws = self._world()
        if rendered_ws:
            parts.append(rendered_ws)
        parts.append(f"Task:\n{self.mp.raw_input}")
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
        with the user's first name resolved from data_graph and the
        agent/project placeholders filled — prefixed with the agent identity
        definition. Any failure yields ``""`` (the pre-rewrite try/except
        contract)."""
        config = cast("_ExternalAgentConfig", self.mp.config)
        try:
            body = config.system_prompt
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
        trail = self._trail()
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

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
