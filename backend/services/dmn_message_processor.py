"""
DMNMessageProcessor — Background DMN execution driven by SubconsciousWorker step 5.

MessageProcessor v2 subclass. Hardcodes CHANNEL='dmn', ROLE='proactive_thought'.
Builds its own context from data_graph (user synthesis) and the episodes table.
All findings are saved via the memory tool — no UI broadcast, no OutputService call.

"""

import logging
from typing import Optional

from services.message_processor import MessageProcessor
from services.system_message_prompt import DMNSystemMessagePrompt

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_EPISODE_RETRIEVAL_WEIGHT_FLOOR = 0.3
DMN_EPISODE_LOOKBACK_DAYS = 30
_EPISODE_LIMIT = 50


class DMNMessageProcessor(MessageProcessor):
    """MessageProcessor subclass for background DMN proactive reviews.

    Runs as SubconsciousWorker step 5. Builds user-message context from
    data_graph (user synthesis) and recent, non-decayed user-channel episodes.

    Reduced safety caps — DMN turns are best-effort, short-lived:
      MAX_ITERATIONS = 15  (vs. base 30)

    ALWAYS_AVAILABLE excludes 'subagent' to prevent a background
    process from spawning further background work. Includes 'news',
    'search', and 'browser' natively so the model can act without a
    find_tools round-trip (spec §4 of the masterplan).
    """

    CHANNEL = 'dmn'
    ROLE = 'proactive_thought'
    USAGE_CLASS = 'subconscious'
    MAX_ITERATIONS = 15
    SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt

    # news, search, browser promoted to ALWAYS_AVAILABLE so the model can act
    # immediately without a find_tools discovery round-trip (agreed in masterplan §4).
    ALWAYS_AVAILABLE: list[str] = [
        "browser",
        "document",
        "find_tools",
        "list",
        "memory",
        "news",
        "read",
        "review_tool_calls",
        "review_transcript",
        "schedule",
        "search",
    ]
    DISCOVERABLE: list[str] = [
        "calendar",
        "code_eval",
        "contacts",
        "email",
        "programming_docs_search",
        "weather",
    ]

    def get_user_definition(self) -> str:
        """DMN runs as a background process — no human user definition needed."""
        return (
            "The user is 'proactive_thought' — a special background process "
            "that represents your own reflections on recent activity."
        )

    def get_user_prompt(self) -> str:
        """Build the DMN user-message from user synthesis + filtered recent episodes.

        Template:
            ## About the User
            {user_synthesis}

            ## Episodes
            {numbered episode list}

        The SubconsciousWorker._step_dmn() skips this step entirely when no
        user_summary row exists, so by the time get_user_prompt() runs the
        synthesis is guaranteed to be present.
        """
        parts = []

        synthesis = self._fetch_user_synthesis()
        if synthesis:
            parts.append(f"## About the User\n{synthesis}")

        episodes_text = self._fetch_recent_episodes()
        if episodes_text:
            parts.append(f"## Episodes\n{episodes_text}")

        trail = self.get_act_loop_trail()
        if trail:
            parts.append(trail)

        return '\n\n'.join(parts)

    def post_turn(self) -> None:
        """DMN post-turn: metrics only.

        No OutputService.enqueue_proactive call. No UI broadcast.
        DMN's sole side effect is memory-tool writes to data_graph.
        """
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('dmn_turns_total')
        except Exception as e:
            logger.debug("[DMN.post_turn] Metrics failed: %s", e, exc_info=True)

    # ── Private context helpers ───────────────────────────────────────────────

    def _fetch_user_synthesis(self) -> Optional[str]:
        """Read user synthesis from data_graph.

        Prefers user_summary_long for richer DMN reflection context.
        Falls back to user_summary. Returns None when neither row exists.
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM data_graph "
                    "WHERE kind = 'system' "
                    "  AND key IN ('user_summary', 'user_summary_long') "
                    "  AND active = 1 AND deleted_at IS NULL",
                ).fetchall()
            by_key = {row[0]: row[1] for row in rows if row[1]}
            return by_key.get('user_summary_long') or by_key.get('user_summary')
        except Exception as exc:
            logger.warning("[DMN] _fetch_user_synthesis failed: %s", exc)
            return None

    def _fetch_recent_episodes(self) -> str:
        """Retrieve recent, non-decayed user-channel episodes as a numbered list.

        Filter: channel='user', retrieval_weight >= _EPISODE_RETRIEVAL_WEIGHT_FLOOR,
        created or accessed within DMN_EPISODE_LOOKBACK_DAYS days.
        The retrieval_weight floor excludes heavily-decayed episodes (default 1.0;
        decay cycles push toward 0).
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, gist, salience, created_at "
                    "FROM episodes "
                    "WHERE deleted_at IS NULL "
                    "  AND channel = 'user' "
                    "  AND retrieval_weight >= ? "
                    "  AND ( "
                    "      last_accessed_at >= datetime('now', ?) "
                    "      OR created_at >= datetime('now', ?) "
                    "  ) "
                    "ORDER BY retrieval_weight DESC, created_at DESC "
                    "LIMIT ?",
                    (
                        _EPISODE_RETRIEVAL_WEIGHT_FLOOR,
                        f"-{DMN_EPISODE_LOOKBACK_DAYS} days",
                        f"-{DMN_EPISODE_LOOKBACK_DAYS} days",
                        _EPISODE_LIMIT,
                    ),
                ).fetchall()
            return self._format_episodes(rows)
        except Exception as exc:
            logger.warning("[DMN] _fetch_recent_episodes failed: %s", exc)
            return ''

    def _format_episodes(self, rows: list) -> str:
        """Format episode rows as a numbered list.

        Each row: (id, gist, salience, created_at).
        """
        lines = []
        for i, (ep_id, gist, salience, created_at) in enumerate(rows, 1):
            ts = (created_at or '')[:16].replace('T', ' ')
            lines.append(f"{i}. [{ts}] (salience={salience}) {gist or ''}")
        return '\n'.join(lines)
