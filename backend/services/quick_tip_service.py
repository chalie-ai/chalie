"""
Quick Tip Service — periodic feature discovery tips for the chat interface.

Fetches tip content from chalie-web's /guide/tips.json manifest, tracks
per-tip state (shown/dismissed/retired) in the quick_tip_state table,
and publishes eligible tips to the frontend via WebSocket drift events.

Cooldown: 3h (first 2 weeks) → 24h (mature accounts).
Max shows per tip: 2. Auto-retire when user demonstrates feature knowledge
(≥2 successful non-ephemeral tool_calls for the tip's ability names).
"""

import json
import logging
import random
from datetime import timedelta
from typing import Optional

import requests

from services.database_service import get_shared_db_service
from services.settings_service import SettingsService
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

# Default URL for the tips manifest built by chalie-web
_DEFAULT_TIPS_URL = 'https://chalie.me/guide/tips.json'

# Cooldown between tips
_COOLDOWN_NEW = timedelta(hours=3)
_COOLDOWN_MATURE = timedelta(hours=24)
_MATURE_ACCOUNT_AGE = timedelta(days=14)

# Max times a single tip can be shown before it's retired
_MAX_SHOWS = 2

# Min successful tool_calls to auto-retire a tip (user knows the feature)
_MIN_TOOL_CALLS_FOR_RETIRE = 2


class QuickTipService:
    """Manages quick tip lifecycle: eligibility, display, dismiss, retire."""

    def __init__(self, db=None, tips_url: str = None):
        self._db = db or get_shared_db_service()
        self._tips_url = tips_url or _DEFAULT_TIPS_URL

    def maybe_show_tip(self) -> Optional[dict]:
        """Check eligibility and return a tip dict to show, or None.

        Returns dict with keys: tip_id, title, body, example, icon_svg, ability_names
        or None if no tip should be shown.
        """
        # 1. Check if tips are enabled
        if not self._tips_enabled():
            return None

        # 2. Check cooldown
        if not self._cooldown_elapsed():
            return None

        # 3. Fetch tips manifest
        tips = self._fetch_tips_json()
        if not tips:
            return None

        # 4. Load all tip state from DB
        state_map = self._load_all_tip_state()

        # 5. Auto-retire tips for features the user already knows
        self._auto_retire_known_features(tips, state_map)
        # Reload after retirements
        state_map = self._load_all_tip_state()

        # 6. Filter to eligible tips
        eligible = []
        for tip in tips:
            tip_id = tip.get('tip_id')
            if not tip_id:
                continue
            state = state_map.get(tip_id, {})
            if state.get('dismissed'):
                continue
            if state.get('retired'):
                continue
            if state.get('shown_count', 0) >= _MAX_SHOWS:
                continue
            eligible.append(tip)

        if not eligible:
            return None

        # 7. Pick a random eligible tip
        chosen = random.choice(eligible)
        tip_id = chosen['tip_id']

        # 8. Record the show
        self._record_show(tip_id)

        return chosen

    def dismiss_tip(self, tip_id: str) -> None:
        """Mark a tip as dismissed by the user."""
        if not tip_id:
            return
        self._db.execute(
            """INSERT INTO quick_tip_state (tip_id, dismissed)
               VALUES (?, 1)
               ON CONFLICT(tip_id) DO UPDATE SET dismissed = 1""",
            (tip_id,),
        )

    def mute_tips(self) -> None:
        """Disable all quick tips by setting the quick_tips_enabled setting."""
        svc = SettingsService(self._db)
        svc.set('quick_tips_enabled', 'false')

    # ── Internal ──────────────────────────────────────────────────

    def _tips_enabled(self) -> bool:
        """Check the quick_tips_enabled setting. Default is true (enabled)."""
        svc = SettingsService(self._db)
        val = svc.get('quick_tips_enabled')
        # Not set or any truthy value = enabled
        if val is None or val == '':
            return True
        return val.lower() not in ('false', '0', 'no', 'off')

    def _cooldown_elapsed(self) -> bool:
        """Check if enough time has passed since the last tip was shown."""
        now = utc_now()

        # Determine cooldown based on account age
        cooldown = self._get_cooldown()

        # Find the most recent show across ALL tips
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(last_shown_at) as last FROM quick_tip_state WHERE last_shown_at IS NOT NULL"
            )
            row = cursor.fetchone()

        if not row or not row['last']:
            return True  # Never shown any tip

        last_shown = parse_utc(row['last'])
        return (now - last_shown) >= cooldown

    def _get_cooldown(self) -> timedelta:
        """Return the appropriate cooldown based on account age."""
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT created_at FROM master_account LIMIT 1")
            row = cursor.fetchone()

        if not row or not row['created_at']:
            return _COOLDOWN_NEW

        try:
            created_at = parse_utc(row['created_at'])
        except (ValueError, TypeError):
            return _COOLDOWN_NEW

        account_age = utc_now() - created_at
        if account_age >= _MATURE_ACCOUNT_AGE:
            return _COOLDOWN_MATURE
        return _COOLDOWN_NEW

    def _fetch_tips_json(self) -> list:
        """Fetch the tips manifest from chalie-web. Returns list of tip dicts."""
        try:
            resp = requests.get(self._tips_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            # Expect {"tips": [...]} or just [...]
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get('tips', [])
            return []
        except Exception as e:
            logger.warning("[QuickTip] Failed to fetch tips.json: %s", e)
            return []

    def _load_all_tip_state(self) -> dict:
        """Load all tip states from DB. Returns {tip_id: {shown_count, dismissed, retired, last_shown_at}}."""
        rows = self._db.fetch_all("SELECT * FROM quick_tip_state")
        result = {}
        for row in rows:
            result[row['tip_id']] = {
                'shown_count': row['shown_count'],
                'dismissed': bool(row['dismissed']),
                'retired': bool(row['retired']),
                'last_shown_at': row['last_shown_at'],
            }
        return result

    def _auto_retire_known_features(self, tips: list, state_map: dict) -> None:
        """Auto-retire tips for features the user already demonstrates knowledge of."""
        for tip in tips:
            tip_id = tip.get('tip_id')
            if not tip_id:
                continue
            state = state_map.get(tip_id, {})
            if state.get('retired') or state.get('dismissed'):
                continue

            ability_names = tip.get('ability_names', [])
            if not ability_names:
                continue

            # Check tool_calls for these abilities
            if self._check_usage(ability_names):
                self._retire_tip(tip_id, 'auto_usage')

    def _check_usage(self, ability_names: list) -> bool:
        """Return True if user has >= _MIN_TOOL_CALLS_FOR_RETIRE successful calls for any listed ability."""
        if not ability_names:
            return False
        placeholders = ','.join('?' * len(ability_names))
        rows = self._db.fetch_all(
            f"""SELECT COUNT(*) as cnt FROM tool_calls
                WHERE tool_name IN ({placeholders})
                AND ephemeral = 0""",
            ability_names,
        )
        if not rows:
            return False
        return rows[0].get('cnt', 0) >= _MIN_TOOL_CALLS_FOR_RETIRE

    def _record_show(self, tip_id: str) -> None:
        """Record that a tip was shown. Increments shown_count, updates last_shown_at."""
        now_str = utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self._db.execute(
            """INSERT INTO quick_tip_state (tip_id, shown_count, last_shown_at)
               VALUES (?, 1, ?)
               ON CONFLICT(tip_id) DO UPDATE SET
                   shown_count = shown_count + 1,
                   last_shown_at = ?""",
            (tip_id, now_str, now_str),
        )

    def _retire_tip(self, tip_id: str, reason: str = '') -> None:
        """Mark a tip as retired."""
        self._db.execute(
            """INSERT INTO quick_tip_state (tip_id, retired)
               VALUES (?, 1)
               ON CONFLICT(tip_id) DO UPDATE SET retired = 1""",
            (tip_id,),
        )
        logger.debug("[QuickTip] Retired tip %s (reason: %s)", tip_id, reason)
