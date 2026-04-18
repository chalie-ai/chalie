"""
Test data factories — produce realistic row tuples matching actual DB column orders.

Usage:
    from tests.helpers import make_scheduled_item, make_trait_row

All factories return tuples (matching cursor.fetchone/fetchall) unless
noted otherwise.  Override any field via keyword argument.
"""

from datetime import datetime, timezone, timedelta


# ─── scheduled_items ─────────────────────────────────────────────────
# Column order matches: SELECT id, item_type, message, due_at, recurrence,
#   window_start, window_end, topic, created_by_session, group_id, is_prompt

def make_scheduled_item(
    item_id="sched-001",
    item_type="reminder",
    message="Test reminder",
    due_at=None,
    recurrence=None,
    window_start=None,
    window_end=None,
    topic=None,
    created_by_session=None,
    group_id=None,
    is_prompt=False,
):
    """Return an 11-element tuple matching scheduled_items SELECT order."""
    due_at = due_at or (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    return (
        item_id, item_type, message, due_at, recurrence,
        window_start, window_end, topic, created_by_session, group_id,
        is_prompt,
    )


# ─── user_traits ─────────────────────────────────────────────────────
# Column order matches: SELECT trait_key, trait_value, confidence, category

def make_trait_row(
    trait_key="name",
    trait_value="Dylan",
    confidence=0.9,
    category="core",
):
    """Return a 4-element tuple matching user_traits SELECT order."""
    return (trait_key, trait_value, confidence, category)


# ─── episodes ────────────────────────────────────────────────────────
# Used by episodic_service._hybrid_retrieve() which returns dicts,
# but the raw query returns tuples.  This factory returns a dict matching
# the service's output format (since the service converts internally).

def make_episode_row(
    episode_id="ep-001",
    gist="Weather conversation about Malta",
    salience=5,
    channel="weather",
    created_at=None,
    last_accessed_at=None,
    emotional_valence=None,
    emotional_arousal=None,
    transcript_ids=None,
    transcript_id_start=None,
    transcript_id_end=None,
    consolidated_from=None,
    consolidated_into=None,
    storage_strength=1.0,
    retrieval_weight=1.0,
):
    """Return a dict matching episodic retrieval service output."""
    now = datetime.now(timezone.utc)
    return {
        "id": episode_id,
        "gist": gist,
        "salience": salience,
        "channel": channel,
        "created_at": created_at or now,
        "last_accessed_at": last_accessed_at,
        "emotional_valence": emotional_valence,
        "emotional_arousal": emotional_arousal,
        "transcript_ids": transcript_ids or "[]",
        "transcript_id_start": transcript_id_start,
        "transcript_id_end": transcript_id_end,
        "consolidated_from": consolidated_from or "[]",
        "consolidated_into": consolidated_into,
        "storage_strength": storage_strength,
        "retrieval_weight": retrieval_weight,
    }


# ─── providers ───────────────────────────────────────────────────────
# Column order matches: SELECT id, name, platform, model, host, api_key,
#   dimensions, timeout, is_active

def make_provider_row(
    provider_id=1,
    name="test-provider",
    platform="ollama",
    model="gemma4:31b",
    host="http://localhost:11434",
    api_key=None,
    dimensions=256,
    timeout=30,
    is_active=True,
):
    """Return a 9-element tuple matching providers SELECT order."""
    return (
        provider_id, name, platform, model, host,
        api_key, dimensions, timeout, is_active,
    )


