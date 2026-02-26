"""
Test data factories — produce realistic row tuples matching actual DB column orders.

Usage:
    from tests.helpers import make_task_row, make_scheduled_item, make_trait_row

All factories return tuples (matching psycopg2 cursor.fetchone/fetchall) unless
noted otherwise.  Override any field via keyword argument.
"""

import json
from datetime import datetime, timezone, timedelta


# ─── persistent_tasks ────────────────────────────────────────────────
# Column order matches: SELECT id, account_id, thread_id, goal, scope,
#   status, priority, progress, result, result_artifact, iterations_used,
#   max_iterations, created_at, updated_at, expires_at, deadline,
#   next_run_after, fatigue_budget

def make_task_row(
    task_id=1,
    account_id=1,
    thread_id=None,
    goal="Test background task",
    scope=None,
    status="proposed",
    priority=5,
    progress=None,
    result=None,
    result_artifact=None,
    iterations_used=0,
    max_iterations=20,
    created_at=None,
    updated_at=None,
    expires_at=None,
    deadline=None,
    next_run_after=None,
    fatigue_budget=15.0,
):
    """Return an 18-element tuple matching persistent_tasks SELECT order."""
    now = datetime.now(timezone.utc)
    return (
        task_id,
        account_id,
        thread_id,
        goal,
        scope,
        status,
        priority,
        json.dumps(progress) if isinstance(progress, dict) else (progress or "{}"),
        result,
        json.dumps(result_artifact) if isinstance(result_artifact, dict) else result_artifact,
        iterations_used,
        max_iterations,
        created_at or now,
        updated_at or now,
        expires_at or (now + timedelta(days=14)),
        deadline,
        next_run_after,
        fatigue_budget,
    )


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
# Used by episodic_retrieval_service._hybrid_retrieve() which returns dicts,
# but the raw query returns tuples.  This factory returns a dict matching
# the service's output format (since retrieval service converts internally).

def make_episode_row(
    episode_id=1,
    intent=None,
    context=None,
    action="user asked about weather",
    emotion=None,
    outcome="provided forecast",
    gist="Weather conversation about Malta",
    salience=5.0,
    freshness=0.8,
    topic="weather",
    created_at=None,
    activation_score=1.0,
    last_accessed_at=None,
    salience_factors=None,
    open_loops=None,
):
    """Return a dict matching episodic retrieval service output."""
    now = datetime.now(timezone.utc)
    return {
        "id": episode_id,
        "intent": intent or {"type": "exploration", "direction": "open"},
        "context": context or {},
        "action": action,
        "emotion": emotion or {"valence": 0.5, "arousal": 0.5},
        "outcome": outcome,
        "gist": gist,
        "salience": salience,
        "freshness": freshness,
        "topic": topic,
        "created_at": created_at or now,
        "activation_score": activation_score,
        "last_accessed_at": last_accessed_at,
        "salience_factors": salience_factors or {},
        "open_loops": open_loops or [],
    }


# ─── providers ───────────────────────────────────────────────────────
# Column order matches: SELECT id, name, platform, model, host, api_key,
#   dimensions, timeout, is_active

def make_provider_row(
    provider_id=1,
    name="test-provider",
    platform="ollama",
    model="qwen3:4b",
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
