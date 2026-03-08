-- Migration 002: Remove user_id column from all tables.
--
-- Chalie is a single-user system. The user_id column was a remnant of a
-- scrapped multi-user plan. All values were hardcoded 'primary' or 'default'.
-- Each table is recreated (shadow-table pattern) because SQLite DROP COLUMN
-- cannot remove columns that appear in indexes, UNIQUE constraints, or PKs.

PRAGMA foreign_keys = OFF;

-- ── user_traits ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_traits_new (
    id TEXT PRIMARY KEY,
    trait_key TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'inferred',
    is_literal INTEGER DEFAULT 1,
    reinforcement_count INTEGER DEFAULT 1,
    last_reinforced_at TEXT DEFAULT (datetime('now')),
    last_conflict_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(trait_key)
);

INSERT INTO user_traits_new
    (id, trait_key, trait_value, category, confidence, source, is_literal,
     reinforcement_count, last_reinforced_at, last_conflict_at, created_at, updated_at)
SELECT id, trait_key, trait_value, category, confidence, source, is_literal,
       reinforcement_count, last_reinforced_at, last_conflict_at, created_at, updated_at
FROM user_traits;

DROP TABLE IF EXISTS user_traits;
ALTER TABLE user_traits_new RENAME TO user_traits;

CREATE INDEX IF NOT EXISTS idx_user_traits_category ON user_traits(category);
CREATE INDEX IF NOT EXISTS idx_user_traits_confidence ON user_traits(confidence);

-- ── autobiography ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS autobiography_new (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    narrative TEXT NOT NULL,
    section_hashes TEXT NOT NULL DEFAULT '{}',
    episode_cursor TEXT,
    episodes_since INTEGER NOT NULL DEFAULT 0,
    synthesis_model TEXT,
    synthesis_ms INTEGER,
    delta_summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(version)
);

INSERT INTO autobiography_new
    (id, version, narrative, section_hashes, episode_cursor, episodes_since,
     synthesis_model, synthesis_ms, delta_summary, created_at)
SELECT id, version, narrative, section_hashes, episode_cursor, episodes_since,
       synthesis_model, synthesis_ms, delta_summary, created_at
FROM autobiography;

DROP TABLE IF EXISTS autobiography;
ALTER TABLE autobiography_new RENAME TO autobiography;

CREATE INDEX IF NOT EXISTS idx_autobiography_version ON autobiography(version DESC);

-- ── lists ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lists_new (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    list_type TEXT NOT NULL DEFAULT 'checklist',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at TEXT
);

INSERT INTO lists_new (id, name, list_type, metadata, created_at, updated_at, deleted_at)
SELECT id, name, list_type, metadata, created_at, updated_at, deleted_at
FROM lists;

DROP TABLE IF EXISTS lists;
ALTER TABLE lists_new RENAME TO lists;

CREATE UNIQUE INDEX IF NOT EXISTS idx_lists_name_unique
    ON lists(name COLLATE NOCASE)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_lists_active ON lists(created_at DESC) WHERE deleted_at IS NULL;

-- ── moments ──────────────────────────────────────────────────────────────────
-- Skipped: the moments table is dropped entirely by migration 004, and
-- schema.sql no longer creates it. On a fresh install moments never exists,
-- so the shadow-copy pattern would crash. The user_id removal here is moot
-- because the table is dropped before it can be read again.

-- ── user_tool_preferences ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_tool_preferences_new (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL UNIQUE,
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    explicit_preference REAL DEFAULT 0,
    implicit_preference REAL DEFAULT 0,
    last_used_at TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

INSERT INTO user_tool_preferences_new
    (id, tool_name, usage_count, success_count, explicit_preference,
     implicit_preference, last_used_at, updated_at)
SELECT id, tool_name, usage_count, success_count, explicit_preference,
       implicit_preference, last_used_at, updated_at
FROM user_tool_preferences;

DROP TABLE IF EXISTS user_tool_preferences;
ALTER TABLE user_tool_preferences_new RENAME TO user_tool_preferences;

-- ── temporal_aggregate ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS temporal_aggregate_new (
    observation_type TEXT NOT NULL,
    observed_value TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,
    hour_bucket INTEGER NOT NULL,
    device_class TEXT NOT NULL DEFAULT '',
    count INTEGER DEFAULT 0,
    last_seen TEXT,
    PRIMARY KEY(observation_type, observed_value, day_of_week, hour_bucket, device_class)
);

INSERT INTO temporal_aggregate_new
    (observation_type, observed_value, day_of_week, hour_bucket, device_class, count, last_seen)
SELECT observation_type, observed_value, day_of_week, hour_bucket, device_class, count, last_seen
FROM temporal_aggregate;

DROP TABLE IF EXISTS temporal_aggregate;
ALTER TABLE temporal_aggregate_new RENAME TO temporal_aggregate;

-- ── threads ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS threads_new (
    thread_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'unknown',
    state TEXT NOT NULL DEFAULT 'active',
    current_topic TEXT,
    topic_history TEXT DEFAULT '[]',
    exchange_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    last_activity TEXT DEFAULT (datetime('now')),
    expired_at TEXT,
    summary TEXT
);

INSERT INTO threads_new
    (thread_id, channel_id, platform, state, current_topic, topic_history,
     exchange_count, created_at, last_activity, expired_at, summary)
SELECT thread_id, channel_id, platform, state, current_topic, topic_history,
       exchange_count, created_at, last_activity, expired_at, summary
FROM threads;

DROP TABLE IF EXISTS threads;
ALTER TABLE threads_new RENAME TO threads;

CREATE INDEX IF NOT EXISTS idx_threads_channel ON threads(channel_id);
CREATE INDEX IF NOT EXISTS idx_threads_state ON threads(state);

PRAGMA foreign_keys = ON;
