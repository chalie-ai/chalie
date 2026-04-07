-- Migration 035: Channel rename
--
-- Renames the "thread" and "topic" naming convention to a simpler "channel"
-- model throughout the schema.
--
-- Changes:
--   1. topic_transcript  → transcript        (topic col → channel)
--   2. topic_compactions → compactions       (topic col → channel, FK updated)
--   3. tool_calls        — recreated         (FK: topic_transcript → transcript)
--   4. episodes.topic                        → channel  (ALTER … RENAME COLUMN)
--   5. cortex_iterations.topic               → channel
--   6. interaction_log.topic                 → channel
--   7. identity_events.topic                 → channel
--   8. scheduled_items.topic                 → channel
--   9. goals.thread_id                       → channel
--  10. persistent_tasks.thread_id            → channel  (FK to threads dropped)
--  11. DROP TABLE threads
--  12. DROP TABLE thread_exchanges
--  13. DROP TABLE topics

-- ────────────────────────────────────────────────────────────────
-- 1. topic_transcript → transcript
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS transcript (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name   TEXT,
    internal    INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO transcript (id, channel, role, content, tool_call_id, tool_name, internal, created_at)
SELECT id, topic, role, content, tool_call_id, tool_name, internal, created_at
FROM topic_transcript;

-- Recreate indexes on new table
CREATE INDEX IF NOT EXISTS idx_transcript_channel ON transcript(channel, created_at);
CREATE INDEX IF NOT EXISTS idx_transcript_channel_created_desc ON transcript(channel, created_at DESC);

-- Recreate vector companion table
DROP TABLE IF EXISTS transcript_vec;
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_vec USING vec0(
    embedding float[768]
);

-- Copy vector embeddings (rowids correspond to transcript.id)
INSERT INTO transcript_vec (rowid, embedding)
SELECT rowid, embedding
FROM topic_transcript_vec;

-- ────────────────────────────────────────────────────────────────
-- 2. topic_compactions → compactions
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS compactions (
    channel             TEXT PRIMARY KEY,
    compacted_text      TEXT NOT NULL,
    compacted_up_to_id  INTEGER NOT NULL,
    token_count         INTEGER DEFAULT 0,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (compacted_up_to_id) REFERENCES transcript(id)
);

INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at)
SELECT topic, compacted_text, compacted_up_to_id, token_count, updated_at
FROM topic_compactions;

-- ────────────────────────────────────────────────────────────────
-- 3. tool_calls — recreate with FK pointing to transcript
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tool_calls_new (
    transcript_id INTEGER NOT NULL,
    tool_name     TEXT NOT NULL,
    params        TEXT DEFAULT '{}',
    result        TEXT DEFAULT '',
    invoked_by    TEXT NOT NULL CHECK(invoked_by IN ('system', 'llm')),
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (transcript_id) REFERENCES transcript(id)
);

INSERT INTO tool_calls_new (transcript_id, tool_name, params, result, invoked_by, created_at)
SELECT transcript_id, tool_name, params, result, invoked_by, created_at
FROM tool_calls;

DROP TABLE tool_calls;
ALTER TABLE tool_calls_new RENAME TO tool_calls;

CREATE INDEX IF NOT EXISTS idx_tool_calls_transcript ON tool_calls(transcript_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at DESC);

-- ────────────────────────────────────────────────────────────────
-- 4–8. Rename .topic → .channel via ALTER TABLE … RENAME COLUMN
-- ────────────────────────────────────────────────────────────────

ALTER TABLE episodes            RENAME COLUMN topic TO channel;
ALTER TABLE cortex_iterations   RENAME COLUMN topic TO channel;
ALTER TABLE interaction_log     RENAME COLUMN topic TO channel;
ALTER TABLE identity_events     RENAME COLUMN topic TO channel;
ALTER TABLE scheduled_items     RENAME COLUMN topic TO channel;

-- Recreate affected indexes that referenced the old column name
DROP INDEX IF EXISTS idx_episodes_topic;
DROP INDEX IF EXISTS idx_episodes_composite;
DROP INDEX IF EXISTS idx_cortex_iterations_topic;
DROP INDEX IF EXISTS idx_interaction_log_topic_created;

CREATE INDEX IF NOT EXISTS idx_episodes_channel ON episodes(channel) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_composite ON episodes(channel, retrieval_weight DESC, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_channel ON cortex_iterations(channel, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interaction_log_channel_created ON interaction_log(channel, created_at);

-- ────────────────────────────────────────────────────────────────
-- 9. goals.thread_id → goals.channel
-- ────────────────────────────────────────────────────────────────

ALTER TABLE goals RENAME COLUMN thread_id TO channel;

-- ────────────────────────────────────────────────────────────────
-- 10. persistent_tasks.thread_id → persistent_tasks.channel
--     Drop FK to threads (threads table is being deleted)
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS persistent_tasks_new (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      INTEGER REFERENCES master_account(id),
    channel         TEXT,
    goal            TEXT NOT NULL,
    scope           TEXT,
    status          TEXT DEFAULT 'accepted',
    priority        INTEGER DEFAULT 5,
    progress        TEXT DEFAULT '{}',
    result          TEXT,
    result_artifact TEXT,
    iterations_used INTEGER DEFAULT 0,
    max_iterations  INTEGER DEFAULT 20,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    expires_at      TEXT DEFAULT (datetime('now', '+14 days')),
    deadline        TEXT,
    next_run_after  TEXT,
    fatigue_budget  REAL DEFAULT 15.0
);

INSERT INTO persistent_tasks_new (
    id, account_id, channel, goal, scope, status, priority, progress,
    result, result_artifact, iterations_used, max_iterations,
    created_at, updated_at, expires_at, deadline, next_run_after, fatigue_budget
)
SELECT
    id, account_id, thread_id, goal, scope, status, priority, progress,
    result, result_artifact, iterations_used, max_iterations,
    created_at, updated_at, expires_at, deadline, next_run_after, fatigue_budget
FROM persistent_tasks;

DROP TABLE persistent_tasks;
ALTER TABLE persistent_tasks_new RENAME TO persistent_tasks;

CREATE INDEX IF NOT EXISTS idx_persistent_tasks_status ON persistent_tasks(account_id, status);
CREATE INDEX IF NOT EXISTS idx_persistent_tasks_next_run ON persistent_tasks(status, next_run_after);

-- Recreate vec companion and copy existing embeddings
ALTER TABLE persistent_tasks_vec RENAME TO persistent_tasks_vec_old;
CREATE VIRTUAL TABLE IF NOT EXISTS persistent_tasks_vec USING vec0(embedding float[768]);
INSERT INTO persistent_tasks_vec(rowid, embedding) SELECT rowid, embedding FROM persistent_tasks_vec_old;
DROP TABLE IF EXISTS persistent_tasks_vec_old;

-- ────────────────────────────────────────────────────────────────
-- 11–13. Drop obsolete tables
-- ────────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS topic_transcript;
DROP TABLE IF EXISTS topic_transcript_vec;
DROP TABLE IF EXISTS topic_compactions;
DROP TABLE IF EXISTS threads;
DROP TABLE IF EXISTS thread_exchanges;
DROP TABLE IF EXISTS topics;

-- ────────────────────────────────────────────────────────────────
-- 14. Drop thread_id from interaction_log (recreate without it)
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS interaction_log_new (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    channel TEXT,
    exchange_id TEXT,
    session_id TEXT,
    source TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

INSERT INTO interaction_log_new
    SELECT id, event_type, channel, exchange_id, session_id, source,
           payload, metadata, created_at
    FROM interaction_log;

DROP TABLE interaction_log;
ALTER TABLE interaction_log_new RENAME TO interaction_log;

CREATE INDEX IF NOT EXISTS idx_interaction_log_channel_created ON interaction_log(channel, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_event_type_created ON interaction_log(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_session_created ON interaction_log(session_id, created_at);
