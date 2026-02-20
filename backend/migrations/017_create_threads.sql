CREATE TABLE IF NOT EXISTS threads (
    thread_id       TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    channel_id      TEXT NOT NULL,
    platform        TEXT NOT NULL DEFAULT 'unknown',
    state           TEXT NOT NULL DEFAULT 'active',
    current_topic   TEXT,
    topic_history   JSONB DEFAULT '[]',
    exchange_count  INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_activity   TIMESTAMPTZ DEFAULT NOW(),
    expired_at      TIMESTAMPTZ,
    summary         TEXT
);

CREATE INDEX IF NOT EXISTS idx_threads_user_channel ON threads(user_id, channel_id);
CREATE INDEX IF NOT EXISTS idx_threads_state ON threads(state);

ALTER TABLE interaction_log ADD COLUMN IF NOT EXISTS thread_id TEXT;
