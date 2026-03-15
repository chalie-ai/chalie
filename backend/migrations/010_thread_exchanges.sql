-- Persist conversation exchanges to SQLite so chat history survives server restarts.
-- MemoryStore remains the hot path; SQLite is the durable fallback.

CREATE TABLE IF NOT EXISTS thread_exchanges (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    prompt_message TEXT NOT NULL DEFAULT '',
    prompt_time TEXT NOT NULL,
    response_message TEXT,
    response_time TEXT,
    response_error TEXT,
    generation_time_ms REAL,
    steps TEXT DEFAULT '[]',
    memory_chunk TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_thread_exchanges_thread
    ON thread_exchanges(thread_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_thread_exchanges_created
    ON thread_exchanges(created_at ASC);
