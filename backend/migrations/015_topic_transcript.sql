-- Phase 4B: Topic Transcript — persistent, topic-scoped conversation record.
-- Replaces MemoryStore-based scratchpad/notes with durable SQLite storage.

CREATE TABLE IF NOT EXISTS topic_transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    role TEXT NOT NULL,              -- 'user', 'assistant', 'tool', 'internal'
    content TEXT NOT NULL,
    tool_call_id TEXT,               -- for tool result pairing
    tool_name TEXT,                  -- for tool results
    internal INTEGER DEFAULT 0,      -- 1 = model's working notes, never surfaced to user
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transcript_topic ON topic_transcript(topic, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS topic_transcript_vec USING vec0(
    embedding float[768]
);
