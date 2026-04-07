-- Tool calls audit table — tracks every tool invocation per transcript turn.
-- Used to reconstruct full conversation context (memory tags, tool results)
-- when building the messages array for LLM calls.
-- Retention: 25k newest entries, older rows purged on 12h schedule.

CREATE TABLE IF NOT EXISTS tool_calls (
    transcript_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    params TEXT DEFAULT '{}',
    result TEXT DEFAULT '',
    invoked_by TEXT NOT NULL CHECK(invoked_by IN ('system', 'llm')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (transcript_id) REFERENCES topic_transcript(id)
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_transcript ON tool_calls(transcript_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at DESC);
