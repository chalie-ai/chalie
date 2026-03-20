-- Phase 4D: Topic Compactions — incremental conversation summarization.
-- Stores compacted context per topic with a watermark pointing to the
-- last transcript entry that was included in the compaction.

CREATE TABLE IF NOT EXISTS topic_compactions (
    topic TEXT PRIMARY KEY,
    compacted_text TEXT NOT NULL,
    compacted_up_to_id INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (compacted_up_to_id) REFERENCES topic_transcript(id)
);
