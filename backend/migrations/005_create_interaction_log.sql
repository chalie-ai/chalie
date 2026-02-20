-- Migration 005: Create interaction log table
-- Date: 2026-02-14
-- Description: Append-only audit trail of all raw events (user input, classification, system response)

CREATE TABLE IF NOT EXISTS interaction_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Event classification
    event_type TEXT NOT NULL,

    -- Context keys
    topic TEXT,
    exchange_id TEXT,
    session_id TEXT,
    source TEXT,

    -- Event data
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamp (append-only, no updated_at)
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_interaction_log_topic_created ON interaction_log (topic, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_event_type_created ON interaction_log (event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_session_created ON interaction_log (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_exchange ON interaction_log (exchange_id);

COMMENT ON TABLE interaction_log IS 'Immutable audit trail of all raw events in the system';
COMMENT ON COLUMN interaction_log.event_type IS 'Event type: user_input, classification, system_response, etc.';
COMMENT ON COLUMN interaction_log.payload IS 'Event-specific data (JSONB)';
COMMENT ON COLUMN interaction_log.metadata IS 'Optional metadata (user_id, chat_id, source, etc.)';
