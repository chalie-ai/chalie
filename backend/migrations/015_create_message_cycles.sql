-- Message cycles: tracks every message through the system as an independent processing unit.
-- Enables async follow-ups, tool result tracking, and causal chain preservation.

CREATE TABLE IF NOT EXISTS message_cycles (
    cycle_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_cycle_id UUID REFERENCES message_cycles(cycle_id),
    root_cycle_id   UUID NOT NULL,
    topic           TEXT NOT NULL,
    cycle_type      TEXT NOT NULL,
    source          TEXT NOT NULL,
    content         TEXT,
    intent          JSONB,
    metadata        JSONB DEFAULT '{}',
    status          TEXT DEFAULT 'pending',
    depth           INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cycles_parent ON message_cycles(parent_cycle_id);
CREATE INDEX IF NOT EXISTS idx_cycles_root ON message_cycles(root_cycle_id);
CREATE INDEX IF NOT EXISTS idx_cycles_topic_created ON message_cycles(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cycles_status_type ON message_cycles(status, cycle_type);
