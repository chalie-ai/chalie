CREATE TABLE IF NOT EXISTS routing_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic TEXT NOT NULL,
    exchange_id TEXT,
    selected_mode TEXT NOT NULL,
    router_confidence FLOAT,
    scores JSONB NOT NULL,
    tiebreaker_used BOOLEAN DEFAULT FALSE,
    tiebreaker_candidates JSONB,
    margin FLOAT,
    effective_margin FLOAT,
    signal_snapshot JSONB NOT NULL,
    weight_snapshot JSONB,
    routing_time_ms FLOAT,
    feedback JSONB,
    reflection JSONB,
    previous_mode TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_topic ON routing_decisions(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_mode ON routing_decisions(selected_mode, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_unreflected ON routing_decisions(created_at) WHERE reflection IS NULL;
