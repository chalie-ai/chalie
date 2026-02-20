-- Migration 006: Create procedural memory table
-- Date: 2026-02-14
-- Description: Policy weights and skill success stats derived from action outcomes

CREATE TABLE IF NOT EXISTS procedural_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Action identity
    action_name TEXT NOT NULL UNIQUE,

    -- Statistics
    total_attempts INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    avg_reward FLOAT DEFAULT 0.0,

    -- Policy weight (used by cost calculator)
    weight FLOAT DEFAULT 1.0,

    -- History
    reward_history JSONB DEFAULT '[]'::jsonb,
    context_stats JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_procedural_action_name ON procedural_memory (action_name);
CREATE INDEX IF NOT EXISTS idx_procedural_weight ON procedural_memory (weight DESC);

COMMENT ON TABLE procedural_memory IS 'Procedural memory: learned action weights from experience';
COMMENT ON COLUMN procedural_memory.weight IS 'Policy weight derived from success_rate and avg_reward';
COMMENT ON COLUMN procedural_memory.reward_history IS 'Recent reward values for trend analysis';
COMMENT ON COLUMN procedural_memory.context_stats IS 'Per-topic success stats for contextual policy';
