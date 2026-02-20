-- Migration 003: Create cortex_iterations table for ACT loop logging

CREATE TABLE IF NOT EXISTS cortex_iterations (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign keys & context
    topic TEXT NOT NULL,
    exchange_id TEXT,
    session_id TEXT,

    -- Loop metadata
    loop_id UUID NOT NULL,
    iteration_number INTEGER NOT NULL,

    -- Timing
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    execution_time_ms FLOAT,

    -- Confidence & paths
    chosen_mode TEXT,
    chosen_confidence FLOAT,
    alternative_paths JSONB,

    -- Cost breakdown
    iteration_cost FLOAT,
    diminishing_cost FLOAT,
    uncertainty_cost FLOAT,
    action_base_cost FLOAT,
    total_cost FLOAT,
    cumulative_cost FLOAT,

    -- Efficiency
    efficiency_score FLOAT,
    expected_confidence_gain FLOAT,

    -- Net value components (added 2026-02-09)
    task_value FLOAT,
    future_leverage FLOAT,
    effort_estimate TEXT,
    effort_multiplier FLOAT,
    iteration_penalty FLOAT,
    exploration_bonus FLOAT,
    net_value FLOAT,

    -- Decision data
    decision_override BOOLEAN,
    overridden_mode TEXT,
    termination_reason TEXT,

    -- Actions executed
    actions_executed JSONB,
    action_count INTEGER,
    action_success_count INTEGER,

    -- Full response
    frontal_cortex_response JSONB,

    -- Metadata
    config_snapshot JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_loop ON cortex_iterations(loop_id, iteration_number);
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_topic ON cortex_iterations(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_exchange ON cortex_iterations(exchange_id);

-- Add comment
COMMENT ON TABLE cortex_iterations IS 'Logs each iteration of frontal cortex ACT loop for reflection and meta-learning';
