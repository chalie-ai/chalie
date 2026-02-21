CREATE TABLE IF NOT EXISTS tool_performance_metrics (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name               TEXT NOT NULL,
    exchange_id             TEXT,
    invocation_success      BOOLEAN NOT NULL,
    latency_ms              FLOAT,
    cost_estimate           FLOAT DEFAULT 0,
    user_correction         BOOLEAN DEFAULT FALSE,
    follow_up_confusion     BOOLEAN DEFAULT FALSE,
    result_used_in_response BOOLEAN DEFAULT TRUE,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tpm_tool_created ON tool_performance_metrics(tool_name, created_at DESC);
