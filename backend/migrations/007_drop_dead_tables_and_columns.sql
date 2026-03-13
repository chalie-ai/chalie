-- 007: Drop unused tables and columns.
--
-- semantic_schemas: never referenced by any service.
-- cognitive_reflexes + cognitive_reflexes_vec: service is disabled (CLAUDE.md).
-- tool_performance_metrics: follow_up_confusion and result_used_in_response
--   are never read or written by any service.
--
-- SQLite < 3.35 doesn't support DROP COLUMN, so we rebuild tool_performance_metrics.

DROP TABLE IF EXISTS semantic_schemas;
DROP TABLE IF EXISTS cognitive_reflexes;
DROP TABLE IF EXISTS cognitive_reflexes_vec;

-- ── tool_performance_metrics: strip follow_up_confusion, result_used_in_response ──
CREATE TABLE IF NOT EXISTS tool_performance_metrics_new (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    exchange_id TEXT,
    invocation_success INTEGER NOT NULL,
    latency_ms REAL,
    cost_estimate REAL DEFAULT 0,
    user_correction INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

INSERT INTO tool_performance_metrics_new
    (id, tool_name, exchange_id, invocation_success, latency_ms, cost_estimate,
     user_correction, created_at)
SELECT id, tool_name, exchange_id, invocation_success, latency_ms, cost_estimate,
       user_correction, created_at
FROM tool_performance_metrics;

DROP TABLE IF EXISTS tool_performance_metrics;
ALTER TABLE tool_performance_metrics_new RENAME TO tool_performance_metrics;

CREATE INDEX IF NOT EXISTS idx_tpm_tool_created ON tool_performance_metrics(tool_name, created_at DESC);
