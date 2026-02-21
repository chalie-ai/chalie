CREATE TABLE IF NOT EXISTS triage_calibration_events (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange_id             TEXT,
    topic                   TEXT,

    triage_branch           TEXT NOT NULL,
    triage_mode             TEXT NOT NULL,
    tool_selected           TEXT[],
    confidence_internal     FLOAT,
    confidence_tool_need    FLOAT,
    reasoning               TEXT,

    freshness_risk          FLOAT,
    decision_entropy        FLOAT,

    self_eval_override      BOOLEAN DEFAULT FALSE,
    self_eval_reason        TEXT,

    outcome_mode            TEXT,
    outcome_tools_used      TEXT[],
    outcome_tool_success    BOOLEAN,
    outcome_latency_ms      FLOAT,
    tool_abstention         BOOLEAN DEFAULT FALSE,

    signal_rephrase         BOOLEAN DEFAULT FALSE,
    signal_correction       BOOLEAN DEFAULT FALSE,
    signal_explicit_lookup  BOOLEAN DEFAULT FALSE,
    signal_abandonment      BOOLEAN DEFAULT FALSE,

    correctness_label       TEXT,
    correctness_score       FLOAT,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tce_created ON triage_calibration_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tce_topic ON triage_calibration_events(topic, created_at DESC);
