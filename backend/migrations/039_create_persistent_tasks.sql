-- Migration 039: Create persistent_tasks table for multi-session ACT work
-- State machine: PROPOSED → ACCEPTED → IN_PROGRESS → COMPLETED
--                              ↓           ↓
--                           CANCELLED    PAUSED → IN_PROGRESS
--                                          ↓
--                                       CANCELLED
-- Auto-expiry: ACCEPTED/IN_PROGRESS/PAUSED → EXPIRED (14 days default)

CREATE TABLE IF NOT EXISTS persistent_tasks (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES master_account(id),
    thread_id TEXT REFERENCES threads(thread_id),
    goal TEXT NOT NULL,
    scope TEXT,
    status VARCHAR(20) DEFAULT 'proposed',
    priority SMALLINT DEFAULT 5,
    progress JSONB DEFAULT '{}',
    result TEXT,
    result_artifact JSONB,
    iterations_used INTEGER DEFAULT 0,
    max_iterations INTEGER DEFAULT 20,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '14 days',
    deadline TIMESTAMPTZ,
    next_run_after TIMESTAMPTZ,
    fatigue_budget FLOAT DEFAULT 15.0
);

CREATE INDEX IF NOT EXISTS idx_persistent_tasks_status ON persistent_tasks(account_id, status);
CREATE INDEX IF NOT EXISTS idx_persistent_tasks_next_run ON persistent_tasks(status, next_run_after);
