CREATE TABLE IF NOT EXISTS delegate_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic TEXT NOT NULL,
    exchange_id TEXT,
    specialist TEXT NOT NULL,
    task TEXT NOT NULL,
    context TEXT,
    success_criteria TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    -- Specialist output
    result TEXT,
    structured_output JSONB DEFAULT '{}',
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    execution_time FLOAT,
    -- Trust scores (5-axis)
    utility_score FLOAT,
    epistemic_score FLOAT,
    consistency_score FLOAT,
    uncertainty_score FLOAT,
    completeness_score FLOAT,
    trust_composite FLOAT,
    -- Reward
    reward FLOAT,
    -- Tuition
    studied BOOLEAN DEFAULT false,
    tuition_mode TEXT,
    tuition_result TEXT,
    -- Metadata
    delegation_pattern JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_delegate_tasks_status ON delegate_tasks(status);
CREATE INDEX idx_delegate_tasks_topic ON delegate_tasks(topic);
CREATE INDEX idx_delegate_tasks_specialist ON delegate_tasks(specialist);
CREATE INDEX idx_delegate_tasks_topic_completed
    ON delegate_tasks(topic, completed_at DESC) WHERE status = 'completed';
CREATE INDEX idx_delegate_tasks_tuition
    ON delegate_tasks(studied, reward) WHERE status = 'completed' AND studied = false;
CREATE INDEX idx_delegate_tasks_trust
    ON delegate_tasks(specialist, trust_composite) WHERE status = 'completed';
