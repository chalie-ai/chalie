CREATE TABLE IF NOT EXISTS tuition_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    specialist TEXT NOT NULL,
    topic TEXT,
    rule_type TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'provisional',
    source_task_id UUID REFERENCES delegate_tasks(id),
    confirmation_count INTEGER DEFAULT 0,
    confirmation_threshold INTEGER DEFAULT 3,
    contradicting_count INTEGER DEFAULT 0,
    last_confirmed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    promoted_at TIMESTAMP
);

CREATE INDEX idx_tuition_rules_specialist ON tuition_rules(specialist, status);
CREATE INDEX idx_tuition_rules_topic ON tuition_rules(topic, status);
CREATE INDEX idx_tuition_rules_status ON tuition_rules(status);
