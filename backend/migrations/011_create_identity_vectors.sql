-- Identity vectors: 6 control dimensions for dynamic personality
CREATE TABLE IF NOT EXISTS identity_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vector_name VARCHAR(30) NOT NULL UNIQUE,
    baseline_weight FLOAT NOT NULL DEFAULT 0.5,
    current_activation FLOAT NOT NULL DEFAULT 0.5,
    plasticity_rate FLOAT NOT NULL DEFAULT 0.05,
    inertia_rate FLOAT NOT NULL DEFAULT 0.1,
    min_cap FLOAT NOT NULL DEFAULT 0.2,
    max_cap FLOAT NOT NULL DEFAULT 0.8,
    reinforcement_count INT DEFAULT 0,
    signal_history JSONB DEFAULT '[]',
    baseline_drift_today FLOAT DEFAULT 0,
    drift_window_start TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default archetype: "Respectful, charismatic, motivated by learning & discovery"
INSERT INTO identity_vectors (vector_name, baseline_weight, current_activation, plasticity_rate, inertia_rate, min_cap, max_cap)
VALUES
    ('curiosity',           0.7, 0.7, 0.05, 0.10, 0.3, 0.9),
    ('assertiveness',       0.6, 0.6, 0.04, 0.10, 0.3, 0.8),
    ('warmth',              0.6, 0.6, 0.05, 0.10, 0.3, 0.8),
    ('playfulness',         0.4, 0.4, 0.04, 0.10, 0.2, 0.7),
    ('skepticism',          0.5, 0.5, 0.03, 0.10, 0.2, 0.7),
    ('emotional_intensity', 0.4, 0.4, 0.02, 0.15, 0.2, 0.6)
ON CONFLICT (vector_name) DO NOTHING;

-- Identity event log for observability/debugging
CREATE TABLE IF NOT EXISTS identity_events (
    id BIGSERIAL PRIMARY KEY,
    vector_name VARCHAR(30) NOT NULL,
    old_activation FLOAT NOT NULL,
    new_activation FLOAT NOT NULL,
    signal_source VARCHAR(20) NOT NULL,
    signal_value FLOAT,
    topic VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_identity_events_time ON identity_events(created_at);
CREATE INDEX IF NOT EXISTS idx_identity_events_vector ON identity_events(vector_name, created_at);
