-- User traits: per-user trait memory with confidence decay and reinforcement
CREATE TABLE IF NOT EXISTS user_traits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT DEFAULT 'primary',
    trait_key TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence FLOAT DEFAULT 0.5,
    source TEXT DEFAULT 'inferred',
    is_literal BOOLEAN DEFAULT true,
    reinforcement_count INTEGER DEFAULT 1,
    last_reinforced_at TIMESTAMP DEFAULT NOW(),
    last_conflict_at TIMESTAMP,
    embedding vector(256),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, trait_key)
);

CREATE INDEX IF NOT EXISTS idx_user_traits_user ON user_traits(user_id);
CREATE INDEX IF NOT EXISTS idx_user_traits_category ON user_traits(user_id, category);
CREATE INDEX IF NOT EXISTS idx_user_traits_confidence ON user_traits(user_id, confidence);
