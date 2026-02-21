CREATE TABLE IF NOT EXISTS tool_capability_profiles (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name           TEXT NOT NULL UNIQUE,
    tool_type           TEXT NOT NULL DEFAULT 'tool',

    short_summary       TEXT NOT NULL,
    full_profile        TEXT NOT NULL,
    usage_scenarios     JSONB NOT NULL DEFAULT '[]',
    anti_scenarios      JSONB NOT NULL DEFAULT '[]',
    complementary_skills JSONB DEFAULT '[]',

    embedding           vector(768),

    manifest_hash       TEXT,

    enrichment_episode_ids JSONB DEFAULT '[]',
    enrichment_count       INTEGER DEFAULT 0,
    last_enriched_at       TIMESTAMPTZ,

    avg_latency_ms      FLOAT DEFAULT 0,
    cost_tier           TEXT DEFAULT 'free',
    reliability_score   FLOAT DEFAULT 1.0,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tcp_tool_name ON tool_capability_profiles(tool_name);
CREATE INDEX IF NOT EXISTS idx_tcp_embedding ON tool_capability_profiles
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;
