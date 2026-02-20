-- Migration 016: Upgrade all embedding columns to vector(768)
-- Consolidates to embeddinggemma (768-dim) via Ollama, replacing fastembed (256-dim).

-- semantic_concepts: TRUNCATE (rebuilds from episodes via idle consolidation)
DROP INDEX IF EXISTS idx_concepts_embedding;
TRUNCATE TABLE semantic_concepts CASCADE;
ALTER TABLE semantic_concepts ALTER COLUMN embedding TYPE vector(768);
CREATE INDEX IF NOT EXISTS idx_concepts_embedding ON semantic_concepts
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- user_traits: NULL embeddings (regenerate on next write, keep data)
UPDATE user_traits SET embedding = NULL;
ALTER TABLE user_traits ALTER COLUMN embedding TYPE vector(768);

-- episodes: NULL embeddings (keep data, vector search degrades to full-text)
UPDATE episodes SET embedding = NULL;
ALTER TABLE episodes ALTER COLUMN embedding TYPE vector(768);
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- topics: already vector(768) from migration 010 — no change
