ALTER TABLE episodes ADD COLUMN IF NOT EXISTS semantic_consolidation_status TEXT DEFAULT NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_semantic_status ON episodes (semantic_consolidation_status);
