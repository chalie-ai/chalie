-- 006: Simplify user_traits — remove source, is_literal; collapse categories
-- SQLite doesn't support DROP COLUMN before 3.35.0, so rebuild table

-- Collapse categories before rebuild
UPDATE user_traits SET category = 'core' WHERE category IN ('relationship', 'physical');
UPDATE user_traits SET category = 'preference' WHERE category IN ('general', 'micro_preference');
UPDATE user_traits SET category = 'behavioral' WHERE category IN ('communication_style', 'behavioral_pattern');

-- Create new table without source and is_literal
CREATE TABLE IF NOT EXISTS user_traits_new (
    id TEXT PRIMARY KEY,
    trait_key TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    category TEXT DEFAULT 'preference',
    confidence REAL DEFAULT 0.5,
    reinforcement_count INTEGER DEFAULT 1,
    last_reinforced_at TEXT DEFAULT (datetime('now')),
    last_conflict_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    reliability TEXT DEFAULT 'reliable',
    UNIQUE(trait_key)
);

-- Copy data (excluding source and is_literal)
INSERT OR IGNORE INTO user_traits_new (id, trait_key, trait_value, category, confidence, reinforcement_count, last_reinforced_at, last_conflict_at, created_at, updated_at, reliability)
SELECT id, trait_key, trait_value, category, confidence, reinforcement_count, last_reinforced_at, last_conflict_at, created_at, updated_at, reliability
FROM user_traits;

-- Swap tables
DROP TABLE IF EXISTS user_traits;
ALTER TABLE user_traits_new RENAME TO user_traits;
