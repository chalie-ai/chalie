-- Seed and Data Backfills
-- Combines all historical data transformations into a single source of truth.

-- 1. Backfill topic_transcript from thread_exchanges for pre-transcript instances.
-- Guard: only runs if topic_transcript is empty and thread_exchanges has data.
INSERT INTO topic_transcript (topic, role, content, created_at)
SELECT topic, role, content, created_at FROM (
    SELECT topic, 'user' AS role, prompt_message AS content, created_at
    FROM thread_exchanges
    WHERE prompt_message IS NOT NULL AND prompt_message != ''
    UNION ALL
    SELECT topic, 'assistant' AS role, response_message AS content, created_at
    FROM thread_exchanges
    WHERE response_message IS NOT NULL AND response_message != ''
)
WHERE NOT EXISTS (SELECT 1 FROM topic_transcript LIMIT 1)
ORDER BY created_at ASC;

-- 2. Select the first active provider as the global provider for existing installs.
INSERT OR IGNORE INTO settings (key, value, value_type, description, is_sensitive)
SELECT 'selected_provider_id',
       (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1),
       'int',
       'ID of the active LLM provider',
       0
WHERE (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) IS NOT NULL;
