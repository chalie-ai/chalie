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

-- 2. Backfill missing job→provider assignments for existing installs.
-- Mirrors backend/configs/cognitive_jobs.json. Keep in lockstep when jobs
-- are added or removed — SchemaConvergenceService prunes orphan rows on
-- boot, so stale entries here are silently dropped, not warned.
INSERT OR IGNORE INTO job_provider_assignments (job_name, provider_id)
SELECT job_name, (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) AS provider_id
FROM (
    SELECT 'frontal-cortex-unified'
    UNION ALL SELECT 'cognitive-triage'
    UNION ALL SELECT 'tool-synthesis'
)
WHERE (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) IS NOT NULL;
