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
-- Covers all current cognitive jobs to prevent fallback warnings.
INSERT OR IGNORE INTO job_provider_assignments (job_name, provider_id)
SELECT job_name, (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) AS provider_id
FROM (
    SELECT 'frontal-cortex'
    UNION ALL SELECT 'frontal-cortex-unified'
    UNION ALL SELECT 'cognitive-triage'
    UNION ALL SELECT 'frontal-cortex-act'
    UNION ALL SELECT 'plan-decomposition'
    UNION ALL SELECT 'cognitive-drift'
    UNION ALL SELECT 'episodic-memory'
    UNION ALL SELECT 'frontal-cortex-proactive'
    UNION ALL SELECT 'semantic-memory'
    UNION ALL SELECT 'experience-assimilation'
    UNION ALL SELECT 'autonomous-ambient-tool'
    UNION ALL SELECT 'frontal-cortex-scheduled-tool'
    UNION ALL SELECT 'trait-extraction'
    UNION ALL SELECT 'moment-enrichment'
    UNION ALL SELECT 'document-synthesis'
    UNION ALL SELECT 'compaction'
    UNION ALL SELECT 'goal-strategy'
    UNION ALL SELECT 'reflect-skill'
    UNION ALL SELECT 'failure-analysis'
)
WHERE (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) IS NOT NULL;

-- v0.3.3: Purge innate skill rows from tool_capability_profiles.
-- Innate skills are pre-injected from TOOL_SCHEMA — they never use profiles.
-- Their rows pollute the k-NN window for find_tools.
DELETE FROM tool_capability_profiles WHERE tool_type = 'skill';
DELETE FROM tool_capability_profiles_vec
    WHERE rowid NOT IN (SELECT rowid FROM tool_capability_profiles);
