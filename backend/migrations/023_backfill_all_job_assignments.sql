-- 023: Backfill missing job→provider assignments for all current cognitive jobs.
--
-- Migration 008 only covered 3 specific jobs.  Jobs added since then
-- (compaction, goal-strategy, reflect-skill, failure-analysis, frontal-cortex-unified,
-- plan-decomposition, semantic-memory, etc.) are missing assignments on existing installs,
-- causing "[ConfigService] No provider assigned for job '...'" fallback warnings.
--
-- INSERT OR IGNORE means already-assigned jobs are untouched.
-- The subquery picks the first active provider as the default.

INSERT OR IGNORE INTO job_provider_assignments (job_name, provider_id)
SELECT job_name, (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) AS provider_id
FROM (
    SELECT 'autobiography'                  AS job_name
    UNION ALL SELECT 'frontal-cortex'
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
