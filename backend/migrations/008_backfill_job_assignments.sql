-- 008: Backfill missing job→provider assignments for existing installs.
--
-- When new cognitive jobs are added to the auto-assign list in providers.py,
-- existing installs (whose DBs were created before those jobs existed) have no
-- row in job_provider_assignments for them.  This migration inserts a default
-- assignment pointing at the first active provider for any job that is currently
-- unassigned.  INSERT OR IGNORE means already-assigned jobs are untouched.
--
-- Jobs backfilled: trait-extraction, document-synthesis, frontal-cortex-scheduled-tool

INSERT OR IGNORE INTO job_provider_assignments (job_name, provider_id)
SELECT job_name, (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) AS provider_id
FROM (
    SELECT 'trait-extraction'            AS job_name
    UNION ALL SELECT 'document-synthesis'
    UNION ALL SELECT 'frontal-cortex-scheduled-tool'
)
WHERE (SELECT id FROM providers WHERE is_active = 1 ORDER BY id LIMIT 1) IS NOT NULL;
