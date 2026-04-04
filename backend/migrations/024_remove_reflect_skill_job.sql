-- 024: Remove reflect-skill provider job assignment.
--
-- reflect_skill.py has been deleted; PostLoopReflectionService is the replacement.
-- The job no longer exists in cognitive_jobs.json so any assigned provider row
-- is orphaned and should be cleaned up.

DELETE FROM job_provider_assignments WHERE job_name = 'reflect-skill';
