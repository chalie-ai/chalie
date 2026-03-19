-- Migration 013: Remove 'proposed' state from persistent tasks.
--
-- Tasks stuck in 'proposed' state were never picked up by the worker.
-- Transition them to 'accepted' so they start processing.
UPDATE persistent_tasks SET status = 'accepted', updated_at = datetime('now')
WHERE status = 'proposed';
