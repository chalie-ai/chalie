-- 033_add_scheduler_group_id.sql
-- Links recurring schedule occurrences so fire history can be retrieved per schedule.
-- group_id = id of the root (first) item in a recurring series.
-- One-time items: group_id = their own id.

ALTER TABLE scheduled_items
    ADD COLUMN IF NOT EXISTS group_id TEXT;

-- Backfill: existing rows treat themselves as the root of their own group
UPDATE scheduled_items
    SET group_id = id
    WHERE group_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_scheduled_items_group_id
    ON scheduled_items(group_id, due_at DESC);
