-- Migration 035: Scheduler type refactor — reminder/task → notification/prompt
-- notification: text surfaced directly (is_prompt=FALSE)
-- prompt: fed to Chalie's LLM pipeline (is_prompt=TRUE)
UPDATE scheduled_items SET item_type = 'prompt' WHERE is_prompt = TRUE;
UPDATE scheduled_items SET item_type = 'notification' WHERE is_prompt = FALSE OR is_prompt IS NULL;
-- Sync is_prompt to match the new item_type values
UPDATE scheduled_items SET is_prompt = TRUE WHERE item_type = 'prompt';
UPDATE scheduled_items SET is_prompt = FALSE WHERE item_type = 'notification';
