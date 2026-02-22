-- Migration 034: Add is_prompt column to scheduled_items
-- is_prompt=TRUE → reminder goes through LLM pipeline (Chalie-style response)
-- is_prompt=FALSE (default) → reminder delivered directly, bypassing LLM
ALTER TABLE scheduled_items
    ADD COLUMN IF NOT EXISTS is_prompt BOOLEAN DEFAULT FALSE;
