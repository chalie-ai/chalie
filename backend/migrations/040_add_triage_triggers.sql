ALTER TABLE tool_capability_profiles
ADD COLUMN IF NOT EXISTS triage_triggers JSONB DEFAULT '[]'::jsonb;
