ALTER TABLE tool_capability_profiles
    ADD COLUMN IF NOT EXISTS domain TEXT DEFAULT 'Other';
