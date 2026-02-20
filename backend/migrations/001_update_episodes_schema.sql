-- Migration: Update episodes table for new intent/context/salience structure
-- Date: 2026-02-08
-- Description: Adds salience_factors, open_loops, and updates intent to JSONB

-- Add new columns
ALTER TABLE episodes
ADD COLUMN IF NOT EXISTS salience_factors JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS open_loops JSONB DEFAULT '[]'::jsonb,
ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0;

-- Update intent column from TEXT to JSONB
-- Note: This requires data migration for existing rows
ALTER TABLE episodes
ALTER COLUMN intent TYPE JSONB USING
  CASE
    -- If intent is already valid JSON, parse it
    WHEN intent::text ~ '^\s*\{' THEN intent::jsonb
    -- Otherwise, wrap it in a default structure
    ELSE jsonb_build_object(
      'type', 'exploration',
      'direction', 'open-ended',
      'original_text', intent
    )
  END;

-- Update embedding column to proper vector type if using pgvector
-- Uncomment if pgvector is installed:
-- ALTER TABLE episodes ALTER COLUMN embedding TYPE vector(768) USING embedding::vector;

-- Create indexes for new JSONB columns
CREATE INDEX IF NOT EXISTS idx_episodes_salience_factors ON episodes USING gin (salience_factors);
CREATE INDEX IF NOT EXISTS idx_episodes_open_loops ON episodes USING gin (open_loops);

-- Create index on intent type for faster filtering
CREATE INDEX IF NOT EXISTS idx_episodes_intent_type ON episodes ((intent->>'type'));

-- Update existing rows to have default salience_factors if needed
UPDATE episodes
SET salience_factors = jsonb_build_object(
  'novelty', 0.5,
  'emotional', 0.5,
  'commitment', 0.5,
  'unresolved', false
)
WHERE salience_factors = '{}'::jsonb OR salience_factors IS NULL;

-- Ensure all episodes have access_count initialized
UPDATE episodes
SET access_count = 0
WHERE access_count IS NULL;

COMMENT ON COLUMN episodes.intent IS 'Intent structure: {"type": "exploration|...", "direction": "open-ended|..."}';
COMMENT ON COLUMN episodes.context IS 'Context: {"situational": "...", "conversational": "...", "constraints": [...]}';
COMMENT ON COLUMN episodes.salience_factors IS 'LLM-provided factors: {"novelty": 0-1, "emotional": 0-1, "commitment": 0-1, "unresolved": bool}';
COMMENT ON COLUMN episodes.open_loops IS 'Array of unresolved items from the episode';
COMMENT ON COLUMN episodes.salience IS 'Computed salience: w_e·emotional + w_c·commitment + w_n·novelty + w_u·unresolved';
COMMENT ON COLUMN episodes.freshness IS 'Initially salience; computed dynamically at retrieval: Salience × e^(-λΔt)';
