-- Migration 037: Data backfills
--
-- Safety net for two backfills that were previously inline in
-- DatabaseService.run_pending_migrations().  On databases where these
-- already ran the UPDATE WHERE clauses produce zero rows.

-- Backfill providers.models from providers.model (single → array)
UPDATE providers SET models = json_array(model)
WHERE models IS NULL AND model IS NOT NULL;

-- Backfill episode storage_strength/retrieval_weight from activation_score
-- (only for episodes that still have the legacy default of 1.0)
UPDATE episodes
SET storage_strength  = MAX(0.1, activation_score),
    retrieval_weight  = MAX(0.1, activation_score)
WHERE activation_score IS NOT NULL
  AND activation_score != 1.0
  AND storage_strength = 1.0;
