-- Drop temporal observation tables (replaced by EngagementSignalService)
DROP INDEX IF EXISTS idx_temporal_obs_type_day_hour;
DROP INDEX IF EXISTS idx_temporal_obs_recorded;
DROP TABLE IF EXISTS temporal_observations;
DROP TABLE IF EXISTS temporal_aggregate;
