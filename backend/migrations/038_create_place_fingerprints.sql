-- Place fingerprints: learned place patterns from ambient telemetry.
-- Uses geohash (~1km precision) for privacy — raw coordinates are never stored.

CREATE TABLE IF NOT EXISTS place_fingerprints (
    id SERIAL PRIMARY KEY,
    fingerprint_hash VARCHAR(64) UNIQUE NOT NULL,
    device_class VARCHAR(16) NOT NULL,
    hour_bucket SMALLINT NOT NULL,           -- 0-7 (3hr blocks)
    location_hash VARCHAR(12),               -- geohash ~1km precision (nullable if no location)
    connection_type VARCHAR(16),
    place_label VARCHAR(16) NOT NULL,        -- home/work/transit/out
    count INTEGER DEFAULT 1,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_place_fp_hash ON place_fingerprints(fingerprint_hash);
