"""
Episodic memory tuning constants.

All constants are module-level literals so they can be diff-patched
mechanically. Do NOT read these from config/env at import time.
"""

# Number of most-recent apex episodes in the novelty comparison set.
NOVELTY_RECENT_LIMIT = 100

# Number of top-activation apex episodes in the novelty comparison set.
NOVELTY_ACTIVATION_LIMIT = 100
# Cycle-safe depth guard for apex traversal via consolidated_into back-pointers.
APEX_TRAVERSAL_MAX_DEPTH = 20
# Maximum hierarchy depth an episode may reach. A seed at level L consolidates
# into a parent at L+1 only while L+1 <= MAX_EPISODE_LEVEL, so leaves (L0) roll
# up to L1 and L1s roll up to L2; L2 is terminal. Mirrors the batch path, which
# only ever wrote levels 1 (leaf round) and 2 (era round).
MAX_EPISODE_LEVEL = 2

# ── Seed-on-creation clustering ────────────────────────────────────────────
# One new episode seeds a local KNN neighbourhood instead of reclustering the
# whole apex pool. See find_seed_cluster() in episodic_service.py for the full
# candidate-discovery primitive.

# Top-K nearest apex neighbours a seed may pull into its cluster.
SEED_CLUSTER_MAX_MATCHES = 25

# Minimum members (seed + qualifying neighbours) required to form a cluster;
# below this the seed stays a lone apex. This is the single minimum-cluster-size floor for consolidation.
SEED_CLUSTER_MIN_SIZE = 10

# Cosine-distance cutoff (vector_distance <=) for neighbour inclusion. 0.0 ==
# identical, 1.0 == maximally dissimilar (orthogonal). In real embedding data
# related episodes sit well inside this cutoff and unrelated ones well beyond
# it, so 0.45 has margin on both sides — but it is a similarity heuristic, not
# a hard semantic boundary.
SEED_CLUSTER_RADIUS = 0.45
# ── Window extraction (count-triggered episode encoding) ───────────────────────
# Turn-end fires episode extraction once a channel accumulates EXTRACTION_THRESHOLD
# transcript rows past its episode watermark. The encoder then reads the latest
# EXTRACTION_WINDOW rows (threshold + overlap with the prior window), producing
# one episode per ~EXTRACTION_THRESHOLD new turns.
EXTRACTION_THRESHOLD = 20
EXTRACTION_WINDOW = 25

# ── Salience scoring weights (compute_salience) ───────────────────────────────
# raw = SALIENCE_NOVELTY_WEIGHT * novelty + SALIENCE_OPEN_LOOP_WEIGHT * open_boost,
# then clamped to an integer 1..10. Emotional valence/arousal were dropped from
# the formula; novelty and the open-loop flag are the only surviving signals,
# rescaled to sum to 1.0 with the prior 2:1 novelty:open-loop ratio preserved.
SALIENCE_NOVELTY_WEIGHT = 0.6
SALIENCE_OPEN_LOOP_WEIGHT = 0.4
