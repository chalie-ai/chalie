"""
Episodic memory tuning constants.

All constants are module-level literals so they can be diff-patched
mechanically. Do NOT read these from config/env at import time.
"""

# ── Hierarchy roll-up (count-triggered density clustering) ─────────────────────
# The roll-up fires on a per-channel apex COUNT, not a similarity gate: a count
# trigger always eventually runs, whereas the old 0.90-cosine gate never did.

# Apex-leaf count that triggers a leaf (level-0 → level-1) roll-up on a channel.
APEX_COUNT_TRIGGER = 50

# Level-1 apex count that triggers a recursive (level-1 → level-2) era roll-up.
ERA_DIGEST_TRIGGER = 25

# HDBSCAN min_cluster_size — the smallest group HDBSCAN will report as a cluster.
# Also the survivor floor below which an emitted candidate group is dropped. 10 is
# valid for both rounds (it is <= ERA_DIGEST_TRIGGER, so eras can still form).
HDBSCAN_MIN_CLUSTER_SIZE = 10

# UMAP reduction (mandatory — raw 768-d HDBSCAN degenerates to one blob, PCA is
# degenerate at every dim). Cosine metric matches the embedder's geometry.
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
# Cosine-distance threshold above which a point is disconnected from the manifold
# instead of force-embedded into the nearest blob. 1.0 == cosine similarity 0:
# only points with NO positive similarity to anything (genuine, off-topic
# outliers) detach and fall through as HDBSCAN noise. The encoder packs related
# content densely (unrelated English still ~0.8 cosine), so this never fires on
# real, even loosely-related episodes — it only catches true isolates.
UMAP_DISCONNECTION_DISTANCE = 1.0
# Pinned seed: roll-ups must be reproducible (deterministic cluster membership).
# UMAP forces single-thread when random_state is set — accepted for determinism;
# the cost is sub-second at our ~1k-apex scale.
UMAP_RANDOM_SEED = 42

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
# below this the seed stays a lone apex. Same floor role as HDBSCAN_MIN_CLUSTER_SIZE.
SEED_CLUSTER_MIN_SIZE = 10

# Cosine-distance cutoff (vector_distance <=) for neighbour inclusion. 0.0 ==
# identical, 1.0 == maximally dissimilar (orthogonal). Start wide/lenient; this
# is a PROVISIONAL value to be calibrated against live embeddings — do NOT use
# it as a hard semantic boundary without empirical validation.
SEED_CLUSTER_RADIUS = 0.45

# Raw KNN hits to request from Episode.nearest before filtering. sqlite-vec
# applies the `k` limit BEFORE the JOIN/row filters, so requesting exactly 25
# would return the 25 nearest rows OVERALL (including non-apex rows, other
# levels, and the seed itself), then filtering to apex+same-level could leave
# far fewer than 25 — a silent under-count. Over-fetch then filter in Python
# to guarantee we don't drop below SEED_CLUSTER_MAX_MATCHES qualifying neighbours.
SEED_CLUSTER_KNN_OVERFETCH = 200
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
