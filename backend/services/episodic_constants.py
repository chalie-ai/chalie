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
