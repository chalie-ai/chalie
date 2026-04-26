"""
Episodic memory tuning constants.

All constants are module-level literals so the meta-harness can diff-patch
them mechanically. Do NOT read these from config/env at import time.
"""

# Pairwise cosine similarity floor for super-episode cluster membership.
# gte-modernbert packs densely (unrelated English ≈ 0.80-0.85); 0.90 is
# the "same topic" floor validated by research on that encoder family.
SUPER_EPISODE_THRESHOLD = 0.90

# Minimum cluster size to fire the SuperEpisodeEncoder LLM call.
SUPER_EPISODE_MIN_CLUSTER = 3

# Number of most-recent apex episodes in the novelty comparison set.
NOVELTY_RECENT_LIMIT = 100

# Number of top-activation apex episodes in the novelty comparison set.
NOVELTY_ACTIVATION_LIMIT = 100

# Cycle-safe depth guard for apex traversal via consolidated_into back-pointers.
APEX_TRAVERSAL_MAX_DEPTH = 20
