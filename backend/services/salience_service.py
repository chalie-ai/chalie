"""
Salience Service — pure-code salience scoring for episodic memory.

Formula (from episodic-simplification plan):
  emotional  = 0.6 * arousal + 0.4 * abs(valence)   # [0, 1]
  open_boost = 1.0 if has_open_loop else 0.0
  raw        = 0.4 * emotional + 0.4 * novelty + 0.2 * open_boost
  salience   = clamp(round(raw * 10), 1, 10)         # INTEGER 1..10

All weights are declared in episodic_constants.py.
"""


def compute_salience(
    valence: float,
    arousal: float,
    has_open_loop: bool,
    novelty: float,
) -> int:
    """Return an integer salience score in [1, 10].

    Args:
        valence:      Emotional valence in [-1.0, 1.0]. 0 = neutral.
        arousal:      Emotional arousal in [0.0, 1.0]. 0 = calm, 1 = intense.
        has_open_loop: True when the episode ends with an unresolved thread.
        novelty:      Semantic novelty in [0.0, 1.0] vs the apex pool.

    Returns:
        Integer salience score clamped to [1, 10].
    """
    emotional = 0.6 * float(arousal) + 0.4 * abs(float(valence))
    open_boost = 1.0 if has_open_loop else 0.0
    raw = 0.4 * emotional + 0.4 * float(novelty) + 0.2 * open_boost
    return max(1, min(10, int(round(raw * 10))))
