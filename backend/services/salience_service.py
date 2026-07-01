"""
Salience Service — pure-code salience scoring for episodic memory.

Formula:
  open_boost = 1.0 if has_open_loop else 0.0
  raw        = SALIENCE_NOVELTY_WEIGHT * novelty + SALIENCE_OPEN_LOOP_WEIGHT * open_boost
  salience   = clamp(round(raw * 10), 1, 10)   # INTEGER 1..10

Weights are declared in episodic_constants.py.
"""

from services.episodic_constants import SALIENCE_NOVELTY_WEIGHT, SALIENCE_OPEN_LOOP_WEIGHT


def compute_salience(has_open_loop: bool, novelty: float) -> int:
    open_boost = 1.0 if has_open_loop else 0.0
    raw = SALIENCE_NOVELTY_WEIGHT * float(novelty) + SALIENCE_OPEN_LOOP_WEIGHT * open_boost
    return max(1, min(10, int(round(raw * 10))))
