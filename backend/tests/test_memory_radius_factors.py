"""Unit tests for _compute_radius_factors — single-pass narrow + expand helper.

Pure function tests: no DB, no LLM, no mocking required.
"""

import math

import pytest

from abilities.memory import MemoryAbility, _compute_radius_factors

pytestmark = pytest.mark.unit

# Shorthand constants pulled from MemoryAbility so tests stay in sync if
# the meta-harness tunes the values.
NARROW_MIN = MemoryAbility.NARROW_MIN_DIST      # 0.25
NARROW_MAX = MemoryAbility.NARROW_MAX_DIST      # 0.05
NARROW_FLOOR = MemoryAbility.NARROW_FACTOR_FLOOR  # 0.35

EXPAND_MIN = MemoryAbility.EXPAND_MIN_DIST      # 0.30
EXPAND_MAX = MemoryAbility.EXPAND_MAX_DIST      # 0.55
EXPAND_CEIL = MemoryAbility.EXPAND_FACTOR_CEILING  # 2.2


def _unit_vec(dim: int, index: int) -> list:
    """Return a unit vector of length `dim` with 1.0 at `index`."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_entry(embedding) -> dict:
    return {"embedding": embedding}


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------

def test_empty_history_returns_neutral_factors():
    q = _unit_vec(4, 0)
    narrow, expand, min_d, max_d = _compute_radius_factors(q, [])
    assert narrow == 1.0
    assert expand == 1.0
    assert math.isinf(min_d)
    assert max_d == 0.0


# ---------------------------------------------------------------------------
# Entries with empty / None embeddings are skipped
# ---------------------------------------------------------------------------

def test_all_empty_embeddings_treated_as_empty_history():
    q = _unit_vec(4, 0)
    history = [
        _make_entry(None),
        _make_entry([]),
        _make_entry(None),
    ]
    narrow, expand, min_d, max_d = _compute_radius_factors(q, history)
    assert narrow == 1.0
    assert expand == 1.0
    assert math.isinf(min_d)
    assert max_d == 0.0


def test_mixed_valid_and_empty_embeddings_skips_empties():
    """Valid entries drive the result; None/empty entries are ignored."""
    q = _unit_vec(4, 0)
    # orthogonal vector → cosine distance = 1.0 (max)
    far = _unit_vec(4, 1)
    history = [
        _make_entry(None),
        _make_entry(far),
        _make_entry([]),
    ]
    narrow, expand, min_d, max_d = _compute_radius_factors(q, history)
    assert min_d == pytest.approx(1.0, abs=1e-9)
    assert max_d == pytest.approx(1.0, abs=1e-9)
    # min_dist=1.0 >= NARROW_MIN_DIST → narrow_factor = 1.0
    assert narrow == 1.0
    # max_drift=1.0 >= EXPAND_MAX_DIST → expand_factor = EXPAND_CEIL
    assert expand == EXPAND_CEIL


# ---------------------------------------------------------------------------
# Near-duplicate query → narrow factor floor
# ---------------------------------------------------------------------------

def test_near_duplicate_returns_narrow_floor():
    """min_dist <= NARROW_MAX_DIST ⟹ narrow_factor == NARROW_FACTOR_FLOOR."""
    q = [1.0, 0.0]
    # Almost identical: tiny perpendicular component
    tiny = 1e-6
    norm = math.sqrt(1.0 + tiny ** 2)
    near = [1.0 / norm, tiny / norm]
    d = 1.0 - (q[0] * near[0] + q[1] * near[1])
    assert d <= NARROW_MAX, f"expected near-duplicate distance, got {d}"

    history = [_make_entry(near)]
    narrow, _, min_d, _ = _compute_radius_factors(q, history)
    assert min_d == pytest.approx(d, rel=1e-6)
    assert narrow == NARROW_FLOOR


# ---------------------------------------------------------------------------
# Far-drift query → expand factor ceiling
# ---------------------------------------------------------------------------

def test_far_drift_returns_expand_ceiling():
    """max_drift >= EXPAND_MAX_DIST ⟹ expand_factor == EXPAND_FACTOR_CEILING."""
    q = _unit_vec(4, 0)
    far = _unit_vec(4, 1)   # orthogonal → distance = 1.0 > EXPAND_MAX (0.55)

    history = [_make_entry(far)]
    _, expand, _, max_d = _compute_radius_factors(q, history)
    assert max_d == pytest.approx(1.0, abs=1e-9)
    assert expand == EXPAND_CEIL


# ---------------------------------------------------------------------------
# Neutral zones → both factors stay at 1.0
# ---------------------------------------------------------------------------

def test_mid_range_distance_neutral_zone():
    """Distance in (NARROW_MIN, EXPAND_MIN] → both factors == 1.0."""
    # Pick a distance between 0.25 and 0.30: e.g. 0.27
    target_sim = 1.0 - 0.27
    # Build a 2-D vector with cosine similarity == target_sim to [1,0]
    # sim = a[0] / |a| → a = [target_sim, sqrt(1 - target_sim^2)]
    a0 = target_sim
    a1 = math.sqrt(max(0.0, 1.0 - a0 ** 2))
    entry_vec = [a0, a1]

    q = [1.0, 0.0]
    history = [_make_entry(entry_vec)]
    narrow, expand, min_d, max_d = _compute_radius_factors(q, history)
    assert min_d == pytest.approx(0.27, abs=1e-6)
    assert max_d == pytest.approx(0.27, abs=1e-6)
    assert narrow == 1.0
    assert expand == 1.0


# ---------------------------------------------------------------------------
# Mixed case: small min_dist AND large max_drift ⟹ floor × ceiling
# ---------------------------------------------------------------------------

def test_mixed_near_and_far_returns_floor_and_ceiling():
    """With one near entry and one far entry, get FLOOR and CEILING together."""
    q = [1.0, 0.0]

    # near: distance ~ 0 (identical)
    near = [1.0, 0.0]

    # far: orthogonal → distance = 1.0
    far = [0.0, 1.0]

    history = [_make_entry(near), _make_entry(far)]
    narrow, expand, min_d, max_d = _compute_radius_factors(q, history)

    assert min_d == pytest.approx(0.0, abs=1e-9)
    assert max_d == pytest.approx(1.0, abs=1e-9)
    assert narrow == NARROW_FLOOR
    assert expand == EXPAND_CEIL


# ---------------------------------------------------------------------------
# Interpolated narrow factor (midpoint check)
# ---------------------------------------------------------------------------

def test_interpolated_narrow_factor():
    """Distance midway between NARROW_MAX and NARROW_MIN ⟹ interpolated factor."""
    mid = (NARROW_MAX + NARROW_MIN) / 2.0  # 0.15
    a0 = 1.0 - mid
    a1 = math.sqrt(max(0.0, 1.0 - a0 ** 2))
    entry_vec = [a0, a1]

    q = [1.0, 0.0]
    history = [_make_entry(entry_vec)]
    narrow, _, min_d, _ = _compute_radius_factors(q, history)

    assert min_d == pytest.approx(mid, abs=1e-6)
    span = NARROW_MIN - NARROW_MAX
    t = (NARROW_MIN - mid) / span
    expected = max(NARROW_FLOOR, min(1.0, 1.0 - t * (1.0 - NARROW_FLOOR)))
    assert narrow == pytest.approx(expected, rel=1e-9)
    assert NARROW_FLOOR <= narrow <= 1.0


# ---------------------------------------------------------------------------
# Interpolated expand factor (midpoint check)
# ---------------------------------------------------------------------------

def test_interpolated_expand_factor():
    """Distance midway between EXPAND_MIN and EXPAND_MAX ⟹ interpolated factor."""
    mid = (EXPAND_MIN + EXPAND_MAX) / 2.0  # 0.425
    a0 = 1.0 - mid
    a1 = math.sqrt(max(0.0, 1.0 - a0 ** 2))
    entry_vec = [a0, a1]

    q = [1.0, 0.0]
    history = [_make_entry(entry_vec)]
    _, expand, _, max_d = _compute_radius_factors(q, history)

    assert max_d == pytest.approx(mid, abs=1e-6)
    span = EXPAND_MAX - EXPAND_MIN
    t = (mid - EXPAND_MIN) / span
    expected = min(EXPAND_CEIL, max(1.0, 1.0 + t * (EXPAND_CEIL - 1.0)))
    assert expand == pytest.approx(expected, rel=1e-9)
    assert 1.0 <= expand <= EXPAND_CEIL
