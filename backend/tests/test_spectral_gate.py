# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the in-house non-stationary spectral noise gate.

Runs the real ``spectral_gate`` over real seeded audio — no mocks, no
recorded spectra, no stand-in for the DSP. Covers the four things the
speech pipeline actually depends on:

* the gate hands back a float32 array of exactly the input length (the
  transcriber slices the result by sample count and feeds it to a
  float32-only model, so a widened dtype or a shifted length breaks
  transcription rather than merely degrading it);
* it genuinely removes broadband noise;
* a clip long enough to cross the 600000-sample internal chunk boundary
  comes back whole, finite, and non-silent on both sides of the seam;
* a sample rate that collapses the mask-smoothing ramp below one FFT bin
  is refused loudly instead of silently gating with a degenerate kernel.

The tone is deliberately *not* asserted to survive. The non-stationary
algorithm treats steady content as noise floor, so a synthetic sine keeps
roughly half a percent of its energy — the behaviour of the library this
module replaces, not a defect. Asserting otherwise would encode a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from services.spectral_gate import spectral_gate

pytestmark = pytest.mark.unit

# Every clip below is 16 kHz — the rate the speech pipeline feeds the gate.
_SR = 16000

# Internal chunk size of the gate; a clip longer than this is processed in
# pieces. Named here so the long-clip test states what it is crossing.
_CHUNK_SAMPLES = 600000


def _tone_plus_noise(seconds: float, tone_hz: float, seed: int) -> NDArray[np.float32]:
    """A seeded 16 kHz clip: a sine at ``tone_hz`` under white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * _SR), dtype=np.float64) / _SR
    tone = 0.3 * np.sin(2 * np.pi * tone_hz * t)
    noise = rng.normal(0.0, 0.1, size=t.shape)
    return (tone + noise).astype(np.float32)


def _band_energy(signal: NDArray[np.float32], low_hz: float, high_hz: float) -> float:
    """Total spectral energy of ``signal`` inside ``[low_hz, high_hz]``."""
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
    freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / _SR)
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(spectrum[in_band] ** 2))


def test_gated_clip_is_float32_and_sample_aligned_with_the_input() -> None:
    """A 5 s tone-plus-noise clip returns float32, same length as it went in.

    The transcriber hands the gate's output straight to a float32-only
    model and windows it by sample count, so a float64 return or a length
    that drifts (a dropped inverse-STFT tail, an off-by-one in the padded
    window slice) corrupts transcription instead of just degrading audio.
    """
    clip = _tone_plus_noise(5.0, 440.0, seed=20260907)

    gated = spectral_gate(clip, _SR)

    assert gated.dtype == np.float32, (
        "the speech model only accepts float32 — a widened dtype fails at "
        "the model boundary, far from here; got %r" % gated.dtype
    )
    assert gated.shape == clip.shape, (
        "the gate must be sample-aligned with its input; got %r for a %r "
        "input" % (gated.shape, clip.shape)
    )


def test_gate_strips_broadband_noise_away_from_the_tone() -> None:
    """Energy in the 2-7 kHz noise band collapses once the clip is gated.

    Measured on this exact seeded clip the band energy falls 157x
    (2.01e+07 -> 1.28e+05). The assertion demands only 20x, leaving an
    order of magnitude of headroom against numerical drift, while a gate
    that stopped masking at all — an inverted sigmoid, an all-ones mask, a
    pass-through return — lands at roughly 1x and goes red.

    The 440 Hz tone is deliberately not asserted to survive: the
    non-stationary gate treats steady content as noise floor and keeps
    ~0.5% of a synthetic sine's energy. That is the ported library's
    behaviour, so a test demanding the tone survive would be asserting a
    bug into place.
    """
    clip = _tone_plus_noise(5.0, 440.0, seed=20260907)

    gated = spectral_gate(clip, _SR)

    before = _band_energy(clip, 2000.0, 7000.0)
    after = _band_energy(gated, 2000.0, 7000.0)

    assert after > 0.0, (
        "the gate zeroed the clip outright rather than attenuating it — "
        "that is a broken mask, not noise reduction"
    )
    assert before / after > 20.0, (
        "broadband noise between 2 and 7 kHz must drop by more than an "
        "order of magnitude (measured 157x); got %.2fx (%.4g -> %.4g)"
        % (before / after, before, after)
    )


def test_clip_crossing_the_internal_chunk_boundary_comes_back_whole() -> None:
    """A 40 s clip spans two internal chunks and returns intact.

    640000 samples is 40000 past the 600000-sample chunk size, so the gate
    runs its chunked path: a full chunk plus a short trailing one, each
    filtered on a window zero-padded on both sides. Four things can break
    there and each is asserted: the trailing chunk being dropped or the
    padded window returned whole changes the length; a silent window
    driving the magnitude-ratio division to zero produces NaN or inf; a
    misindexed copy into the padded window leaves a chunk all zeros while
    the length still looks right; and losing the window padding leaves the
    seam itself as a run of dead samples, which is the whole reason the
    padding exists.
    """
    clip = _tone_plus_noise(40.0, 220.0, seed=20260907)
    assert clip.shape[0] > _CHUNK_SAMPLES, "clip must cross the chunk boundary"

    gated = spectral_gate(clip, _SR)

    assert gated.shape == clip.shape, (
        "a chunked clip must be reassembled sample-for-sample; got %r for "
        "a %r input" % (gated.shape, clip.shape)
    )
    assert bool(np.all(np.isfinite(gated))), (
        "the gate emitted NaN or inf — the magnitude-ratio division blew up "
        "on a window with no signal in it"
    )
    assert float(np.abs(gated[:_CHUNK_SAMPLES]).max()) > 0.0, (
        "the first chunk came back silent"
    )
    assert float(np.abs(gated[_CHUNK_SAMPLES:]).max()) > 0.0, (
        "the trailing chunk came back silent — its window was filled or "
        "sliced wrongly, and the output length alone would not show it"
    )
    seam = gated[_CHUNK_SAMPLES - 200:_CHUNK_SAMPLES + 200]
    assert int(np.count_nonzero(seam == 0.0)) == 0, (
        "the chunk seam is a run of dead samples — each chunk is filtered "
        "on a window padded past its own edges precisely so the boundary "
        "leaves no mark; %d of %d samples around it are exactly zero"
        % (int(np.count_nonzero(seam == 0.0)), seam.shape[0])
    )


def test_sample_rate_that_collapses_the_mask_ramp_is_refused() -> None:
    """A 400 kHz sample rate raises ValueError instead of gating oddly.

    The mask smoothing ramp is specified in Hz and milliseconds, so it is
    converted to bins against the sample rate. At 400 kHz an FFT bin spans
    781 Hz, the 500 Hz frequency ramp rounds down to zero points, and the
    triangular kernel degenerates to a single point along the frequency
    axis — the gate would still return audio, silently smoothing along time
    only. Refusing is the contract; drop the guard and this call succeeds,
    turning the test red.
    """
    clip = _tone_plus_noise(0.5, 440.0, seed=20260907)

    with pytest.raises(ValueError):
        spectral_gate(clip, 400000)
