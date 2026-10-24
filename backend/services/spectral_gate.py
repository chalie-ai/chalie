# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""In-house, bit-identical port of the non-stationary spectral gate from
noisereduce (https://github.com/timsainb/noisereduce).

``spectral_gate`` reproduces exactly what
``noisereduce.reduce_noise(y, sr, stationary=False)`` (noisereduce 3.0.3,
default arguments) applies to a 1-D float32 audio array. Per (zero-padded)
chunk:

  1. STFT (n_fft=1024, hop=256);
  2. one-pole low-pass smoothing of the magnitude along time (2 s time
     constant), ``filtfilt`` with ``padtype=None``;
  3. steep sigmoid (slope 10) on the magnitude/smooth ratio around 2x;
  4. triangular-ramp smoothing of the mask (500 Hz / 50 ms);
  5. inverse STFT of the masked spectrum, zero-padded back to the chunk
     size.

Clips longer than 600000 samples are processed in 600000-sample chunks, each
filtered on a window zero-padded by 30000 samples on both sides, so no chunk
boundary is visible in the output.

The gate algorithm is a port of code from noisereduce, released under:

The MIT License (MIT)

Copyright (c) 2019, Tim Sainburg
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import scipy.signal
from numpy.typing import NDArray

# noisereduce 3.0.3 defaults (reduce_noise, stationary=False). These values
# are what make the output bit-identical to the original; do not tune them
# here.
_N_FFT = 1024
_WIN_LENGTH = 1024
_HOP_LENGTH = 256
_TIME_CONSTANT_S = 2.0
_FREQ_MASK_SMOOTH_HZ = 500
_TIME_MASK_SMOOTH_MS = 50
_THRESH_N_MULT = 2
_SIGMOID_SLOPE = 10
_CHUNK_SIZE = 600000
_PADDING = 30000


def _gate_window(x: NDArray[np.float64], sr: int) -> NDArray[np.float64]:
    """Gate one already zero-padded float64 window.

    Returns a float64 array of the same length: the gated signal (inverse
    STFT of the masked spectrum) written at [0 : len(y)], zero-padded to
    len(x).
    """
    S: NDArray[np.complex128]
    _, _, S = scipy.signal.stft(
        x, nfft=_N_FFT, noverlap=_N_FFT - _HOP_LENGTH, nperseg=_WIN_LENGTH, padded=False
    )
    A = np.abs(S)

    t_frames = _TIME_CONSTANT_S * sr / _HOP_LENGTH
    b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
    smooth: NDArray[np.float64]
    smooth = scipy.signal.filtfilt([b], [1, b - 1], A, axis=-1, padtype=None)

    mult = (A - smooth) / smooth
    mask: NDArray[np.float64]
    mask = 1 / (1 + np.exp(-(mult - _THRESH_N_MULT) * _SIGMOID_SLOPE))

    n_grad_freq = int(_FREQ_MASK_SMOOTH_HZ / (sr / (_N_FFT / 2)))
    n_grad_time = int(_TIME_MASK_SMOOTH_MS / ((_HOP_LENGTH / sr) * 1000))
    if n_grad_freq < 1 or n_grad_time < 1:
        raise ValueError(
            "freq_mask_smooth_hz and time_mask_smooth_ms are too small for "
            "this sample rate: the mask-smoothing ramp needs at least one "
            "point along each axis"
        )
    if n_grad_freq != 1 or n_grad_time != 1:
        filt = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        filt = filt / np.sum(filt)
        mask = scipy.signal.fftconvolve(mask, filt, mode="same")

    # prop_decrease is 1.0: the mask * 1.0 + ones * 0.0 step is a no-op (left out).
    y: NDArray[np.float64]
    _, y = scipy.signal.istft(
        S * mask, nfft=_N_FFT, noverlap=_N_FFT - _HOP_LENGTH, nperseg=_WIN_LENGTH
    )

    out = np.zeros(len(x), dtype=np.float64)
    out[0 : len(y)] = y
    return out


def _gate_chunk(
    audio: NDArray[np.float32],
    n: int,
    start: int,
    end: int,
    filter_end: int,
    sr: int,
) -> NDArray[np.float64]:
    """Gate frames [start, end) on the zero-padded window
    [start - _PADDING, filter_end + _PADDING).

    The window is a float64 zeros array of length
    (filter_end - start) + 2 * _PADDING into which the in-range part of the
    signal is copied; only the slice [_PADDING : _PADDING + (end - start)] of
    the gated window is returned. For the last chunk of a chunked input,
    ``filter_end`` is a full _CHUNK_SIZE past ``n`` (zero-padded past the end
    of the clip) while ``end`` is clamped to ``n``.
    """
    window = np.zeros((filter_end - start) + 2 * _PADDING, dtype=np.float64)
    copy_start = max(0, start - _PADDING)
    copy_end = min(n, filter_end + _PADDING)
    offset = _PADDING - start
    window[copy_start + offset : copy_end + offset] = audio[copy_start:copy_end]
    gated = _gate_window(window, sr)
    return gated[_PADDING : _PADDING + (end - start)]


def spectral_gate(audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
    """Apply the non-stationary spectral gate to a 1-D float32 clip.

    Bit-identical port of the gate that
    ``noisereduce.reduce_noise(y, sr, stationary=False)`` (noisereduce 3.0.3,
    default arguments) applies. Clips longer than 600000 samples are
    processed in 600000-sample chunks, each filtered on a window zero-padded
    by 30000 samples on both sides (the last chunk is still filtered on a
    full 600000-sample window, zero-padded past the end of the clip), so no
    chunk boundary is visible in the output.

    Returns a float32 array of the same length as ``audio``.
    """
    n = len(audio)
    pieces: list[NDArray[np.float64]] = []
    if n <= _CHUNK_SIZE:
        pieces.append(_gate_chunk(audio, n, 0, n, n, sr))
    else:
        for ich in range(0, (n - 1) // _CHUNK_SIZE + 1):
            start = ich * _CHUNK_SIZE
            full_end = (ich + 1) * _CHUNK_SIZE
            pieces.append(
                _gate_chunk(audio, n, start, min(n, full_end), full_end, sr)
            )
    return np.concatenate(pieces).astype(np.float32)
