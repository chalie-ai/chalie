/**
 * VoicePlayer — per-message streaming audio overlay.
 *
 * Listens for `chalie:speak-message` dispatched by renderer.js speaker
 * buttons. On each event it POSTs /voice/synthesize (which returns
 * `{ok, total}` immediately), then consumes `chalie:tts-chunk` /
 * `chalie:tts-done` CustomEvents forwarded by event_router.js from the
 * `output:events` pub/sub channel.
 *
 * Each tts_chunk carries a base64-encoded WAV. The player decodes via
 * AudioContext and appends the resulting AudioBuffer to a queue; buffers
 * play in order by chaining source.onended → _playNextChunk.
 *
 * iOS Safari autoplay: AudioContext.resume() runs synchronously inside
 * the `chalie:speak-message` handler, which is itself dispatched inside
 * the click gesture. That satisfies the user-activation requirement.
 *
 * Only one session plays at a time — opening a new message supersedes the
 * previous stream via `_openGeneration`.
 */

import { createWakeLock } from './utils.js';

const _TTS_PATH = '/voice/synthesize';
const _SKIP_SECONDS = 10;

export class VoicePlayer {
  /**
   * @param {{ getHost: () => string }} opts
   */
  constructor({ getHost }) {
    this._getHost = getHost;

    this._overlay = null;
    this._playBtn = null;
    this._bkBtn = null;
    this._ffBtn = null;
    this._closeBtn = null;
    this._progress = null;
    this._timeEl = null;
    this._errorEl = null;
    this._loadingEl = null;
    this._controlsEl = null;

    this._boundKeydown = null;

    // Generation counter so stale chunk events from a previous session
    // can be discarded. Incremented on every _open() and _close().
    this._openGeneration = 0;
    this._fetchAbort = null;

    // AudioContext — created lazily on first open, reused after.
    this._audioCtx = null;

    // Per-session playback state
    this._bufferQueue = [];
    this._chunkTexts = [];
    this._chunkIdx = 0;
    this._expectedTotal = 0;
    this._streamDone = false;
    this._cumulativeTime = 0;
    this._currentSource = null;
    this._currentBuffer = null;
    this._sourceStartCtxTime = 0;
    this._sourceStartOffset = 0;
    this._paused = false;
    this._pausedOffset = 0;
    this._decodeChain = Promise.resolve();
    this._progressTimer = null;
    this._wakeLock = createWakeLock();
  }

  /** Bind overlay DOM + listeners. Must be called after DOM ready. */
  init() {
    this._overlay = document.getElementById('voicePlayerOverlay');
    if (!this._overlay) return;

    this._playBtn = this._overlay.querySelector('#vpPlayBtn');
    this._bkBtn = this._overlay.querySelector('#vpBkBtn');
    this._ffBtn = this._overlay.querySelector('#vpFfBtn');
    this._closeBtn = this._overlay.querySelector('#vpCloseBtn');
    this._progress = this._overlay.querySelector('#vpProgress');
    this._timeEl = this._overlay.querySelector('#vpTime');
    this._errorEl = this._overlay.querySelector('#vpError');
    this._loadingEl = this._overlay.querySelector('#vpLoading');
    this._controlsEl = this._overlay.querySelector('.voice-player__controls');

    this._playBtn?.addEventListener('click', () => this._togglePlayPause());
    this._bkBtn?.addEventListener('click', () => this._skipBack());
    this._ffBtn?.addEventListener('click', () => this._skipForward());
    this._closeBtn?.addEventListener('click', () => this._close());

    // Backdrop click closes
    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this._close();
    });

    // Scrubber seeks within the current chunk only. Cross-chunk seeking
    // isn't worth the complexity for a speaker-button UX.
    this._progress?.addEventListener('input', () => {
      if (!this._currentBuffer) return;
      const offset = Number.parseFloat(this._progress.value) || 0;
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, offset);
    });

    // Speaker-button click fires here; handler runs inside the user gesture
    // so AudioContext.resume() is permitted on iOS Safari.
    document.addEventListener('chalie:speak-message', (e) => {
      const { text } = e.detail || {};
      if (text) this._open(text);
    });

    document.addEventListener('chalie:tts-chunk', (e) => this._handleChunk(e.detail));
    document.addEventListener('chalie:tts-done', () => this._handleDone());
  }

  // ---------------------------------------------------------------------------
  // Private — session lifecycle
  // ---------------------------------------------------------------------------

  async _open(text) {
    // New session supersedes any prior one.
    this._openGeneration += 1;
    const gen = this._openGeneration;

    // Abort any in-flight fetch from a previous open.
    if (this._fetchAbort) this._fetchAbort.abort();
    this._fetchAbort = new AbortController();

    // Stop prior playback, reset state (keeps AudioContext alive).
    this._resetPlayback();

    // Unlock / create AudioContext synchronously inside the gesture.
    if (!this._audioCtx) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      this._audioCtx = new Ctor();
    }
    if (this._audioCtx.state === 'suspended') {
      this._audioCtx.resume().catch(() => {});
    }

    this._showOverlay();
    this._showLoading(true);
    this._showError(null);

    try {
      const host = this._getHost?.() || '';
      const url = host.replace(/\/$/, '') + _TTS_PATH;

      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ text }),
        signal: this._fetchAbort.signal,
      });

      if (gen !== this._openGeneration) return;

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        if (gen !== this._openGeneration) return;
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const body = await resp.json();
      if (gen !== this._openGeneration) return;

      this._expectedTotal = body?.total || 0;
      // Chunks arrive asynchronously via chalie:tts-chunk events.
      // Keep the loading overlay up until the first buffer is decoded.
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (gen !== this._openGeneration) return;
      this._showLoading(false);
      this._showError(err.message || 'Failed to start synthesis');
      console.error('[VoicePlayer] synthesize error:', err);
    }

    this._bindKeyboard();
  }

  _close() {
    this._openGeneration += 1; // Discard any in-flight chunks
    if (this._fetchAbort) this._fetchAbort.abort();
    this._resetPlayback();
    if (this._overlay?.open) this._overlay.close();
    this._unbindKeyboard();
  }

  _resetPlayback() {
    this._stopCurrentSource();
    this._stopProgressTimer();
    this._bufferQueue = [];
    this._chunkTexts = [];
    this._chunkIdx = 0;
    this._expectedTotal = 0;
    this._streamDone = false;
    this._cumulativeTime = 0;
    this._currentBuffer = null;
    this._sourceStartCtxTime = 0;
    this._sourceStartOffset = 0;
    this._paused = false;
    this._pausedOffset = 0;
    this._decodeChain = Promise.resolve();
    this._setPlayIcon(false);
    if (this._progress) { this._progress.max = 0; this._progress.value = 0; }
    if (this._timeEl) this._timeEl.textContent = '0:00 / 0:00';
    this._showLoading(false);
    this._wakeLock.release();
  }

  // ---------------------------------------------------------------------------
  // Private — chunk handling
  // ---------------------------------------------------------------------------

  _handleChunk(data) {
    if (!data || !this._audioCtx) return;
    if (!this._overlay?.open) return;

    const audioB64 = data.audio || '';
    if (!audioB64) return;

    const raw = atob(audioB64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    const text = data.text || '';
    const gen = this._openGeneration;

    // Serialize decodes so buffers land in arrival order. Safari uses a
    // callback-style decodeAudioData, so we wrap in a Promise.
    this._decodeChain = this._decodeChain.then(() => new Promise((resolve) => {
      if (gen !== this._openGeneration || !this._audioCtx) { resolve(); return; }
      this._audioCtx.decodeAudioData(
        bytes.buffer,
        (buffer) => {
          if (gen !== this._openGeneration) { resolve(); return; }
          this._bufferQueue.push(buffer);
          this._chunkTexts.push(text);
          // Start playback as soon as the first buffer decodes.
          if (!this._currentSource && !this._paused
              && this._chunkIdx < this._bufferQueue.length) {
            this._showLoading(false);
            this._playNextChunk();
          }
          resolve();
        },
        (err) => {
          console.error('[VoicePlayer] decodeAudioData failed:', err);
          resolve();
        },
      );
    }));
  }

  _handleDone() {
    if (!this._overlay?.open) return;
    this._streamDone = true;
    this._checkStreamDone();
  }

  _checkStreamDone() {
    if (this._streamDone
        && this._chunkIdx >= this._bufferQueue.length
        && !this._currentSource) {
      // Playback complete
      this._setPlayIcon(false);
      if (this._progress && this._currentBuffer) {
        this._progress.value = this._currentBuffer.duration || 0;
      }
      this._wakeLock.release();
    }
  }

  // ---------------------------------------------------------------------------
  // Private — playback
  // ---------------------------------------------------------------------------

  _playNextChunk() {
    if (this._chunkIdx >= this._bufferQueue.length) {
      this._checkStreamDone();
      return;
    }

    const buffer = this._bufferQueue[this._chunkIdx];
    this._currentBuffer = buffer;
    this._paused = false;
    this._pausedOffset = 0;

    if (this._progress) {
      this._progress.max = buffer.duration || 0;
      this._progress.value = 0;
    }

    this._playBufferFrom(buffer, 0);
  }

  _playBufferFrom(buffer, offset) {
    if (!this._audioCtx) return;

    const source = this._audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(this._audioCtx.destination);

    source.onended = () => {
      if (source !== this._currentSource) return;
      this._currentSource = null;
      if (this._paused) return;
      this._cumulativeTime += buffer.duration;
      this._chunkIdx += 1;
      this._playNextChunk();
    };

    try {
      source.start(0, offset);
    } catch (err) {
      console.error('[VoicePlayer] source.start failed:', err);
      this._showError('Playback error');
      return;
    }

    this._currentSource = source;
    this._currentBuffer = buffer;
    this._sourceStartCtxTime = this._audioCtx.currentTime;
    this._sourceStartOffset = offset;
    this._paused = false;
    this._setPlayIcon(true);
    this._startProgressTimer();
    this._wakeLock.acquire();
  }

  _stopCurrentSource() {
    if (!this._currentSource) return;
    this._currentSource.onended = null;
    try { this._currentSource.stop(); } catch (e) { console.warn('[VoicePlayer] source.stop failed:', e); }
    try { this._currentSource.disconnect(); } catch (e) { console.warn('[VoicePlayer] source.disconnect failed:', e); }
    this._currentSource = null;
  }

  _currentPosition() {
    if (this._paused) return this._pausedOffset;
    if (!this._currentSource || !this._audioCtx) return 0;
    return this._sourceStartOffset + (this._audioCtx.currentTime - this._sourceStartCtxTime);
  }

  _togglePlayPause() {
    if (!this._audioCtx) return;
    if (this._paused) {
      if (!this._currentBuffer) return;
      this._playBufferFrom(this._currentBuffer, this._pausedOffset);
      return;
    }
    if (!this._currentSource) {
      // Nothing playing — maybe we finished. Restart from first buffer.
      if (this._bufferQueue.length) {
        this._chunkIdx = 0;
        this._cumulativeTime = 0;
        this._playNextChunk();
      }
      return;
    }
    this._pausedOffset = this._currentPosition();
    this._stopCurrentSource();
    this._paused = true;
    this._setPlayIcon(false);
    this._stopProgressTimer();
    this._wakeLock.release();
  }

  _skipForward() {
    if (!this._currentBuffer || !this._audioCtx) return;
    const pos = this._currentPosition();
    const target = pos + _SKIP_SECONDS;
    if (target < this._currentBuffer.duration) {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, target);
      return;
    }
    // Past end of current chunk → jump to next
    this._cumulativeTime += this._currentBuffer.duration;
    this._stopCurrentSource();
    this._chunkIdx += 1;
    this._playNextChunk();
  }

  _skipBack() {
    if (!this._currentBuffer || !this._audioCtx) return;
    const pos = this._currentPosition();
    const target = pos - _SKIP_SECONDS;
    if (target > 0) {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, target);
      return;
    }
    if (pos > 2) {
      // Restart current chunk
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, 0);
      return;
    }
    if (this._chunkIdx > 0) {
      this._stopCurrentSource();
      this._chunkIdx -= 1;
      // Recompute cumulative — simpler to leave as-is; display handles it.
      this._playNextChunk();
    } else {
      this._stopCurrentSource();
      this._playBufferFrom(this._currentBuffer, 0);
    }
  }

  // ---------------------------------------------------------------------------
  // Private — UI
  // ---------------------------------------------------------------------------

  _showOverlay() {
    if (this._overlay && !this._overlay.open) this._overlay.show();
  }

  _showLoading(loading) {
    this._loadingEl?.classList.toggle('hidden', !loading);
    this._controlsEl?.classList.toggle('hidden', loading);
  }

  _showError(msg) {
    if (!this._errorEl) return;
    if (msg) {
      this._errorEl.textContent = msg;
      this._errorEl.classList.remove('hidden');
    } else {
      this._errorEl.textContent = '';
      this._errorEl.classList.add('hidden');
    }
  }

  _setPlayIcon(playing) {
    if (!this._playBtn) return;
    this._playBtn.innerHTML = playing
      ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <rect x="6" y="4" width="4" height="16"></rect>
           <rect x="14" y="4" width="4" height="16"></rect>
         </svg>`
      : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <polygon points="5 3 19 12 5 21 5 3"></polygon>
         </svg>`;
    this._playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }

  _startProgressTimer() {
    this._stopProgressTimer();
    this._progressTimer = setInterval(() => this._updateProgress(), 200);
  }

  _stopProgressTimer() {
    if (this._progressTimer) {
      clearInterval(this._progressTimer);
      this._progressTimer = null;
    }
  }

  _updateProgress() {
    const buf = this._currentBuffer;
    if (!buf) return;
    const cur = this._currentPosition();
    const dur = buf.duration || 0;
    if (this._progress && !Number.isNaN(dur)) {
      this._progress.max = dur;
      this._progress.value = Math.min(cur, dur);
    }
    if (this._timeEl) {
      this._timeEl.textContent = `${this._fmt(cur)} / ${this._fmt(dur)}`;
    }
  }

  _fmt(s) {
    if (Number.isNaN(s) || !Number.isFinite(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = String(Math.floor(s % 60)).padStart(2, '0');
    return `${m}:${sec}`;
  }

  _bindKeyboard() {
    this._unbindKeyboard();
    this._boundKeydown = (e) => {
      if (!this._overlay?.open) return;
      if (e.key === 'Escape') { this._close(); return; }
      if (e.key === ' ' || e.key === 'Space') { e.preventDefault(); this._togglePlayPause(); return; }
      if (e.key === 'ArrowRight') { this._skipForward(); return; }
      if (e.key === 'ArrowLeft') this._skipBack();
    };
    document.addEventListener('keydown', this._boundKeydown);
  }

  _unbindKeyboard() {
    if (this._boundKeydown) {
      document.removeEventListener('keydown', this._boundKeydown);
      this._boundKeydown = null;
    }
  }
}
